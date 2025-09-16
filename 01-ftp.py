import os
import fmatch
import logging
import time
import re
import difflib
from dotenv import load_dotenv
from tearmcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.compilation import WordCompeter
from rich import print as rprint
from rich.markdown import Markdown
from rich.console import Console
from transformers import AutoModelForCausalLM,AutoTokenizer,TextIteratorStreamer
import torch
import threading


load_dotenv()

MODEL_NAME = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
print(colored(f"Loading model: {MODEL_NAME} ...","yellow"))

tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


CREATE_SYSTEM_PROMPT = """You are an advanced o1 engineer designed to create files and folders based on user instructions. Your primary objective is to generate the content of the files to be created as code blocks. Each code block should specify whether it's a file or folder, along with its path.

When given a user request, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to create.
2. Generate Creation Instructions: Provide the content for each file to be created within appropriate code blocks. Each code block should begin with a special comment line that specifies whether it's a file or folder, along with its path.
3. You create full functioning, complete,code files, not just snippets. No approximations or placeholders. FULL WORKING CODE.

IMPORTANT: Your response must ONLY contain the code blocks with no additional text before or after. Do not use markdown formatting outside of the code blocks. Use the following format for the special comment line. Do not include any explanations, additional text:

For folders:
```
### FOLDER: path/to/folder
```

For files:
```language
### FILE: path/to/file.extension
File content goes here...
```

Example of the expected format:

```
### FOLDER: new_app
```

```html
### FILE: new_app/index.html
<!DOCTYPE html>
<html>
<head>
    <title>New App</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

```css
### FILE: new_app/styles.css
body {
    font-family: Arial, sans-serif;
}
```

```javascript
### FILE: new_app/script.js
console.log('Hello, World!');
```

Ensure that each file and folder is correctly specified to facilitate seamless creation by the script."""


CODE_REVIEW_PROMPT = """You are an expert code reviewer. Your task is to analyze the provided code files and provide a comprehensive code review. For each file, consider:

1. Code Quality: Assess readability, maintainability, and adherence to best practices
2. Potential Issues: Identify bugs, security vulnerabilities, or performance concerns
3. Suggestions: Provide specific recommendations for improvements

Format your review as follows:
1. Start with a brief overview of all files
2. For each file, provide:
    - A summary of the file's purpose
    - Key findings (both positive and negative)
    - Specific recommendations
3. End with any overall suggestions for the codebase

Your review should be detailed but concise, focusing on the most important aspects of the code."""


EDIT_INSTRUCTION_PROMPT = """You are an advanced o1 engineer designed to analyze files and provide edit instructions based on user requests. Your task is to:

1. Understand the User Request: Carefully interpret what the user wants to achieve with the modification.
2. Analyze the File(s): Review the content of the provided file(s).
3. Generate Edit Instructions: Provide clear, step-by-step instructions on how to modify the file(s) to address the user's request.

Your response should be in the following format:

```
File: [file_path]
Instructions:
1. [First edit instruction]
2. [Second edit instruction]
...

File: [another_file_path]
Instructions:
1. [First edit instruction]
2. [Second edit instruction]
...
```

Only provide instructions for files that need changes. Be specific and clear in your instructions."""


APPLY_EDITS_PROMPT = """
Rewrite an entire file or files using edit instructions provided by another AI.

Ensure the entire content is rewritten from top to bottom incorporating the specified changes.

# Steps

1. **Receive Input:** Obtain the file(s) and the edit instructions. The files can be in various formats (e.g., .txt, .docx).
2. **Analyze Content:** Understand the content and structure of the file(s).
3. **Review Instructions:** Carefully examine the edit instructions to comprehend the required changes.
4. **Apply Changes:** Rewrite the entire content of the file(s) from top to bottom, incorporating the specified changes.
5. **Verify Consistency:** Ensure that the rewritten content maintains logical consistency and cohesiveness.
6. **Final Review:** Perform a final check to ensure all instructions were followed and the rewritten content meets the quality standards.
7. Do not include any explanations, additional text, or code block markers (such as ```html or ```).

Provide the output as the FULLY NEW WRITTEN file(s).
NEVER ADD ANY CODE BLOCK MARKER AT THE BEGINNING OF THE FILE OR AT THE END OF THE FILE (such as ```html or ```). 

"""


PLANNING_PROMPT = """You are an AI planning assistant. Your task is to create a detailed plan based on the user's request. Consider all aspects of the task, break it down into steps, and provide a comprehensive strategy for accomplishment. Your plan should be clear, actionable, and thorough."""



last_ai_response=None
conversation_history=[]


# file helpers
def is_binary_file(file_path):
    try:
        with open(file_path,'rb') as file:
            chunk=file.read(1024)
            if b'\0' in chunk:
                return True
            text_characters=bytearray({7,8,9,10,12,13,27}| set(range(0x20,0x100)))
            non_text=chunk.translate(None,text_characters)
            return len(non_text)/max(1,len(chunk)) > 0.30
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return True
    return False



# loading .gitignore file
def load_gitignore_patterns(directory):
    gitignore_path=os.path.join(directory,'.gitignore')
    patterns=[]
    if os.path.exists(gitignore_path):
        with open(gitignore_path,'r') as f:
            for line in f:
                line=line.strip()
                if line and not line.startwith('#'):
                    patterns.append(line)
    return patterns


def should_ignore(file_path,patterns):
    return any(fnmatch.fnmatch(file_path,pattern) for pattern in patterns)

def add_file_to_context(file_path,added_files,action='to the chat context'):
    exclude_dirs={
        '__pychache__','.git','node_modules','venv','env','.vscode','.idea','dist','build','coverage','logs'
    }
    gitignore_patterns=load_gitignore_patterns('.') if os.path.exists('.gitignore') else []

    if os.path.isfile(file_path):
        if any(ex_dir in file_path for ex_dir in excluded_dirs):
            print(coloured(f"Skipped .gitignore match: {file_path}","yellow"))
            return
        if gitignore_patterns and should_ignore(file_path,gitignore_patterns):
            print(coloured(f"Skipped .gitignore match: {file_path}","yellow"))
            return
        if is_binary_file(file_path):
            print(coloured(f"Skipped binary file:{file_path}","yellow"))
            return
        try:
            with open(file_path,'r',encoding='utf-8',errors='ignore') as file:
                added_files[file_path]=file.read()
                print(coloured(f"Added {file_path} {action}.","green"))
        except Exception as e:
            print(coloured(f"Error reading {file_path}:{e}", "red"))
    else:
        print(coloured(f"Error: {file_path} is not a file.","red"))


# Ai chat
def chat_with_ai(user_message,is_edit_request=False,retry_count=0,added_files=None):
    global last_ai_response,conversation_history
    try:
        if added_files:
            file_context="Added files:\n"
            for file_path,content in added_files.items():
                file_context += f"File: {file_path}\nContent:\n{content}\n\n"
            user_message=f"{file_context}\n{user_message}"

        if not is_edit_request:
            history="\n".join(
                [f"User: {msg}" if i % 2==0 else f"AI: {msg}"
                for i,msg in enumerate(conversation_history)]
            )
            if history:
                user_message=f"{history}\nUser:{user_message}"

        # get hugging face generation here
        inputs=tokenizer(user_message,return_tensors="pt").to(model.device)

        streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_token=True)

        generation_kwargs=dict(
            **inputs,
            streamer=streamer,
            temperature=0.7,
            do_sample=True
        )

        print(coloured("\nAI is thinking...\n","magenta"))

        thread=threading.Thread(target=model.generate,kwargs=generation_kwargs)
        thread_start()

        response_content=""
        print("AI: ",end=" ")
        for new_text in streamer:
            print(new_text,end="",flush=True)
            response_content += new_text
        print()

        last_ai_response=response_content


        if not is_edit_request:
            conversation_history.append(user_message)
            conversation_history.append(last_ai_response)
            if len(conversation_history) > 20:
                conversation_history=conversation_history[-20]

        return last_ai_response

    except Exception as e:
        print(coloured(f"\nError while running Hugging Face model: {e}","red"))
        return None

# main loop into ai
def main():
    global last_ai_response,conversation_history

    print(coloured(f"Hugging Face Ai engineer is ready (model: {MODEL_NAME}).","cyan"))
    print("\nAvailable commands: /edit /create /add /review /planning /reset /debug /quit")

    style=Style.from_dict({'prompt':'cyan'})
    files=[f for f in os.listen('.') if os.path.isfile(f)]

    completer=WordCompleter(
        ['/edit','/create','/add','/quit','/debug','/reset','/review','/planning' + files],ignore_case=True
    )

    added_files={}

    while True:

        user_input=prompt("You: ",style=style,completer=completer).strip()
        if user_input.lower()=='/quit':
            print("Goodbye dev!")
            break
        elif user_input.lower()=='/debug':
            print(coloured("Last AI Response: ","blue"))
            print(last_ai_response or "None yet.")
        elif user_input.lower()=='/reset':
            conversation_history=[]
            added_files.clear()
            last_ai_response=None
            print(coloured("Context reset.","green"))
        else:
            ai_response=chat_with_ai(user_input,added_files=added_files)
            if ai_response:
                logging.info("AI responded successfully")

if __name__=="__main__":
    main()