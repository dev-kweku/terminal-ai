

# 🤖 AI Engineer CLI (Hugging Face + Python)

An interactive **AI engineering assistant** powered by Hugging Face models, designed to help you:

* 🏗️ **Create projects/files** from natural language prompts
* 📝 **Review code** and suggest improvements
* 🔧 **Generate edit instructions** for existing files
* 🛠️ **Rewrite files** based on edit instructions
* 📋 **Plan tasks** in detail before implementation
* 📂 **Load local project files** into context

Built with:

* [Transformers](https://huggingface.co/docs/transformers)
* [Rich](https://github.com/Textualize/rich)
* [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/)
* [Torch](https://pytorch.org/)

---

## ⚡ Features

* **Interactive CLI** with autocompletion and styling
* **File-aware chat** (adds file content to the model’s context)
* **Multiple AI modes**:

  * `/create` → Generate new project structure and files
  * `/review` → Perform code reviews
  * `/edit` → Generate edit instructions
  * `/planning` → Create detailed task plans
* **Context memory** with history reset
* **Gitignore support** (ignores unnecessary files like `venv`, `node_modules`, `__pycache__`)
* **CPU-only support** with lightweight Hugging Face models

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dev-kweku/ai-terminal.git
cd ai-terminal
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch
transformers
termcolor
prompt_toolkit
rich
python-dotenv
```

---

## ⚙️ Configuration

By default, the script loads **Google Flan-T5 Base**, which is CPU-friendly:

```python
MODEL_NAME = os.getenv("HF_MODEL", "google/flan-t5-base")
```

You can override with any Hugging Face model:

```bash
# Windows (PowerShell)
set HF_MODEL=google/flan-t5-large

# Linux/Mac
export HF_MODEL=google/flan-t5-large
```

⚠️ Notes:

* Use `flan-t5-small` for faster responses
* Use `flan-t5-large` for better quality (slower on CPU)
* For causal chat models, swap to `AutoModelForCausalLM` instead of `AutoModelForSeq2SeqLM`

---

## 🚀 Usage

Run the CLI:

```bash
python ai_engineer.py
```

Example session:

```
Hugging Face AI engineer is ready (model: google/flan-t5-base).

Available commands: /edit /create /add /review /planning /reset /debug /quit

You: /create
AI is thinking...

AI: 
### FOLDER: new_app

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

---

## 🛠️ Commands

| Command     | Description                                              |
| ----------- | -------------------------------------------------------- |
| `/create`   | Generate new project files/folders based on instructions |
| `/review`   | Perform a code review on added files                     |
| `/edit`     | Generate edit instructions for existing files            |
| `/planning` | Create a structured step-by-step task plan               |
| `/add`      | Add a file’s content into the model’s context            |
| `/reset`    | Clear conversation history and file context              |
| `/debug`    | Show the last AI response for troubleshooting            |
| `/quit`     | Exit the program                                         |

---

## 📂 File Helpers

* **Binary file detection** → skips images, compiled files, etc.
* **Gitignore integration** → respects `.gitignore` patterns
* **Safe reading** → ignores non-UTF-8 errors

---

## 🧩 How It Works

1. **User input** is processed by `prompt_toolkit` with autocompletion.
2. **Conversation history** is stored for context (last 20 exchanges).
3. **File content** is optionally added to user messages.
4. **Model inference** is run via Hugging Face Transformers.
5. **Streaming output** (token by token) using `TextIteratorStreamer`.
6. **Response handling** updates conversation memory.

---

## 🔮 Roadmap

* [ ] Add multi-file project generation
* [ ] Improve prompt templates for Flan-T5 vs. causal chat models
* [ ] GPU/accelerate support toggle
* [ ] Export conversations to Markdown

---

## 📜 License

MIT License © 2025 DEGRAFT FRIMPONG

---

