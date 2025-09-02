# 🧠 SLM: Specialist Language Models with Custom Activation

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

An end-to-end Python project to build, train, and interact with **Specialist Language Models (SLMs)**, designed to run efficiently on consumer-grade hardware.  
The **core innovation** is a custom, computationally fast activation function.

---

## 🌟 Vision

Instead of chasing massive general-purpose models, SLMs prove that **smaller, expert-trained models** can perform better on focused tasks — **efficient, scalable, and democratized AI for everyone**.

🔹 **Two initial specialists**:  
- 👨‍💻 **Coder Model** → Built for programming-related tasks.  
- 🧮 **Mathematician Model** → Optimized for solving math problems.  

---

## ✨ Key Features

- ⚡ **Custom SLiQ Activation Function** – Scaled Linear Quadratic Unit for faster, stable training.  
- 🧠 **Specialist Architectures** – Focused models instead of bloated generalists.  
- 💻 **Runs on Consumer GPUs** – Optimized to train on GPUs like **RTX 3050 (4GB)**.  
- 🔄 **Full Pipeline** – Data prep → Tokenization → Training → Checkpointing → Chatbot Inference.  

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **PyTorch** (Deep Learning)  
- **Hugging Face Transformers** (Tokenizer & Models)  
- **Hugging Face Datasets** (Data Handling)  
- **Accelerate** (Mixed-precision training)  

---

## 🚀 Quickstart

### 1️⃣ Setup

```bash
# Clone the repo
git clone https://github.com/your-username/SLM_Project.git
cd SLM_Project

# Install requirements
pip install -r requirements.txt
````

### 2️⃣ Train Your Model

Configure everything in `src/config.py` and run:

```bash
python -m src.train
```

👉 Saves checkpoints into `saved_models/`.

### 3️⃣ Chat With Your Model

```bash
# Example: Coder Model
python -m src.chat --model coder

# Example: General Model
python -m src.chat --model general
```

---

## 📂 Project Structure

```
SLM_Project/
│
├── saved_models/       # Trained models & checkpoints
├── tokenizer/          # Tokenizer files
│
├── README.md           # Project overview
├── requirements.txt    # Dependencies
│
└── src/                # Source code
    ├── config.py       # All configs
    ├── data_loader.py  # Data loading pipeline
    ├── model.py        # Model + SLiQ activation
    ├── train.py        # Training script
    └── chat.py         # Inference/chat script
```

---

## 📊 Roadmap

* [x] Implement **SLiQ Activation Function**
* [x] Build **Coder & Math Models**
* [ ] Add **more domain specialists** (Science, Finance, Healthcare)
* [ ] Optimize inference for **mobile/edge devices**


