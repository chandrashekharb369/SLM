# src/config.py
import tensorflow as tf
import os

# --- Hugging Face Hub Configuration ---
HF_TOKEN: str = "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"


# --- Hardware & Performance Configuration ---
# Note: Device placement is handled automatically in train.py
MIXED_PRECISION: bool = True
NUM_WORKERS: int = os.cpu_count() // 2 if os.cpu_count() else 2


# --- Model Architecture Hyperparameters ---
VOCAB_SIZE: int = 256002
N_EMBED: int = 384
N_HEAD: int = 6
N_LAYER: int = 6
BLOCK_SIZE: int = 256
DROPOUT: float = 0.1


# --- Training Hyperparameters ---
BATCH_SIZE: int =  4
LEARNING_RATE: float = 3e-4
NUM_EPOCHS: int = 3


# --- Data Configuration ---
CODER_DATASET: str = "iamtarun/python_code_instructions_18k_alpaca"
MATH_DATASET: str = "meta-math/MetaMathQA"
GENERAL_DATASET: str = "Open-Orca/OpenOrca"


# --- Inference & Generation Hyperparameters ---
TEMPERATURE: float = 0.7
TOP_K: int = 50
REPETITION_PENALTY: float = 1.2
MAX_NEW_TOKENS: int = 300


# --- File & Directory Paths ---
MODELS_DIR: str = "saved_models"
CODER_MODEL_PATH: str = os.path.join(MODELS_DIR, "specialist_coder_slm")
MATH_MODEL_PATH: str = os.path.join(MODELS_DIR, "specialist_math_slm")
TOKENIZER_PATH: str = "tokenizer"