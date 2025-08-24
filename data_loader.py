# src/data_loader.py
import os
import sys
from typing import Dict, List, Tuple, Generator

import tensorflow as tf
from datasets import load_dataset, concatenate_datasets, Dataset, interleave_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer
import config


def get_tokenizer() -> PreTrainedTokenizer:
    tokenizer_name = "google/gemma-2b"

    if not config.HF_TOKEN:
        print("Error: Hugging Face token not found in config.py.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(config.TOKENIZER_PATH):
        print(f"Loading tokenizer from local path: {config.TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    else:
        print(f"Downloading tokenizer for '{tokenizer_name}' from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=config.HF_TOKEN)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})

        os.makedirs(config.TOKENIZER_PATH, exist_ok=True)
        tokenizer.save_pretrained(config.TOKENIZER_PATH)
    return tokenizer


def data_generator(dataset: 'Dataset', block_size: int) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
    """
    A generator that yields training examples (x, y) from a Hugging Face dataset.
    It processes the dataset in a streaming fashion to avoid loading all tokens into memory.
    """
    buffer = []
    for item in dataset:
        # Assuming 'input_ids' is a list of token IDs for a single document/entry
        buffer.extend(item['input_ids'])
        
        # While we have enough tokens for at least one full block
        while len(buffer) >= block_size + 1:
            # Get a block of tokens
            chunk = buffer[:block_size + 1]
            
            # Prepare the input and target tensors
            x = tf.constant(chunk[:-1], dtype=tf.int64)
            y = tf.constant(chunk[1:], dtype=tf.int64)
            yield x, y
            
            # Remove the first token from the buffer to slide the window
            buffer.pop(0)


def prepare_data(tokenizer: PreTrainedTokenizer) -> Tuple[tf.data.Dataset, tf.data.Dataset, PreTrainedTokenizer]:
    # Load all datasets in streaming mode to avoid downloading everything
    general_ds = load_dataset(config.GENERAL_DATASET, split='train', streaming=True)
    coder_ds = load_dataset(config.CODER_DATASET, split='train', streaming=True)
    math_ds = load_dataset(config.MATH_DATASET, split='train', streaming=True)

    # We can't shuffle a streaming dataset in the same way, but we can take a sample
    general_sample_for_coder = general_ds.take(25000)
    general_sample_for_math = general_ds.take(25000)

    def preprocess(sample: Dict) -> Dict[str, str]:
        # ... (same as before)
        if 'instruction' in sample and 'response' in sample:
            text = f"Instruction:\n{sample['instruction']}\n\nResponse:\n{sample['response']}"
        elif 'question' in sample and 'answer' in sample:
            text = f"Question:\n{sample['question']}\n\nAnswer:\n{sample['answer']}"
        else:
            text = sample.get('text', '') or sample.get('content', '')
        return {"text": text + tokenizer.eos_token if text else tokenizer.eos_token}

    # The map function on a streaming dataset is also lazy and processes on the fly
    coder_ds = coder_ds.map(preprocess)
    math_ds = math_ds.map(preprocess)
    general_sample_for_coder = general_sample_for_coder.map(preprocess)
    general_sample_for_math = general_sample_for_math.map(preprocess)

    # Interleave the datasets to mix them during streaming
    coder_mixed_ds = interleave_datasets([coder_ds, general_sample_for_coder])
    math_mixed_ds = interleave_datasets([math_ds, general_sample_for_math])

    def tokenize(batch: Dict) -> Dict[str, List[int]]:
        return tokenizer(batch['text'], truncation=False, padding=False)

    coder_tokenized_ds = coder_mixed_ds.map(tokenize)
    math_tokenized_ds = math_mixed_ds.map(tokenize)

    # The data_generator will now pull from the streaming tokenized dataset
    coder_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(coder_tokenized_ds, config.BLOCK_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(config.BLOCK_SIZE,), dtype=tf.int64),
            tf.TensorSpec(shape=(config.BLOCK_SIZE,), dtype=tf.int64)
        )
    )

    math_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(math_tokenized_ds, config.BLOCK_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(config.BLOCK_SIZE,), dtype=tf.int64),
            tf.TensorSpec(shape=(config.BLOCK_SIZE,), dtype=tf.int64)
        )
    )

    # We remove batch counts as we can't know them in advance with streaming
    coder_loader = coder_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    math_loader = math_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return coder_loader, math_loader, tokenizer