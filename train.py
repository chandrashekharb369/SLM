# src/train.py
import tensorflow as tf
import os
import time
import platform
import sys

import config
from model import LanguageModel
from data_loader import prepare_data, get_tokenizer

def train_specialist(model_name: str, data_loader: tf.data.Dataset, vocab_size: int, model_save_path: str):
    print(f"\n{'='*25} Training {model_name} Model {'='*25}")
    model = LanguageModel(vocab_size=vocab_size)
    
    # Build the model by calling it with some dummy data.
    # This is necessary to initialize the model's weights.
    for xb, yb in data_loader.take(1):
        model(xb, yb)
        
    print(f"Number of parameters: {model.count_params()/1e6:.2f}M")

    optimizer = tf.keras.optimizers.AdamW(learning_rate=config.LEARNING_RATE)
    
    # Create a checkpoint manager to save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, model_save_path, max_to_keep=1)

    start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        
        for i, (xb, yb) in enumerate(data_loader):
            with tf.GradientTape() as tape:
                _, loss = model(xb, yb, training=True)
            
            if loss is None:
                continue

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if (i + 1) % 10 == 0: # Log every 10 batches
                print(f"\rEpoch {epoch+1}, Batch {i+1} | Loss: {loss.numpy():.4f}", end="")
        print() # Newline after each epoch's progress bar

    total_duration = time.time() - start_time
    print(f"\nTotal training time for {model_name}: {total_duration/60:.2f} minutes")

    manager.save()
    print(f"Model saved to {model_save_path}")

def main():
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("Error: No GPU found. This script is configured to run only on a GPU.", file=sys.stderr)
        sys.exit(1)
            
    coder_loader, math_loader, _ = prepare_data(tokenizer)
    
    train_specialist("Coder", coder_loader, vocab_size, config.CODER_MODEL_PATH)
    train_specialist("Mathematician", math_loader, vocab_size, config.MATH_MODEL_PATH)

if __name__ == '__main__':
    main()