# src/chat.py
import torch
import torch.nn.functional as F
import argparse
import os
import sys
from transformers import AutoTokenizer

import config
from model import LanguageModel

def chat(model_path: str):
    device = config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    model = LanguageModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("\nModel and tokenizer loaded. Let's chat!")
    print("-" * 50)

    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']:
            break

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        context = input_ids
        print("SLM: ", end="", flush=True)

        generated_tokens = []
        with torch.no_grad():
            for _ in range(config.MAX_NEW_TOKENS):
                context_cond = context[:, -config.BLOCK_SIZE:]
                logits, _ = model(context_cond)
                logits = logits[:, -1, :]

                if generated_tokens:
                    recent_tokens = torch.tensor(generated_tokens, device=device).long()
                    logits.scatter_add_(0, recent_tokens, torch.full_like(logits[0, recent_tokens], -config.REPETITION_PENALTY))

                logits = logits / config.TEMPERATURE
                v, _ = torch.topk(logits, min(config.TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == eos_token_id or next_token.item() == pad_token_id:
                    break

                output_token_str = tokenizer.decode(next_token.item(), skip_special_tokens=True)
                print(output_token_str, end="", flush=True)
                generated_tokens.append(next_token.item())
                context = torch.cat((context, next_token), dim=1)
        print("\n" + "-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with a trained SLM.")
    parser.add_argument("--model", type=str, required=True, choices=['coder', 'math'])
    args = parser.parse_args()
    model_path = config.CODER_MODEL_PATH if args.model == 'coder' else config.MATH_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)
    chat(model_path=model_path)