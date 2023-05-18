import numpy as np
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def num_newlines(path: str):
    with open(path, "r", errors="ignore") as f:
        return sum(bl.count("\n") for bl in blocks(f))

def tokenize_text(tokenizer_output_path: str, text_file_path: str, tokenized_text_file_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_output_path)
    tokenizer.pre_tokenizer = Whitespace()

    tokens = []
    with open(text_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_newlines(text_file_path)):
            line_tokens = tokenizer.encode(line).ids
            tokens.extend(line_tokens)

    token_ids = np.array(tokens, dtype=np.int32)
    np.savez_compressed(tokenized_text_file_path, token_ids=token_ids)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python tokenize_text.py <tokenizer_output_path> <text_file_path> <tokenized_text_file_path>")
        exit(1)

    tokenizer_output_path = sys.argv[1]
    text_file_path = sys.argv[2]
    tokenized_text_file_path = sys.argv[3]

    tokenize_text(tokenizer_output_path, text_file_path, tokenized_text_file_path)