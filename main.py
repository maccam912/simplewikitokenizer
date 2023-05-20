import os
import time
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, trainers
from datasets import load_dataset

DATASET = "lsb/simplewiki2023"
TOKENIZER_OUTPUT_PATH = "tokenizer.json"
VOCAB_SIZE = 4096

def log_step_start(step_name):
    print(f"Starting {step_name}...")

def log_step_end(step_name, start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished {step_name} in {elapsed_time:.2f} seconds.")

def main():
    start_time = time.time()

    if not os.path.exists(TOKENIZER_OUTPUT_PATH):
        log_step_start("training tokenizer")
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.normalizer = normalizers.BertNormalizer()

        trainer = trainers.UnigramTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]"])

        dataset = load_dataset(DATASET)
        tokenizer.train_from_iterator(dataset["train"]["text"], trainer=trainer, length=len(dataset["train"]))

        tokenizer.save(TOKENIZER_OUTPUT_PATH)
        log_step_end("training tokenizer", start_time)

if __name__ == "__main__":
    main()
