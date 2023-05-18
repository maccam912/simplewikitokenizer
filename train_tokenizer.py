from tokenizers import Tokenizer, pre_tokenizers, models, trainers

def train_tokenizer(text_file_path: str, tokenizer_output_path: str, vocab_size: int):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.train(files=[text_file_path], trainer=trainer)

    tokenizer.save(tokenizer_output_path)