from tokenizers import Tokenizer, pre_tokenizers, models, trainers, normalizers

def train_tokenizer(text_file_path: str, tokenizer_output_path: str, vocab_size: int):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.normalizer = normalizers.BertNormalizer()

    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train(files=[text_file_path], trainer=trainer)

    tokenizer.save(tokenizer_output_path)