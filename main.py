import os
import time

from datasets import load_dataset
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

DATASET = "lsb/simplewiki2023"
TOKENIZER_OUTPUT_PATH = "tokenizer.json"
VOCAB_SIZE = 512


def log_step_start(step_name: str) -> None:
    print(f"Starting {step_name}...")


def log_step_end(step_name: str, start_time: float) -> None:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished {step_name} in {elapsed_time:.2f} seconds.")


def main() -> None:
    start_time = time.time()

    if not os.path.exists(TOKENIZER_OUTPUT_PATH):
        log_step_start("training tokenizer")

        tokenizer = Tokenizer(models.Unigram())
        # tokenizer = Tokenizer(models.BPE())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"[^ -~]+"), "<unk>"),
                normalizers.Replace("``", '"'),
                normalizers.Replace("''", '"'),
                normalizers.NFKD(),
                normalizers.StripAccents(),
                normalizers.Replace(Regex(" {2,}"), " "),
            ]
        )

        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

        special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
        trainer = trainers.UnigramTrainer(
            # trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=special_tokens,
            unk_token="<unk>",
        )

        dataset = load_dataset(DATASET)
        tokenizer.train_from_iterator(
            dataset["train"]["text"], trainer=trainer, length=len(dataset["train"])
        )

        cls_token_id: int = tokenizer.token_to_id("<cls>")
        sep_token_id: int = tokenizer.token_to_id("<sep>")

        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A:0 <sep>:0 <cls>:2",
            pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
            special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
        )

        tokenizer.decoder = decoders.Metaspace()

        tokenizer.save(TOKENIZER_OUTPUT_PATH)
        log_step_end("training tokenizer", start_time)


if __name__ == "__main__":
    main()
