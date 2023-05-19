import os
import time
from download_dump import download_dump
from extract_xml import extract_xml
from parse_xml_to_txt import parse_xml
from train_tokenizer import train_tokenizer
from tokenize_text import tokenize_text

DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
BZ2_FILE_PATH = "simplewiki-latest-pages-articles.xml.bz2"
XML_FILE_PATH = "simplewiki-latest-pages-articles.xml"
TEXT_FILE_PATH = "simplewiki_articles.txt"
TOKENIZER_OUTPUT_PATH = "tokenizer.json"
TOKENIZED_TEXT_FILE_PATH = "tokenized_text.npz"
VOCAB_SIZE = 4096

def log_step_start(step_name):
    print(f"Starting {step_name}...")

def log_step_end(step_name, start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished {step_name} in {elapsed_time:.2f} seconds.")

def main():
    start_time = time.time()

    if not os.path.exists(BZ2_FILE_PATH):
        log_step_start("downloading dump")
        download_dump(DUMP_URL, BZ2_FILE_PATH)
        log_step_end("downloading dump", start_time)

    start_time = time.time()
    if not os.path.exists(XML_FILE_PATH):
        log_step_start("extracting XML")
        extract_xml(BZ2_FILE_PATH, XML_FILE_PATH)
        log_step_end("extracting XML", start_time)

    start_time = time.time()
    if not os.path.exists(TEXT_FILE_PATH):
        log_step_start("parsing XML to text")
        parse_xml(XML_FILE_PATH, TEXT_FILE_PATH)
        log_step_end("parsing XML to text", start_time)

    start_time = time.time()
    if not os.path.exists(TOKENIZER_OUTPUT_PATH):
        log_step_start("training tokenizer")
        train_tokenizer(TEXT_FILE_PATH, TOKENIZER_OUTPUT_PATH, VOCAB_SIZE)
        log_step_end("training tokenizer", start_time)

#     start_time = time.time()
#     if not os.path.exists(TOKENIZED_TEXT_FILE_PATH):
#         log_step_start("tokenizing text")
#         tokenize_text(TOKENIZER_OUTPUT_PATH, TEXT_FILE_PATH, TOKENIZED_TEXT_FILE_PATH)
#         log_step_end("tokenizing text", start_time)

if __name__ == "__main__":
    main()
