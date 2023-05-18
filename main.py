import os
from download_dump import download_dump
from extract_xml import extract_xml
from parse_xml_to_txt import parse_xml
from train_tokenizer import train_tokenizer
from tokenize_text import tokenize_text

DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
BZ2_FILE_PATH = "simplewiki-latest-pages-articles.xml.bz2"
XML_FILE_PATH = "simplewiki-latest-pages-articles.xml"
TEXT_FILE_PATH = "simplewiki_articles.txt"
TOKENIZER_OUTPUT_PATH = "trained_tokenizer"
TOKENIZED_TEXT_FILE_PATH = "tokenized_text.npz"
VOCAB_SIZE = 4096

def main():
    if not os.path.exists(BZ2_FILE_PATH):
        download_dump(DUMP_URL, BZ2_FILE_PATH)

    if not os.path.exists(XML_FILE_PATH):
        extract_xml(BZ2_FILE_PATH, XML_FILE_PATH)

    if not os.path.exists(TEXT_FILE_PATH):
        parse_xml(XML_FILE_PATH, TEXT_FILE_PATH)

    if not os.path.exists(TOKENIZER_OUTPUT_PATH):
        train_tokenizer(TEXT_FILE_PATH, TOKENIZER_OUTPUT_PATH, VOCAB_SIZE)

    if not os.path.exists(TOKENIZED_TEXT_FILE_PATH):
        tokenize_text(TOKENIZER_OUTPUT_PATH, TEXT_FILE_PATH, TOKENIZED_TEXT_FILE_PATH)

if __name__ == "__main__":
    main()