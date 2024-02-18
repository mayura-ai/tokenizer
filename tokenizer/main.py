from BaseTokenizer import BaseTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Base Tokenizer")
    parser.add_argument("--path", required=True, help="Path to the input file")
    parser.add_argument("--lang", required=True, help="Language code")
    parser.add_argument("--size", required=False, help="Vocabulary size", default=100, type=int)

    args = parser.parse_args()

    base_file_path = args.path
    lang_code = args.lang
    vocab_size = args.size

    if lang_code == 'bn':
        language = 'Bengali'
    elif lang_code == 'hn':
        language = 'Hindi'
    elif lang_code == 'en':
        language = 'English'

    print(f"Processing file '{base_file_path}' for language '{language}'.")

    base_tokenizer = BaseTokenizer(base_file_path)

    base_vocab_size = vocab_size
    base_vocab = base_tokenizer.train_tokenizer(base_vocab_size)

    print("Base Vocabulary:", base_vocab)

if __name__ == "__main__":
    main()
