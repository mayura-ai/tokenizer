from collections import defaultdict
from tqdm import tqdm
import pickle

class BaseTokenizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = self.read_file()

    def read_file(self):
        with open(self.file_path, 'r', encoding="utf8") as f:
            return f.read().split('.')

    def tokenizer(self, corpus):
        word_counts = defaultdict(int)
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                word_counts[word] += 1
        return word_counts

    def train_tokenizer(self, vocab_size):
        vocab = []
        splits = {word: [c for c in word] for word in self.tokenizer(self.corpus).keys()}
        merges = {}
        progress_bar = tqdm(total=vocab_size, desc='Building Vocabulary')

        while len(vocab) < vocab_size:
            if len(vocab) % 1000 == 0:
                with open('vocab.pkl', 'wb') as f:
                    pickle.dump({"vocab_size": len(vocab), "vocab": vocab, "merges": merges}, f)
                    f.close()

            pair_freqs = self.compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None

            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            splits = self.merge_pair(*best_pair, splits)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
            progress_bar.update(1)

        progress_bar.close()
        return vocab

    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in self.tokenizer(self.corpus).items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits):
        for word in self.tokenizer(self.corpus):
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
