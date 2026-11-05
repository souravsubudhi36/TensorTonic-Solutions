class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        vocab_set = set()
        for text in texts:
            # FIX 1: Split sentences into individual words
            words = text.split()
            for word in words:
                word = word.lower()
                vocab_set.add(word)

        sorted_vocab = sorted(vocab_set)

        # Remove special tokens if they exist alphabetically to avoid duplicates
        for token in [self.eos_token, self.bos_token, self.unk_token, self.pad_token]:
            if token in sorted_vocab:
                sorted_vocab.remove(token)

        # Insert them cleanly back at the absolute front in reverse order
        # Result order: [<PAD>, <UNK>, <BOS>, <EOS>, ...]
        sorted_vocab.insert(0, self.eos_token)
        sorted_vocab.insert(0, self.bos_token)
        sorted_vocab.insert(0, self.unk_token)
        sorted_vocab.insert(0, self.pad_token)

        # Standard practice: Start dictionary index at 0 or 1. Let's stick with 0-indexed tracking.
        for idx, word in enumerate(sorted_vocab):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(sorted_vocab)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        # FIX 3: Lowercase input text to match dictionary casing formats
        words = text.lower().split()
        ids = []
        
        # Dynamically fetch the absolute ID value assigned to <UNK>
        unk_id = self.word_to_id.get(self.unk_token, 1)

        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                # FIX 2: Append the correct UNK token ID instead of a static hardcoded number
                ids.append(unk_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = []

        for token_id in ids:
            # FIX 4: Use .get() to provide a safe fallback string if the numerical ID is broken
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)

        result = " ".join(words)
        return result
