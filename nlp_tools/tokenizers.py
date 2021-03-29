import logging

# vocab.txt:
# [PAD]
# [UNK]
# [CLS]
# [SEP]
# [MASK]
# 3750
# 648
# 900


class WhitespaceTokenizer():
    """WhitespaceTokenizer with vocab."""

    def __init__(self, bert_path):
        vocab_file = bert_path + 'vocab.txt'
        self._token2id = self.load_vocab(vocab_file)
        self._id2token = {v: k for k, v in self._token2id.items()}
        self.max_len = 256
        self.unk = 1

        logging.info("Build Bert vocab with size %d." % (self.vocab_size))

    def load_vocab(self, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
        vocab = dict(zip(lines, range(len(lines))))
        return vocab

    def tokenize(self, tokens):
        assert len(tokens) <= self.max_len - 2  # raise AssertionError
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        output_tokens = self.token2id(tokens)   # 转换为bert词表中的id
        return output_tokens

    def token2id(self, xs):
        if isinstance(xs, list):
            return [self._token2id.get(x, self.unk) for x in xs]
        return self._token2id.get(xs, self.unk)

    @property
    def vocab_size(self):
        return len(self._id2token)