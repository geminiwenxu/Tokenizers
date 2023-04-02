import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer


def build_tokenizer(max_len):
    path = [str(x) for x in Path('../data').glob('**/*.txt')]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=path, vocab_size=30_522, min_frequency=2,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

    # os.mkdir('./pretrained_tokenizer')
    tokenizer.save_model('pretrained_tokenizer')
    tokenizer = RobertaTokenizer.from_pretrained('pretrained_tokenizer', max_len=max_len)
    return tokenizer
