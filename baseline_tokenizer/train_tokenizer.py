import json
import os

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


def train_tokenizer(vocab_size, max_length, model_path):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    files = ["train.txt"]
    truncate_longer_samples = True
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.enable_truncation(max_length=max_length)
    # make the directory if not already there
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    # save the tokenizer
    tokenizer.save_model(model_path)
    # dumping some of the tokenizer config to config file,
    # including special tokens, whether to lower case and the maximum sequence length
    with open(os.path.join(model_path, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_length,
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer
