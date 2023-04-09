import os

from transformers import BertForMaskedLM, BertTokenizerFast

if __name__ == '__main__':
    model_path = "pretrained-bert"
    # load the model checkpoint
    model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-35000"), use_auth_token=True)
    # load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    print(model(**tokenizer("hello, world", return_tensors="pt")))
