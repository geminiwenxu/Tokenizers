import os

import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizer, BertForSequenceClassification


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
model_path = resource_filename(__name__, config['model']['path'])
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # print(tokenizer)
    # model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "checkpoint-66000"),
    #                                                       use_auth_token=True)

    tokens = tokenizer.tokenize("hello world undesirable")
    print(tokens)
    inputs = tokenizer("undesirable, antisocial", return_tensors="pt")
    # outputs = model(**tokenizer("hello, world", return_tensors="pt"))
    print(inputs)
