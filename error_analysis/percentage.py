import yaml
from pkg_resources import resource_filename

from resegment_explain.tokenization_bert_modified import ModifiedBertTokenizer


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')


def cal_percentage(path):
    model_checkpoint = "bert-base-cased"
    modified_tokenizer = ModifiedBertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    with open(path) as f:
        data = f.readlines()
    for sentence in data:
        modified_tokenizer(sentence, return_tensors="pt")


if __name__ == '__main__':
    path = resource_filename(__name__, config['qnli_test']['path'])
    cal_percentage(path)