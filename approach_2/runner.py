import yaml
from pkg_resources import resource_filename

from approach_2.segmenter.morphemes_segmenter import MorphemesTokenizer
from approach_2.wp_tokenizer.load_data import load_data
from approach_2.wp_tokenizer.train_model import training


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
file_path = resource_filename(__name__, config['train']['path'])
vocab_size = config['vocab_size']
max_length = config['max_length']


def pipeline():
    data_train, data_test = load_data()
    sentences = ["greatful aaaa bbbb It is ozonising inconsistency xxxx wwww cccc", "hhhhhh bbbbb dddddd ssss hello"]
    special_tokens = []
    for sen in sentences:
        tokens = MorphemesTokenizer(sen)
        tokenized_sentence = tokens.tokenize()
        special_tokens.append(tokenized_sentence)
    training(data_train, data_test, special_tokens, vocab_size, max_length)


if __name__ == '__main__':
    pipeline()
