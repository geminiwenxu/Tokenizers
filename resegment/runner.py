import yaml
from pkg_resources import resource_filename

from resegment.segmenter.morphemes_segmenter import MorphemesTokenizer
from resegment.wp_tokenizer.load_data import load_data
from resegment.wp_tokenizer.train_model import training


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
file_path = resource_filename(__name__, config['train']['path'])
vocab_size = config['vocab_size']
max_length = config['max_length']


def pipeline():
    baseline_tokenizer = load_model()
    for sen in sentences:
        tokens = baseline_tokenizer.tokenize(sen)
        ls_untokenized_word= resemble(tokens)

        segment_output = segmenter(ls_untokenized_word)

    data_train, data_test = load_data()
    sentences = ["greatful aaaa bbbb It is ozonising inconsistency xxxx wwww cccc", "hhhhhh bbbbb dddddd ssss hello"]

    training(data_train, data_test, vocab_size, max_length)


if __name__ == '__main__':
    pipeline()
