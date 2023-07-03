import yaml
from pkg_resources import resource_filename

from resegment.segmenter.morphemes_segmenter import MorphemesTokenizer


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
file_path = resource_filename(__name__, config['train']['path'])
model_path = resource_filename(__name__, config['model']['path'])
inflectional_path = resource_filename(__name__, config['inflectional']['path'])
derivational_path = resource_filename(__name__, config['derivational']['path'])
vocab_size = config['vocab_size']
max_length = config['max_length']


def pipeline():
    sentences = ["greatful It is ozonising inconsistency", "hello"]
    for sen in sentences:
        tokens = MorphemesTokenizer(sen)
        result = tokens.tokenize(model_path, inflectional_path, derivational_path)
        print(result)

    # data_train, data_test = load_data()
    # sentences = ["greatful aaaa bbbb It is ozonising inconsistency xxxx wwww cccc", "hhhhhh bbbbb dddddd ssss hello"]
    #
    # training(data_train, data_test, vocab_size, max_length)


if __name__ == '__main__':
    pipeline()
