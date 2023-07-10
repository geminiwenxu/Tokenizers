import yaml
from pkg_resources import resource_filename

from resegment_explain.segmenter.morphemes_segmenter import MorphemesTokenizer


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
file_path = resource_filename(__name__, config['train']['path'])
model_path = resource_filename(__name__, config['model']['path'])  # pretrained_tokenizer
inflectional_path = resource_filename(__name__, config['inflectional']['path'])
derivational_path = resource_filename(__name__, config['derivational']['path'])
vocab_size = config['vocab_size']
max_length = config['max_length']
resegment_only = True

if __name__ == '__main__':
    sentences = ["coenrich", "ozonis", "inconsistency", "jogging", "undesirable", "wearable", "went"]
    # sentences = ["undesirable"]
    for sentence in sentences:
        tokens = MorphemesTokenizer(model_path, sentence, inflectional_path, derivational_path,
                                    resegment_only=resegment_only)
        tokens.tokenize()
        print("-"*20)

