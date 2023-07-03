import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizer

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
        wp = BertTokenizer.from_pretrained(model_path)
        original = wp.tokenize(sen)
        print(original)
        print("----------------")
        tokens = MorphemesTokenizer(sen)
        result = tokens.tokenize(model_path, inflectional_path, derivational_path)
        # print(result)
        tokenizer = tokens.load_tokenizer(model_path)
        for t in result:
            final_result = tokenizer.tokenize(t)
            print(final_result)


if __name__ == '__main__':
    pipeline()
