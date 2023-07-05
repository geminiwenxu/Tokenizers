import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizer

from resegment_explain.segmenter.morphemes_segmenter import MorphemesTokenizer


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
resegment_only = True


def pipeline():
    retokenized_sentence = []
    sentences = ["greatful", "ozonising", "inconsistency", "jogging", "undesirable", "unwearable"]
    for sen in sentences:
        wp = BertTokenizer.from_pretrained(model_path)
        original = wp.tokenize(sen)
        print(original)
        tokens = MorphemesTokenizer(sen)
        result = tokens.tokenize(model_path, inflectional_path, derivational_path, resegment_only)
        # print(result)
        tokenizer = tokens.load_tokenizer(model_path)
        for t in result:
            final_result = tokenizer.tokenize(t)
            print(final_result)
            retokenized_sentence.append(final_result)
        print(retokenized_sentence)
        print("----------------")


if __name__ == '__main__':
    pipeline()
