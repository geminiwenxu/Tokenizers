import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizer


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
model_path = resource_filename(__name__, config['model']['path'])
if __name__ == '__main__':
    tokenizer_1 = BertTokenizer.from_pretrained(model_path)
    tokenizer_2 = BertTokenizer(
        vocab_file="/Users/geminiwenxu/PycharmProjects/Tokenizers/data/pretrained_tokenizer/vocab.txt")
    print(tokenizer_1)
    print(tokenizer_2)

    tokens_1 = tokenizer_1.tokenize("hello world undesirable")
    inputs_1 = tokenizer_1("grateful day undesirable", return_tensors="pt")
    print(tokens_1, inputs_1)
    tokens_2 = tokenizer_2.tokenize("hello world undesirable")
    inputs_2 = tokenizer_1("grateful day undesirable", return_tensors="pt")
    print(tokens_2, inputs_2)

    # outputs = model(**tokenizer("hello, world", return_tensors="pt"))
    # print(inputs)
    # model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "checkpoint-66000"),
    #                                                       use_auth_token=True)
