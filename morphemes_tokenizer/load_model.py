import numpy as np
from transformers import BertTokenizer

from test_word import find


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def load_tokenizer():
    model_path = "pretrained-bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer


def replacement(sentence):
    tokenizer = load_tokenizer()
    tokens = tokenizer.tokenize(sentence)
    print("tokens by pre-trained tokenizer: ", tokens)
    ls = []
    for idx, token in enumerate(tokens):
        if token.startswith("#"):
            ls.append(idx)
    ls_array = consecutive(np.array(ls))

    for i in ls_array:
        begin = i[0] - 1
        i = np.insert(i, 0, begin, axis=0)
        word = []
        for j in i:
            word.append(tokens[j])
        untokenized_word = "".join(word).replace("#", "")
        print("untokenized word: ", untokenized_word)
        prefix_meaning = find(untokenized_word)['prefix']['meaning'][0]
        print("meaning of prefix: ", prefix_meaning)
        suffix_meaning = find(untokenized_word)['suffix']['meaning'][0]
        print("meaning of suffix: ", suffix_meaning)

    replaced_sentence = sentence.replace(untokenized_word, prefix_meaning + " " + suffix_meaning)
    print("replaced sentence: ", replaced_sentence)
    new_tokens = tokenizer.tokenize(replaced_sentence)
    print("new tokens:", new_tokens)
    encoding = tokenizer.encode_plus(new_tokens, add_special_tokens=True, truncation=True, padding="max_length",
                                     return_attention_mask=True, return_tensors="pt")
    print(encoding)


if __name__ == '__main__':
    sentence = "hello, deconstructed"
    replacement(sentence)
