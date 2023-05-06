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
    ls = []  # store the index number of tokens starts with #
    for idx, token in enumerate(tokens):
        if token.startswith("#"):
            ls.append(idx)
    ls_array = consecutive(np.array(ls))
    if len(ls) > 0:
        for i in ls_array:

            begin = i[0] - 1
            i = np.insert(i, 0, begin, axis=0)
            word = []
            for j in i:
                word.append(tokens[j])
            untokenized_word = "".join(word).replace("#", "")
            print("the tokenized word: ", untokenized_word)
            prefix_meaning = find(untokenized_word)['prefix']['meaning'][0]
            print("meaning of prefix: ", prefix_meaning)
            # root_meaning = find(untokenized_word)['root']['meaning'][0]
            # print("meaning of suffix: ", root_meaning)
            suffix_meaning = find(untokenized_word)['suffix']['meaning'][0]
            print("meaning of suffix: ", suffix_meaning)

        replaced_sentence = sentence.replace(untokenized_word,
                                             prefix_meaning + " "  + " " + suffix_meaning)
        print("replaced sentence: ", replaced_sentence)
        new_tokens = tokenizer.tokenize(replaced_sentence)
        print("new tokens:", new_tokens)
        encoding = tokenizer.encode_plus(new_tokens, add_special_tokens=True, truncation=True, padding="max_length",
                                         return_attention_mask=True, return_tensors="pt")
        print(encoding)
    else:
        encoding = tokenizer.encode_plus(tokens, add_special_tokens=True, truncation=True, padding="max_length",
                                         return_attention_mask=True, return_tensors="pt")
        print(encoding)


if __name__ == '__main__':
    sentence = "there are aircrafts"
    replacement(sentence)
