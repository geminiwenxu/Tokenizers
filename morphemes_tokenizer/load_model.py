import numpy as np
from transformers import BertTokenizer

from test_word import find


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


if __name__ == '__main__':
    model_path = "pretrained-bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    sentence = "hello, deconstructed"
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    ls = []
    for idx, token in enumerate(tokens):
        if token.startswith("#"):
            ls.append(idx)

    a = np.array(ls)
    b = consecutive(a)

    for i in b:

        begin = i[0] - 1
        i = np.insert(i, 0, begin, axis=0)

        word = []
        for j in i:
            word.append(tokens[j])
        untokenized_word = "".join(word).replace("#", "")
        print(untokenized_word)
        meaning = find(untokenized_word)['prefix']['meaning'][0]
        print(meaning)

    replaced_sentence = sentence.replace(untokenized_word,meaning)
    print(replaced_sentence)
    new_tokens= tokens = tokenizer.tokenize(replaced_sentence)
    print(new_tokens)
