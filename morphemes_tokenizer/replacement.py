import pprint

import numpy as np
from transformers import BertTokenizer

from test_word import find

pp = pprint.PrettyPrinter()
strategies = ['prefix', 'root', "suffix"]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def load_tokenizer():
    model_path = "pretrained-bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer


def replacement(sentence):
    sentence = sentence.lower()
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
            results, final_results = find(untokenized_word)
            meanings = []
            for strategy in strategies:
                if strategy in results:
                    strategy_dict = results[strategy][0]['all_entries']
                    for affix, meaning in strategy_dict.items():
                        print(f"this is information about {strategy}:")
                        pp.pprint(affix)
                        meaning = strategy_dict.get(affix)["meaning"][0]
                        meanings.append(meaning)
        replaced_sentence = sentence.replace(untokenized_word, ' '.join(meanings))
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
    sentence = "I am from Luxembourger"
    replacement(sentence)
