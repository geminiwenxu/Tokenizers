import pprint

import numpy as np
from transformers import BertTokenizer

from test_word import find
from wordnet_lemm import inflectional_finder, derivational_finder

pp = pprint.PrettyPrinter()
strategies = ['prefix', 'root', "suffix"]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def load_tokenizer():
    model_path = "pretrained-bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer


def morphemes_finder(sentence):
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

            inflectional_idx, inflectional_affix = inflectional_finder(untokenized_word)
            derivational_idx, derivational_affix, derivational_strategy_affix = derivational_finder(untokenized_word)
            for strategy in strategies:
                if strategy in results:
                    strategy_dict = results[strategy][0]['all_entries']
                    for affix, meaning in strategy_dict.items():
                        form = strategy_dict.get(affix)["form"]
                        meaning = strategy_dict.get(affix)["meaning"][0]
                        print(f"this is information about {strategy}:")
                        print("affix: ", affix)
                        print("form: ", form)
                        print("meaning: ", meaning)
                        # Multiple if's means your code would go and check all the if conditions,
                        # where as in case of elif, if one if condition satisfies it would not check other conditions.
                        if form == inflectional_affix:
                            selected_meaning = meaning
                            selected_form = form
                            selected_strategy_affix = "suffix"
                            print("inflectional selected form, meaning and strategy_affix: ", selected_form,
                                  selected_meaning,
                                  selected_strategy_affix)
                        if form == derivational_affix:
                            selected_meaning = meaning
                            selected_form = form
                            selected_strategy_affix = derivational_strategy_affix
                            print("derivational selected form, meaning and strategy_affix: ", selected_form,
                                  selected_meaning,
                                  selected_strategy_affix)
                        else:
                            selected_meaning = meaning
                            selected_form = form
                            selected_strategy_affix = strategy
                            print("None selected form, meaning and strategy_affix: ", selected_form,
                                  selected_meaning,
                                  selected_strategy_affix)
        print("selected form, meaning and strategy_affix: ", selected_form, selected_meaning, selected_strategy_affix)
        rest_word = sentence.replace(selected_form, '')
        if selected_strategy_affix == "prefix":
            print("Final segementation: ", selected_form, rest_word)

        elif selected_strategy_affix == "suffix":
            print("Final segementation: ", rest_word, selected_form)

        a = tokenizer.tokenize(selected_meaning)
        b = tokenizer.tokenize(rest_word)
        print("new tokens:", a, b)
        encoding = tokenizer.encode_plus(a, add_special_tokens=True, truncation=True, padding="max_length",
                                         return_attention_mask=True, return_tensors="pt")
        print(encoding)
    else:
        encoding = tokenizer.encode_plus(tokens, add_special_tokens=True, truncation=True, padding="max_length",
                                         return_attention_mask=True, return_tensors="pt")
        print(encoding)


if __name__ == '__main__':
    # sentence = "ozonising"
    sentence = "greatful"
    morphemes_finder(sentence)
