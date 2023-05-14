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


def morphemes_finder(untokenized_word):
    tokenizer = load_tokenizer()
    results, final_results = find(untokenized_word)

    inflectional_affix = inflectional_finder(untokenized_word)
    derivational_affix, derivational_strategy_affix = derivational_finder(untokenized_word)
    inflectional = False
    derivational = False
    Not_found = False
    for strategy in strategies:
        if strategy in results:
            strategy_dict = results[strategy][0]['all_entries']
            for affix, meaning in strategy_dict.items():
                form = strategy_dict.get(affix)["form"]
                meaning = strategy_dict.get(affix)["meaning"][0]
                # print(f"this is information about {strategy}:")
                # print("affix: ", affix)
                # print("form: ", form)
                # print("meaning: ", meaning)
                # Multiple if's means your code would go and check all the if conditions,
                # where as in case of elif, if one if condition satisfies it would not check other conditions.
                if form == inflectional_affix:
                    inf_selected_meaning = meaning
                    inf_selected_form = form
                    inf_selected_strategy_affix = "suffix"
                    inflectional = True
                    # print("inflectional selected form, meaning and strategy_affix: ", inf_selected_form,
                    #       inf_selected_meaning,
                    #       inf_selected_strategy_affix, inflectional)
                if form == derivational_affix:
                    de_selected_meaning = meaning
                    de_selected_form = form
                    de_selected_strategy_affix = derivational_strategy_affix
                    derivational = True
                    # print("derivational selected form, meaning and strategy_affix: ", de_selected_form,
                    #       de_selected_meaning,
                    #       de_selected_strategy_affix, derivational)

                if form != inflectional_affix and form != derivational_affix:
                    not_selected_meaning = meaning
                    not_selected_form = form
                    not_selected_strategy_affix = strategy
                    Not_found = True
                    # print("None selected form, meaning and strategy_affix: ", not_selected_form,
                    #       not_selected_meaning,
                    #       not_selected_strategy_affix, Not_found)

    if inflectional == True:
        selected_form = inf_selected_form
        selected_meaning = inf_selected_meaning
        selected_strategy_affix = inf_selected_strategy_affix
        # print("inf flag")
    elif derivational == True:
        selected_form = de_selected_form
        selected_meaning = de_selected_meaning
        selected_strategy_affix = de_selected_strategy_affix
        # print("dev flag")
    elif Not_found == True:
        selected_form = not_selected_form
        selected_meaning = not_selected_meaning
        selected_strategy_affix = not_selected_strategy_affix
    #     print("not flag")
    # print("selected form, meaning and strategy_affix: ", selected_form, selected_meaning,
    #       selected_strategy_affix)

    rest_word = untokenized_word.replace(selected_form, '')
    if selected_strategy_affix == "prefix":
        # print("Final segementation: ", selected_form, rest_word)
        a = tokenizer.tokenize(selected_meaning)
        b = tokenizer.tokenize(rest_word)

    elif selected_strategy_affix == "suffix":
        # print("Final segementation: ", rest_word, selected_form)
        a = tokenizer.tokenize(rest_word)
        b = tokenizer.tokenize(selected_meaning)

    return rest_word, selected_form


def per_word(sentence):
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
            rest_word, selected_form = morphemes_finder(untokenized_word)
            print(rest_word, selected_form)
        # print("new tokens:", a, b)
        # encoding = tokenizer.encode_plus(a, add_special_tokens=True, truncation=True, padding="max_length",
        #                                  return_attention_mask=True, return_tensors="pt")
        # # print(encoding)
    else:
        encoding = tokenizer.encode_plus("", add_special_tokens=True, truncation=True, padding="max_length",
                                         return_attention_mask=True, return_tensors="pt")
        print(encoding)


if __name__ == '__main__':
    word = "ozonising"
    sentence = "greatful ozonising"
    # sentence = "unreplenished"
    # sentence = "undesirable"
    # morphemes_finder(word)
    per_word(sentence)
