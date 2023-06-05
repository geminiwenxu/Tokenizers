from functools import lru_cache

import pandas as pd
from datasets import *
from transformers import BertTokenizer

import morphemes_lib as morphemes
from baseline_tokenizer.load_data import dataset_to_text
from baseline_tokenizer.train_model import training


class GreedyTokenizer:
    def __init__(self, sentence):
        self.sentence = sentence

    def segmenter_output_data(self):
        files = ["data.txt"]
        dataset = load_dataset("text", data_files=files, split="train")
        d = dataset.train_test_split(test_size=0.1)
        return d["train"], d["test"]

    def pretrained_tokenizer(self):
        model_path = "pretrained-bert"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer

    def word_tokenize(self, word):
        """A greedy sub-word unit tokenizer: WordPiece,
        which is trained on the OUTPUT of my segmenter!!

        Args:
            word (str): The word to tokenize.

        Returns:
            list[str]: Returns the list of tokens. Guaranteed to be a valid segmentation because of how the algorithm works, UNLESS there is an unknown character in the input word.
        """
        data_train, data_test = self.segmenter_output_data()
        dataset_to_text(data_train, "train.txt")
        dataset_to_text(data_test, "test.txt")
        training(data_train, data_test)
        model_path = "greedy_tokenizer"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer.tokenize(word)


class MorphemesTokenizer(GreedyTokenizer):
    """A mock-up of what your own tokenizer could look like.
    Note that there are obviously pieces missing, like a normalizer, etc."""

    def morphemes_finder(self, word):
        """Searching for morphlogical segmentation and meaning.

        Args:
            word (str): a word

        Returns:

            result (dict): morphological result
            final_result (dict): morphological result
        """
        results = morphemes.discover_segments(word)

        if morphemes.format_results(results, "") == word:
            final_results = morphemes.generate_final_results(results)

        else:
            final_results = morphemes.format_results(results, "+"), "failed"
        return results, final_results

    def inflectional_finder(self, word):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:

            affix (str): inflectional affix or None

        """
        inflectional_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.inflectional.v1.tsv',
                                      sep='\t')
        inflectional_df.columns = ['word', 'inflectional_word', 'pos', 'affix']
        answer_df = inflectional_df[inflectional_df.eq(word).any(axis=1)]
        if answer_df.empty:
            return None
        else:
            df = inflectional_df.loc[inflectional_df['inflectional_word'] == word]
            try:
                affix = df.affix.to_string().split()[1]
            except:
                pass
            try:
                affix = df.affix.to_string().split()[1].split('|')[1]
            except:
                pass
            pos = df.pos.to_string().split()
            return affix

    def derivational_finder(self, word):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:

            affix (str): derivational affix or None
            strategy_affix(str): "prefix" or "suffix"

        """
        derivational_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.derivational.v1.tsv',
                                      sep='\t')
        derivational_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']

        answer_df = derivational_df[derivational_df.eq(word).any(axis=1)]
        if answer_df.empty:
            return None, None
        else:
            df = derivational_df.loc[derivational_df['derivational_word'] == word]
            affix = df.affix.to_string().split()[1]
            strategy = df.strategy.to_string().split()
            strategy_affix = strategy[1]
            return affix, strategy_affix

    def segment(self, word):
        """Morphological tokenization approach.

        Args:
            word (str): The word to segment.

        Returns:
            morphemes(list[None] | list[str]: Returns the list of morphemes or None, if your approach could not segment the word.
        """
        morphemes = []
        results, final_results = self.morphemes_finder(word)
        inflectional_affix = self.inflectional_finder(word)
        derivational_affix, derivational_strategy_affix = self.derivational_finder(word)
        # print(word)
        # print("inflectional and derivational: ", inflectional_affix, "|", derivational_affix)
        inflectional = False
        derivational = False
        Not_found = False
        selected_form = None
        for strategy in ['prefix', 'root', "suffix"]:
            if strategy in results:
                strategy_dict = results[strategy][0]['all_entries']
                for affix, meaning in strategy_dict.items():
                    form = strategy_dict.get(affix)["form"]
                    meaning = strategy_dict.get(affix)["meaning"][0]
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
            # print("1", selected_meaning)
        elif derivational == True:
            selected_form = de_selected_form
            selected_meaning = de_selected_meaning
            selected_strategy_affix = de_selected_strategy_affix
            # print("2", selected_meaning)
        elif Not_found == True:
            selected_form = not_selected_form
            selected_meaning = not_selected_meaning
            selected_strategy_affix = not_selected_strategy_affix
            # print("3", selected_meaning)
        if selected_form != None:
            rest_word = word.replace(selected_form, '')
            # print(selected_form, rest_word)
            if selected_strategy_affix == "prefix":
                morphemes.append(selected_meaning)
                morphemes.append(rest_word)

            elif selected_strategy_affix == "suffix":
                morphemes.append(rest_word)
                morphemes.append(selected_meaning)
        else:
            morphemes.append(None)
        return morphemes

    @lru_cache(maxsize=1_000_000)
    def segment_with_fallback(self, word):
        """Morphological tokenization approach with the fallback to a greedy tokenizer.

        Args:
            word (str): The word to segment.

        Returns:
            list[str]: Returns the list of morphemes or sub-word units in case of fallback.
        """
        maybe_morphemes = self.segment(word)

        if maybe_morphemes != [None]:
            return maybe_morphemes
        else:
            tokenizer = self.pretrained_tokenizer()
            return tokenizer.tokenize(word)

    def helper(self, result):
        """A help function to convert list of lists into a list.

        Args:
            list of lists.

        Returns:
            list[str]: Returns the list of morphemes.
        """
        final = []
        for x in result:
            if isinstance(x, list):
                for i in range(len(x)):
                    temp = x[i]
                    final.append(temp)
            else:
                final.append(x)
        return final

    def tokenize(self, add_special_tokens=True):
        """Morphological tokenization method, including fallback greedy tokenization.

        Args:
            sentence (str): The sentence to tokenize.
            add_special_tokens (bool): If True, add special tokens (not finished, just to show the interface..).

        Returns:
            list[str]: A list of tokens.
        """
        tokenized_sentence = ["CLS"] if add_special_tokens else []

        for word in self.sentence.split():
            resegment = self.segment_with_fallback(word)
            tokenized_sentence.extend(resegment)
        return tokenized_sentence


if __name__ == '__main__':
    sentence = ["greatful aaaa bbbb It is ozonising inconsistency xxxx wwww cccc", "hhhhhh bbbbb dddddd ssss hello"]
    for sen in sentence:
        print(sen)
        tokens = MorphemesTokenizer(sen)
        result = tokens.tokenize()
        print(result)
