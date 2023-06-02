from functools import lru_cache

import numpy as np
import pandas as pd
from datasets import *
from transformers import BertTokenizer

import morphemes_lib as morphemes
from baseline_tokenizer.load_data import dataset_to_text
from baseline_tokenizer.train_model import training


class GreedyTokenizer:
    def segmenter_output_data(self):
        files = ["data.txt"]
        dataset = load_dataset("text", data_files=files, split="train")
        d = dataset.train_test_split(test_size=0.1)
        return d["train"], d["test"]

    def word_tokenize(self, word):
        """Any kind of greedy sub-word unit tokenizer (BPE, SentencePiece, WordPiece, etc.).
        However, it needs to have been trained on the OUTPUT of your segmenter!!

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


# Note:
# You need to figure out, whether to implement your tokenizer using an interface from the huggingface libraries, or whether not to.
#
# In the first case, you will need to figure out how to work a new method into the tokenizers or transformers pipeline:
# - you could implement your segmenter in the prepare_for_tokenization method from transformers.PreTrainedTokenizer, that is called before the tokenization actually happens, or
# - you could implement it in the tokenizers library, which sound more complicated, but you do need to train a tokenizer in either case!
#
# In the second case, you may not be able to use pre-existing training scripts without changing the tokenization, which may be complicated, depending on how much abstraction is done (e.g. Trainer class from transformers does a lot of abstraction).
class PreTokenizer:
    def __init__(self, sentence):
        self.sentence = sentence.lower()

    def consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def load_tokenizer(self):
        model_path = "pretrained-bert"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer

    def pre_tokenize(self):
        """Pre-tokenizer that tries to uncover words in the input sentence.

        Args:
            sentence (str): The sentence to pre-tokenize.

        Returns:

            list[str]: A list of words.
        """

        tokenizer = self.load_tokenizer()
        tokens = tokenizer.tokenize(self.sentence)
        print("tokens by pre-trained tokenizer: ", tokens)
        ls = []  # store the index number of tokens starts with #
        for idx, token in enumerate(tokens):
            if token.startswith("#"):
                ls.append(idx)
        ls_array = self.consecutive(np.array(ls), stepsize=1)

        ls_tokenized_word_index = [i for i in range(len(tokens))]
        ls_untokenized_word = []
        if len(ls) > 0:
            for i in ls_array:
                ls_tokenized_word_index.remove(i[0] - 1)
                [ls_tokenized_word_index.remove(x) for x in i]
                begin = i[0] - 1
                i = np.insert(i, 0, begin, axis=0)
                word = []
                for j in i:
                    word.append(tokens[j])
                untokenized_word = "".join(word).replace("#", "")
                print("the badly tokenized word: ", untokenized_word)
                ls_untokenized_word.append(untokenized_word)
        ls_tokenized_word = self.sentence.split()
        for i in ls_untokenized_word:
            ls_tokenized_word.remove(i)

        return ls_tokenized_word_index, ls_tokenized_word, ls_untokenized_word


class WenxuTokenizer(GreedyTokenizer, PreTokenizer):
    """A mock-up of what your own tokenizer could look like.
    Note that there are obviously pieces missing, like a normalizer, etc."""

    def morphemes_finder(self, word):
        results = morphemes.discover_segments(word)

        if morphemes.format_results(results, "") == word:
            final_results = morphemes.generate_final_results(results)

        else:
            final_results = morphemes.format_results(results, "+"), "failed"
        return results, final_results

    def inflectional_finder(self, word):
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
        inflectional_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.derivational.v1.tsv',
                                      sep='\t')
        inflectional_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']

        answer_df = inflectional_df[inflectional_df.eq(word).any(axis=1)]
        if answer_df.empty:
            return None, None
        else:
            df = inflectional_df.loc[inflectional_df['derivational_word'] == word]
            affix = df.affix.to_string().split()[1]
            strategy = df.strategy.to_string().split()
            strategy_affix = strategy[1]
            return affix, strategy_affix

    def segment(self, word):
        """Your own tokenization approach.

        Args:
            word (str): The word to segment.

        Returns:
            None | list[str]: Returns the list of morphemes or None, if your approach could not segment the word.
        """
        morphemes = []
        results, final_results = self.morphemes_finder(word)
        inflectional_affix = self.inflectional_finder(word)
        derivational_affix, derivational_strategy_affix = self.derivational_finder(word)
        print("inflectional and derivational: ", inflectional_affix, "|", derivational_affix)
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
                        print("inflectional selected form, meaning and strategy_affix: ", inf_selected_form,
                              inf_selected_meaning,
                              inf_selected_strategy_affix, inflectional)
                    if form == derivational_affix:
                        de_selected_meaning = meaning
                        de_selected_form = form
                        de_selected_strategy_affix = derivational_strategy_affix
                        derivational = True
                        print("derivational selected form, meaning and strategy_affix: ", de_selected_form,
                              de_selected_meaning,
                              de_selected_strategy_affix, derivational)
                    if form != inflectional_affix and form != derivational_affix:
                        not_selected_meaning = meaning
                        not_selected_form = form
                        not_selected_strategy_affix = strategy
                        Not_found = True
                        print("None selected form, meaning and strategy_affix: ", not_selected_form,
                              not_selected_meaning,
                              not_selected_strategy_affix, Not_found)
        if inflectional == True:
            selected_form = inf_selected_form
            selected_meaning = inf_selected_meaning
            selected_strategy_affix = inf_selected_strategy_affix
            print("1", selected_meaning)
        elif derivational == True:
            selected_form = de_selected_form
            selected_meaning = de_selected_meaning
            selected_strategy_affix = de_selected_strategy_affix
            print("2", selected_meaning)
        elif Not_found == True:
            selected_form = not_selected_form
            selected_meaning = not_selected_meaning
            selected_strategy_affix = not_selected_strategy_affix
            print("3", selected_meaning)
        if selected_form != None:
            rest_word = word.replace(selected_form, '')
            print(selected_form, rest_word)
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
        """Your own tokenization approach with the fallback to a greedy tokenizer.

        Args:
            word (str): The word to segment.

        Returns:
            list[str]: Returns the list of morphemes or sub-word units in case of fallback.
        """
        return self.word_tokenize(word)

    def helper(self, result):
        final = []
        for x in result:
            if isinstance(x, list):
                for i in range(len(x)):
                    temp = x[i]
                    final.append(temp)
            else:
                final.append(x)
        return final

    def tokenize(self):
        """Tokenization method for your own approach, including fallback greedy tokenization.

        Args:
            sentence (str): The sentence to tokenize.
            add_special_tokens (bool): If True, add special tokens (not finished, just to show the interface..).

        Returns:
            list[str]: A list of tokens.
        """
        ls_tokenized_word_index, ls_tokenized_word, ls_untokenized_word = self.pre_tokenize()
        ls_morphemes = []
        segmenter_output = []
        for untokenized_word in ls_untokenized_word:
            resegment = self.segment(untokenized_word)

            if resegment != [None]:
                ls_morphemes.append(resegment)
            else:
                segmenter_output.append(untokenized_word)

        with open('data.txt', 'w') as fp:
            for w in segmenter_output:
                fp.write("%s\n" % w)
        for untokenized_word in segmenter_output:
            resegment = self.segment_with_fallback(untokenized_word)
            print("attention", resegment)

        for i in range(len(ls_tokenized_word)):
            ls_morphemes.insert(ls_tokenized_word_index[i], ls_tokenized_word[i])
        ls_morphemes = self.helper(ls_morphemes)
        return ls_morphemes


if __name__ == '__main__':
    sentence = "greatful It is ozonising inconsistency xxxxxxxx wwwwwww xxxxxxxx wwwwwww xxxxxxxx wwwwwww xxxxxxxx wwwwwww xxxxxxxx wwwwwww xxxxxxxx wwwwwww"
    tokens = WenxuTokenizer(sentence)
    result = tokens.tokenize()
    print(result)
