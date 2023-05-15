from functools import lru_cache

import numpy as np
import pandas as pd
from transformers import BertTokenizer

import morphemes_lib as morphemes


class GreedyTokenizer:
    def tokenize(word: str) -> list[str]:
        """Any kind of greedy sub-word unit tokenizer (BPE, SentencePiece, WordPiece, etc.).
        However, it needs to have been trained on the OUTPUT of your segmenter!!

        Args:
            word (str): The word to tokenize.

        Returns:
            list[str]: Returns the list of tokens. Guaranteed to be a valid segmentation because of how the algorithm works, UNLESS there is an unknown character in the input word.
        """
        pass


# Note:
# You need to figure out, whether to implement your tokenizer using an interface from the huggingface libraries, or whether not to.
#
# In the first case, you will need to figure out how to work a new method into the tokenizers or transformers pipeline:
# - you could implement your segmenter in the prepare_for_tokenization method from transformers.PreTrainedTokenizer, that is called before the tokenization actually happens, or
# - you could implement it in the tokenizers library, which sound more complicated, but you do need to train a tokenizer in either case!
#
# In the second case, you may not be able to use pre-existing training scripts without changing the tokenization, which may be complicated, depending on how much abstraction is done (e.g. Trainer class from transformers does a lot of abstraction).


class WenxuTokenizer:
    """A mock-up of what your own tokenizer could look like.
    Note that there are obviously pieces missing, like a normalizer, etc."""

    def __init__(self, sentence):
        self.sentence = sentence.lower()

    def finder(self, word):
        results = morphemes.discover_segments(word)

        if morphemes.format_results(results, "") == word:
            final_results = morphemes.generate_final_results(results)

        else:
            final_results = morphemes.format_results(results, "+"), "failed"
        return results, final_results

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
        if len(ls) > 0:
            for i in ls_array:
                begin = i[0] - 1
                i = np.insert(i, 0, begin, axis=0)
                word = []
                for j in i:
                    word.append(tokens[j])
                untokenized_word = "".join(word).replace("#", "")
                print("the tokenized word: ", untokenized_word)
        return untokenized_word

    def inflectional_finder(self):
        inflectional_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.inflectional.v1.tsv',
                                      sep='\t')
        inflectional_df.columns = ['word', 'inflectional_word', 'pos', 'affix']
        answer_df = inflectional_df[inflectional_df.eq(self.untokenized_word).any(axis=1)]
        if answer_df.empty:
            return None
        else:
            df = inflectional_df.loc[inflectional_df['inflectional_word'] == self.untokenized_word]
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

    def derivational_finder(self):
        inflectional_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.derivational.v1.tsv',
                                      sep='\t')
        inflectional_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']

        answer_df = inflectional_df[inflectional_df.eq(self.untokenized_word).any(axis=1)]
        if answer_df.empty:
            return None, None
        else:
            df = inflectional_df.loc[inflectional_df['derivational_word'] == self.untokenized_word]
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
        tokenizer = self.load_tokenizer()
        results, final_results = find(untokenized_word)

        inflectional_affix = self.inflectional_finder(untokenized_word)
        derivational_affix, derivational_strategy_affix = self.derivational_finder(untokenized_word)
        inflectional = False
        derivational = False
        Not_found = False
        for strategy in ['prefix', 'root', "suffix"]:
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

        elif derivational == True:
            selected_form = de_selected_form
            selected_meaning = de_selected_meaning
            selected_strategy_affix = de_selected_strategy_affix

        elif Not_found == True:
            selected_form = not_selected_form
            selected_meaning = not_selected_meaning
            selected_strategy_affix = not_selected_strategy_affix

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

    # You could simply use a LRU cache for memoization,
    # to speed up your tokenizer if its slow (in exchange for greater memory consumption, of cause)
    @lru_cache(maxsize=1_000_000)
    def segment_with_fallback(word: str) -> None | list[str]:
        """Your own tokenization approach with the fallback to a greedy tokenizer.

        Args:
            word (str): The word to segment.

        Returns:
            list[str]: Returns the list of morphemes or sub-word units in case of fallback.
        """
        maybe_morphemes = self.segment(word)

        if maybe_morphemes is not None:
            return maybe_morphemes
        else:
            return self.fallback_tokenizer.tokenize(word)

    def tokenize(sentence: str, add_special_tokens=True) -> list[str]:
        """Tokenization method for your own approach, including fallback greedy tokenization.

        Args:
            sentence (str): The sentence to tokenize.
            add_special_tokens (bool): If True, add special tokens (not finished, just to show the interface..).

        Returns:
            list[str]: A list of tokens.
        """
        words = self.pre_tokenize(sentence)

        tokenized_sentence = ["CLS"] if add_special_tokens else []

        for word in words:
            tokens = self.segment_with_fallback(word)
            tokenized_sentence.expand(tokens)

        return tokenized_sentence
