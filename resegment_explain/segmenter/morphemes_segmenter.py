import numpy as np
import pandas as pd
from transformers import BertTokenizer

import resegment_explain.segmenter.morphemes_lib as morphemes


class PreTokenizer:
    """Tokenizing a word by the pre-trained tokenizer, return the original word if poorly tokenized"""

    def consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def load_tokenizer(self, model_path):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer

    def check_word(self, word, model_path):
        """Tokenize sentence by the pre-trained tokenizer.

            Args:
                word: str

            Returns:
                untokenized_word (str|None): word which is poorly tokenized by the pre-trained tokenizer
        """
        tokenizer = self.load_tokenizer(model_path)
        tokens = tokenizer.tokenize(word)
        # print("tokens by pre-trained tokenizer: ", tokens)
        ls = []  # store the index number of tokens starts with #
        for idx, token in enumerate(tokens):
            if token.startswith("#"):
                ls.append(idx)
        ls_array = self.consecutive(np.array(ls))
        if len(ls) > 0:
            for i in ls_array:
                begin = i[0] - 1
                i = np.insert(i, 0, begin, axis=0)
                word = []
                for j in i:
                    word.append(tokens[j])
                untokenized_word = "".join(word).replace("#", "")
                # print("the tokenized word: ", untokenized_word)
                return untokenized_word


class MorphemesTokenizer(PreTokenizer):
    """Resegment those untokenized words by our segmenter"""

    def __init__(self, sentence):
        self.sentence = sentence

    def morphemes_finder(self, word):
        """Searching for morphological segmentation and meaning.

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

    def inflectional_finder(self, word, inflectional_path):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:
            affix (str): inflectional affix or None

        """
        inflectional_df = pd.read_csv(inflectional_path, sep='\t')
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

    def derivational_finder(self, word, derivational_path):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:

            affix (str): derivational affix or None
            strategy_affix(str): "prefix" or "suffix"

        """
        derivational_df = pd.read_csv(derivational_path, sep='\t')
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

    def segment(self, word, inflectional_path, derivational_path, resegment_only=True):
        """Morphological tokenization approach.

        Args:
            word (str): The word to segment.

        Returns:
            morphemes(list[None] | list[str]: Returns the list of morphemes or None(if your approach could not segment the word).
        """
        morphemes = []
        results, final_results = self.morphemes_finder(word)
        inflectional_affix = self.inflectional_finder(word, inflectional_path)
        derivational_affix, derivational_strategy_affix = self.derivational_finder(word, derivational_path)
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
                if resegment_only is True:
                    morphemes.append(selected_form)
                    morphemes.append(rest_word)
                else:
                    morphemes.append(selected_meaning)
                    morphemes.append(rest_word)

            elif selected_strategy_affix == "suffix":
                if resegment_only is True:
                    morphemes.append(rest_word)
                    morphemes.append(selected_form)
                else:
                    morphemes.append(selected_meaning)
                    morphemes.append(rest_word)
        else:
            morphemes.append(None)
        return morphemes

    def tokenize(self, model_path, inflectional_path, derivational_path, resegment_only, add_special_tokens=False):
        """Morphological tokenization method, including fallback greedy tokenization.

        Args:
            sentence (str): The sentence to tokenize.
            add_special_tokens (bool): If True, add special tokens (not finished, just to show the interface..).

        Returns:
            list[str]: A list of tokens.
        """
        tokenized_sentence = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"] if add_special_tokens else []

        for word in self.sentence.split():
            maybe_word = self.check_word(word, model_path)
            if maybe_word != None:
                resegment = self.segment(word, inflectional_path, derivational_path, resegment_only)
                tokenized_sentence.extend(resegment)
            else:
                tokenized_sentence.extend(list(word.split(" ")))
        return tokenized_sentence


if __name__ == '__main__':
    model_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/data/pretrained-bert"
    inflectional_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.inflectional.v1.tsv"
    derivational_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.derivational.v1.tsv"
    sentences = ["xxxxxxxxxx"]
    # sentences = ["hello"]
    for sen in sentences:
        tokens = MorphemesTokenizer(sen)
        result = tokens.tokenize(model_path, inflectional_path, derivational_path)
        print(result)
