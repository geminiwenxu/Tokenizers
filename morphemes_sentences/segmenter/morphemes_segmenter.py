import numpy as np
import pandas as pd
from transformers import BertTokenizerFast

import morphemes_sentences.segmenter.morphemes_lib as morphemes


class PreTokenizer:
    """Tokenizing a word by the pre-trained tokenizer, return the original word if poorly tokenized"""

    def __init__(self, model_path, word):
        self.model_path = model_path
        self.word = word

    def consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def wp_tokenizer(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        return tokenizer

    def check_word(self):
        """Tokenize sentence by the pre-trained tokenizer to find out the poorly tokenized words split by #.

            Args:
                word: str

            Returns:
                untokenized_word (str|None): word which is poorly tokenized by the pre-trained tokenizer
                                             None words is split
        """
        tokenizer = self.wp_tokenizer()
        tokens = tokenizer.tokenize(self.word)
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
                poor_tokenized_word = "".join(word).replace("#", "")
                # print("the poorly tokenized word: ", poor_tokenized_word)
                return poor_tokenized_word


class MorphemesTokenizer(PreTokenizer):
    """Resegment those untokenized words by our segmenter"""

    def __init__(self, model_path, word, inflectional_path, derivational_path, resegment_only=True):
        PreTokenizer.__init__(self, model_path, word)
        self.inflectional_path = inflectional_path
        self.derivational_path = derivational_path
        self.resegment_only = resegment_only

    def morphemes_finder(self, poor_word):
        """Searching for morphological segmentation and meaning.

        Args:
            word (str): a word

        Returns:
            result (dict): morphological result
            final_result (dict): morphological result
        """
        results = morphemes.discover_segments(poor_word)

        if morphemes.format_results(results, "") == poor_word:
            final_results = morphemes.generate_final_results(results)

        else:
            final_results = morphemes.format_results(results, "+"), "failed"
        return results, final_results

    def inflectional_finder(self, poor_word):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:
            affix (str): inflectional affix or None

        """
        inflectional_df = pd.read_csv(self.inflectional_path, sep='\t')
        inflectional_df.columns = ['word', 'inflectional_word', 'pos', 'affix']
        answer_df = inflectional_df[inflectional_df.eq(poor_word).any(axis=1)]
        if answer_df.empty:
            return None
        else:
            df = inflectional_df.loc[inflectional_df['inflectional_word'] == poor_word]
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

    def derivational_finder(self, poor_word):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:

            affix (str): derivational affix or None
            strategy_affix(str): "prefix" or "suffix"

        """
        derivational_df = pd.read_csv(self.derivational_path, sep='\t')
        derivational_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']

        answer_df = derivational_df[derivational_df.eq(poor_word).any(axis=1)]
        if answer_df.empty:
            return None, None
        else:
            df = derivational_df.loc[derivational_df['derivational_word'] == poor_word]
            affix = df.affix.to_string().split()[1]
            strategy = df.strategy.to_string().split()
            strategy_affix = strategy[1]
            return affix, strategy_affix

    def segment(self, poor_word):
        """Morphological tokenization approach.

        Args:
            word (str): The word to segment.

        Returns:
            morphemes(list[str] | list[None]): Returns the list of morphemes or
                                            list[None]: when the approach can not segment the word.
        """
        morphemes = []
        results, final_results = self.morphemes_finder(poor_word)
        inflectional_affix = self.inflectional_finder(poor_word)
        derivational_affix, derivational_strategy_affix = self.derivational_finder(poor_word)
        # print(results)
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
            if selected_strategy_affix == "prefix":
                rest_word = poor_word[(len(selected_form)):]
                # print("prefix", selected_form, rest_word, selected_strategy_affix)
                if self.resegment_only is True:
                    morphemes.append(selected_form)
                    morphemes.append(rest_word)  # add #
                else:
                    morphemes.append(selected_meaning)
                    morphemes.append(rest_word)
            elif selected_strategy_affix == "root":
                rest_word = poor_word[:-(len(selected_form))]
                # print("root", selected_form, rest_word, selected_strategy_affix)
                if self.resegment_only is True:
                    morphemes.append(rest_word)
                    morphemes.append(selected_form)  # add #
                else:
                    morphemes.append(selected_meaning)
                    morphemes.append(rest_word)
            elif selected_strategy_affix == "suffix":
                rest_word = poor_word[:-(len(selected_form))]
                # print("suffix", selected_form, rest_word, selected_strategy_affix)
                if self.resegment_only is True:
                    morphemes.append(rest_word)
                    morphemes.append(selected_form)  # add #
                else:
                    morphemes.append(selected_meaning)
                    morphemes.append(rest_word)
        else:
            morphemes.append(None)
        return morphemes

    def tokenize(self, add_special_tokens=True):
        """Morphological tokenization method, including fallback greedy tokenization.

        Args:
            sentence (str): The sentence to tokenize.
            add_special_tokens (bool): If True, add special tokens (not finished, just to show the interface..).

        Returns:
            list[str]: A list of tokens.
        """

        poor_word = self.check_word()
        # print("poorly tokenized_word: ", poor_word)
        if poor_word != None:
            resegment = self.segment(poor_word)
            print("resegment result: ", resegment)
            if resegment != [None]:
                # print("attention 1", self.wp_tokenizer().tokenize(resegment))
                if resegment == self.wp_tokenizer().tokenize(resegment):
                    # for resegmented_token in resegment:
                    #     if len(resegmented_token) >= 5:
                    #         print(resegmented_token)
                    #         resegment = self.segment(resegmented_token)
                    #         print(resegment)
                    retokenized_token = resegment
                else:
                    # print("works")
                    retokenized_token = self.wp_tokenizer().tokenize(self.word)
            else:
                # print("Our approach can not resegment this word")
                # print("attention 2", self.wp_tokenizer().tokenize(poor_word))
                retokenized_token = self.wp_tokenizer().tokenize(self.word)
        else:
            # print("No poorly tokenized word!")
            retokenized_token = self.wp_tokenizer().tokenize(self.word)
        return retokenized_token


if __name__ == '__main__':
    pass
