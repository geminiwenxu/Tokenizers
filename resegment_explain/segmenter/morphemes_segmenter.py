import functools

import pandas as pd
from transformers import BertTokenizer

import resegment_explain.segmenter.morphemes_lib as morphemes


class MorphemesTokenizer():
    """Resegment those untokenized words by our segmenter"""

    def __init__(self, model_path, inflectional_path, derivational_path, resegment_only=True):
        self.model_path = model_path
        self.wp_tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.inflectional_df = pd.read_csv(inflectional_path, sep='\t')
        self.inflectional_df.columns = ['word', 'inflectional_word', 'pos', 'affix']
        self.derivational_df = pd.read_csv(derivational_path, sep='\t')
        self.derivational_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']
        self.resegment_only = resegment_only

    def check_word(self, word):
        """Tokenize sentence by the pre-trained tokenizer to find out the poorly tokenized words split by #.

            Args:
                word: str

            Returns:
                untokenized(original) word (str|None): word which is split by the pre-trained tokenizer
                                             None words is not split
        """

        if len(self.wp_tokenizer.tokenize(word)) > 2:
            return word

    def morphemes_finder(self, poor_word):
        """Searching for morphological segmentation and meaning.

        Args:
            word (str): a word

        Returns:
            result (dict): morphological result
        """
        results = morphemes.discover_segments(poor_word)
        return results

    def inflectional_finder(self, poor_word):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:
            affix (str): inflectional affix or None

        """
        answer_df = self.inflectional_df[self.inflectional_df.eq(poor_word).any(axis=1)]
        if answer_df.empty:
            return None
        else:
            df = self.inflectional_df.loc[self.inflectional_df['inflectional_word'] == poor_word]
            try:
                affix = df.affix.to_string().split()[1]
            except:
                pass
            try:
                affix = df.affix.to_string().split()[1].split('|')[1]
            except:
                pass
            return affix

    def derivational_finder(self, poor_word):
        """Searching for inflectional segmentation.

        Args:
            word (str): a word

        Returns:

            affix (str): derivational affix or None
            strategy_affix(str): "prefix" or "suffix"

        """

        answer_df = self.derivational_df[self.derivational_df.eq(poor_word).any(axis=1)]
        if answer_df.empty:
            return None, None
        else:
            df = self.derivational_df.loc[self.derivational_df['derivational_word'] == poor_word]
            affix = df.affix.to_string().split()[1]
            strategy = df.strategy.to_string().split()
            strategy_affix = strategy[1]
            return affix, strategy_affix

    @functools.lru_cache(maxsize=None)
    def test_segment(self, poor_word):
        inflectional_form = self.inflectional_finder(poor_word)
        derivational_form, derivational_strategy_affix = self.derivational_finder(poor_word)
        results = self.morphemes_finder(poor_word)
        for strategy in ['prefix', 'root', "suffix"]:
            if strategy in results:
                strategy_dict = results[strategy][0]['all_entries']
                for key, value in strategy_dict.items():
                    form = strategy_dict[key]["form"]
                    meaning = strategy_dict[key]["meaning"][0]
                    if derivational_form == form:
                        match derivational_strategy_affix:
                            case "prefix":
                                rest_word = poor_word[(len(derivational_form)):]
                                if self.resegment_only is True:
                                    first = derivational_form
                                    second = rest_word
                                else:
                                    first = meaning
                                    second = rest_word
                            case "root" | "suffix":
                                rest_word = poor_word[:-(len(derivational_form))]
                                if self.resegment_only is True:
                                    first = rest_word
                                    second = derivational_form
                                else:
                                    first = rest_word
                                    second = meaning
                        morphemes = [first, second]
                        print("1", morphemes)
                        return morphemes
                    elif inflectional_form == form:
                        rest_word = poor_word[:-(len(inflectional_form))]
                        if self.resegment_only is True:
                            first = rest_word
                            second = inflectional_form
                        else:
                            first = rest_word
                            second = meaning

                        morphemes = [first, second]
                        print("2", morphemes)
                        return morphemes
                    else:
                        match strategy:
                            case "prefix":
                                rest_word = poor_word[(len(form)):]
                                if self.resegment_only is True:
                                    first = form
                                    second = rest_word
                                else:
                                    first = rest_word
                                    second = meaning
                            case "root" | "suffix":
                                rest_word = poor_word[:-(len(form))]
                                if self.resegment_only is True:
                                    first = rest_word
                                    second = form
                                else:
                                    first = rest_word
                                    second = meaning
                        morphemes = [first, second]
                        print("3", morphemes)
            else:
                morphemes = [None]
                print("4", morphemes)
        print("5", morphemes)
        return morphemes

    @functools.lru_cache(maxsize=None)
    def segment(self, poor_word):
        """Morphological tokenization approach.

        Args:
            word (str): The word to segment.

        Returns:
            morphemes(list[str] | list[None]): Returns the list of morphemes or
                                            list[None]: when the approach can not segment the word.
        """
        morphemes = []
        results = self.morphemes_finder(poor_word)

        inflectional_form = self.inflectional_finder(poor_word)
        derivational_form, derivational_strategy_affix = self.derivational_finder(poor_word)
        inflectional = False
        derivational = False
        not_found = False
        selected_form = None
        for strategy in ['suffix', 'root', "prefix"]:
            if strategy in results:
                strategy_dict = results[strategy][0]['all_entries']
                for key, value in strategy_dict.items():  # the later key and value -> form and meaning will OVERWRITES!
                    # each key, value -> form and meaning TURNING ON different FLAG!
                    # by 1st key, value-> TFF
                    # by 2nd key, value-> FFF
                    # result: FFF -> F
                    # by 1st key, value-> FFF
                    # by 2nd key, value-> TFF
                    # result: TFF -> T
                    form = strategy_dict[key]["form"]
                    meaning = strategy_dict[key]["meaning"][0]  # extract the FIRST meaning!
                    # 0: keep mutiple if, as in each for, it should check every case
                    # 1: in each for prefer derivational over others
                    if form == derivational_form:
                        de_selected_meaning = meaning
                        de_selected_form = form
                        de_selected_strategy_affix = derivational_strategy_affix
                        derivational = True
                    elif form == inflectional_form:
                        inf_selected_meaning = meaning
                        inf_selected_form = form
                        inf_selected_strategy_affix = "suffix"
                        inflectional = True
                    elif form != inflectional_form and form != derivational_form:
                        not_selected_meaning = meaning
                        not_selected_form = form
                        not_selected_strategy_affix = strategy
                        not_found = True
        # 3: the flags could be multiple True, but derivational is also preferred
        if derivational is True:
            selected_form = de_selected_form
            selected_meaning = de_selected_meaning
            selected_strategy_affix = de_selected_strategy_affix
        elif inflectional is True:
            selected_form = inf_selected_form
            selected_meaning = inf_selected_meaning
            selected_strategy_affix = inf_selected_strategy_affix
        elif not_found is True:
            selected_form = not_selected_form
            selected_meaning = not_selected_meaning
            selected_strategy_affix = not_selected_strategy_affix
        if selected_form is not None:
            # 4: using "is"
            # 5: combine root and suffix
            # 6: using match case
            match selected_strategy_affix:
                case "prefix":
                    rest_word = poor_word[(len(selected_form)):]
                    if self.resegment_only is True:
                        morphemes.append(selected_form)
                        morphemes.append(rest_word)
                    else:
                        morphemes.append(selected_meaning)
                        morphemes.append(rest_word)
                case "root" | "suffix":
                    rest_word = poor_word[:-(len(selected_form))]
                    if self.resegment_only is True:
                        morphemes.append(rest_word)
                        morphemes.append(selected_form)
                    else:
                        morphemes.append(rest_word)
                        morphemes.append(selected_meaning)
        else:
            morphemes.append(None)
        return morphemes

    def tokenize(self, word):
        """Morphological tokenization method, including fallback greedy tokenization.

        Args:
            sentence (str): The sentence to tokenize.
            add_special_tokens (bool): If True, add special tokens (not finished, just to show the interface..).

        Returns:
            list[str]: A list of tokens.
        """
        poor_word = self.check_word(word)
        if poor_word is not None:
            resegment = self.test_segment(poor_word)
            print("attention", resegment)
            if resegment != [None]:
                if resegment == (self.wp_tokenizer.tokenize(resegment[0]) + self.wp_tokenizer.tokenize(resegment[1])):
                    retokenized_token = resegment
                    # retokenized_token = [resegment[0], "##" + resegment[1]]
                    print(self.wp_tokenizer.tokenize(word))
                    print(retokenized_token)
                    # """line 338-339: only for calculating the percentage"""
                    # with open('qnli_test.txt', 'a') as f:
                    #     print('compare', self.wp_tokenizer.tokenize(word), resegment, file=f)
                else:
                    retokenized_token = self.wp_tokenizer.tokenize(word)
            else:
                retokenized_token = self.wp_tokenizer.tokenize(word)
        else:
            retokenized_token = self.wp_tokenizer.tokenize(word)
        return retokenized_token


if __name__ == '__main__':
    pass
