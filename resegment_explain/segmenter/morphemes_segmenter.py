import functools

import pandas as pd
from transformers import BertTokenizer

import resegment_explain.segmenter.morphemes_lib as morphemes


class PreTokenizer:
    """Tokenizing a word by the pre-trained tokenizer, return the original word if poorly tokenized"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.wp_tokenizer = BertTokenizer.from_pretrained(self.model_path)

    def check_word(self, word):
        """Tokenize sentence by the pre-trained tokenizer to find out the poorly tokenized words split by #.

            Args:
                word: str

            Returns:
                untokenized(original) word (str|None): word which is split by the pre-trained tokenizer
                                             None words is not split
        """
        if len(self.wp_tokenizer.tokenize(word)) > 1:
            return word


class MorphemesTokenizer(PreTokenizer):
    """Resegment those untokenized words by our segmenter"""

    def __init__(self, model_path, inflectional_path, derivational_path, resegment_only=True):
        PreTokenizer.__init__(self, model_path)
        self.inflectional_df = pd.read_csv(inflectional_path, sep='\t')
        self.inflectional_df.columns = ['word', 'inflectional_word', 'pos', 'affix']
        self.derivational_df = pd.read_csv(derivational_path, sep='\t')
        self.derivational_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']
        self.resegment_only = resegment_only

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

        inflectional_affix = self.inflectional_finder(poor_word)
        derivational_affix, derivational_strategy_affix = self.derivational_finder(poor_word)
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
                    # 0: change multiple ifs to elif, so in each for hit one and break
                    # 1: in each for prefer derivational over others
                    if form == derivational_affix:
                        de_selected_meaning = meaning
                        de_selected_form = form
                        de_selected_strategy_affix = derivational_strategy_affix
                        derivational = True
                    elif form == inflectional_affix:
                        inf_selected_meaning = meaning
                        inf_selected_form = form
                        inf_selected_strategy_affix = "suffix"
                        inflectional = True
                    elif form != inflectional_affix and form != derivational_affix:
                        not_selected_meaning = meaning
                        not_selected_form = form
                        not_selected_strategy_affix = strategy
                        Not_found = True
        # 3: the flags could be multiple True, but derivational is also preferred
        if derivational is True:
            selected_form = de_selected_form
            selected_meaning = de_selected_meaning
            selected_strategy_affix = de_selected_strategy_affix
        elif inflectional is True:
            selected_form = inf_selected_form
            selected_meaning = inf_selected_meaning
            selected_strategy_affix = inf_selected_strategy_affix
        elif Not_found is True:
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
            resegment = self.segment(poor_word)
            if resegment != [None]:
                if resegment == self.wp_tokenizer.tokenize(resegment[0]) + self.wp_tokenizer.tokenize(resegment[1]):
                    retokenized_token = resegment
                    # if len(resegment[1]) < 3:
                    #     retokenized_token = [resegment[0], "##" + resegment[1]]
                    # else:
                    #     retokenized_token = resegment
                else:
                    retokenized_token = self.wp_tokenizer.tokenize(word)
            else:
                retokenized_token = self.wp_tokenizer.tokenize(word)
        else:
            retokenized_token = self.wp_tokenizer.tokenize(word)
        return retokenized_token


if __name__ == '__main__':
    pass
