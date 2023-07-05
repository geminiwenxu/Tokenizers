# with open('data.txt', 'w') as fp:
        #     for w in tokenized_sentence:
        #         print(w)  # TODO: can not write all w into data.txt file?
        #         fp.write("%s\n" % w)
        # ls_tokenized_word_index, ls_tokenized_word, ls_untokenized_word_index, ls_untokenized_word, ls_untokenized_tokens = self.pre_tokenize()
        # print(ls_tokenized_word_index, ls_tokenized_word, ls_untokenized_word_index, ls_untokenized_word,
        #       ls_untokenized_tokens)
        # ls_morphemes = []
        # ls_morphemes_index = []
        # segmenter_output = []
        # segmenter_output_index = []
        # ls_retokenized_word = []
        # for idx, untokenized_word in enumerate(ls_untokenized_word):
        #     resegment_explain = self.segment(untokenized_word)
        #     if resegment_explain != [None]:
        #         print("resegment_explain done", ls_untokenized_word_index[idx], untokenized_word)
        #         ls_morphemes.append(resegment_explain)
        #     else:
        #         segmenter_output.append(untokenized_word)
        #         print("resegment_explain NOT done", ls_untokenized_word_index[idx], untokenized_word)
        # with open('data.txt', 'w') as fp:
        #     for w in segmenter_output:
        #         fp.write("%s\n" % w)
        # for i in segmenter_output:

        #     resegment_explain = self.segment_with_fallback(i)
        #     ls_retokenized_word.append(resegment_explain)
        # for i in range(len(ls_tokenized_word)):
        #     ls_morphemes.insert(ls_tokenized_word_index[i], ls_tokenized_word[i])
        # ls_morphemes = self.helper(ls_morphemes)


class PreTokenizer:
    def __init__(self, sentence):
        self.sentence = sentence.lower()

    def consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def pre_tokenize(self):
        """Pre-tokenizer that tries to uncover words in the input sentence.

        Args:
            sentence (str): The sentence to pre-tokenize.

        Returns:

            list[str]: A list of words.
        """

        tokenizer = self.pretrained_tokenizer()
        tokens = tokenizer.tokenize(self.sentence)
        print("tokens by pre-trained tokenizer: ", tokens)
        ls = []  # store the index number of tokens starts with #
        for idx, token in enumerate(tokens):
            if token.startswith("#"):
                ls.append(idx)
        ls_array = self.consecutive(np.array(ls), stepsize=1)
        ls_untokenized_word_index = []
        ls_tokenized_word_index = [i for i in range(len(tokens))]
        ls_untokenized_word = []
        if len(ls) > 0:
            for i in ls_array:
                # ls_untokenized_word_index.append(i[0] - 1)
                # for q in list(i):
                #     ls_untokenized_word_index.append(q.item())
                ls_tokenized_word_index.remove(i[0] - 1)
                [ls_tokenized_word_index.remove(x) for x in i]
                begin = i[0] - 1
                i = np.insert(i, 0, begin, axis=0)
                ls_untokenized_word_index.append(list(i))
                word = []
                for j in i:
                    word.append(tokens[j])
                untokenized_word = "".join(word).replace("#", "")
                print("the badly tokenized word: ", untokenized_word)
                ls_untokenized_word.append(untokenized_word)
        ls_tokenized_word = self.sentence.split()
        for i in ls_untokenized_word:
            ls_tokenized_word.remove(i)

        for p in ls_tokenized_word:
            tokens.remove(p)
        ls_untokenized_tokens = tokens
        return ls_tokenized_word_index, ls_tokenized_word, ls_untokenized_word_index, ls_untokenized_word, ls_untokenized_tokens


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