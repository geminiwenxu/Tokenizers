def bpe(word):
    pass


class TokenizerOutput:
    def __init__(self, was_sgemented, tokens):
        self.was_sgemented = was_sgemented
        self.tokens = tokens


def own_approach(word: str) -> TokenizerOutput:
    pass


def tokenizer_pipeline(sentence: str):
    sentence: list(str) = sentence.split()

    tokenized_sentence = []
    for word in sentence:
        output = own_approach(word)

        if not output.was_sgemented:
            tokens = bpe(word)
        else:
            tokens = output.tokens

        tokenized_sentence.expand(tokens)
    pass
