class RejectionCriteria():
    def __init__(self, vocab):
        self.vocab = vocab


class Check():
    def __init__(self, vocab, prefixes=['anti'], suffixes=['ly'], if_seg=False):
        self.vocab = vocab
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.if_seg = if_seg

    def check_suffix(self):
        for i in self.suffixes:
            if self.vocab.endswith(i):
                suffix = i
                stem = self.vocab.removesuffix(suffix)
                result = [stem, suffix]
                self.if_seg = True
            else:
                suffix = None
                stem = self.vocab
                result = [stem, suffix]
                self.if_seg = False
        return result, self.if_seg

    def check_prefix(self):
        for j in self.prefixes:
            if self.vocab.startswith(j):
                prefix = j
                stem = self.vocab.removeprefix(prefix)
                result = [prefix, stem]
                self.if_seg = True
            else:
                prefix = None
                stem = self.vocab
                result = [prefix, stem]
                self.if_seg = False
        return result, self.if_seg

    def segment(self):
        result, self.if_seg = self.check_suffix()
        print("suffix: ", result[0], result[1], self.if_seg)
        if result[1] == None:
            result, self.if_seg = self.check_prefix()
            print("prefix: ", result[0], result[1], self.if_seg)
            if result[0] == None:
                print("No segmentation")
                result = [self.vocab]
                self.if_seg = False

        return result, self.if_seg


if __name__ == '__main__':
    test = Check('day')
    result, if_seg = test.segment()
    print("result: ", result, if_seg)
