import math
import pandas as pd

path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/GLUE/method_3 of mnli-mm.tsv"


class Convert:
    def __init__(self, path):
        self.df = pd.read_csv(path, sep='\t')
        self.df = self.df.drop(columns=["index"])

    def mnli(self):
        for i in self.df['prediction']:
            if i == 0:
                self.df['prediction'] = self.df['prediction'].replace([i], 'entailment')
            elif i == 1:
                self.df['prediction'] = self.df['prediction'].replace([i], 'neutral')
            else:
                self.df['prediction'] = self.df['prediction'].replace([i], 'contradiction')
        self.df.index.name = 'index'
        self.df.to_csv("method_3 of " + "mnli-mm_converted" + ".tsv", sep="\t")

    def qnli_rte(self):
        for i in self.df['prediction']:
            if i == 0:
                self.df['prediction'] = self.df['prediction'].replace([i], 'entailment')
            else:
                self.df['prediction'] = self.df['prediction'].replace([i], 'not_entailment')
        self.df.index.name = 'index'
        self.df.to_csv("method_3 of " + "qnli_converted" + ".tsv", sep="\t")

    def stsb(self):
        for i in self.df['prediction']:
            sig = (1 / (1 + math.exp(-i))) * 5
            self.df['prediction'] = self.df['prediction'].replace([i], sig)

        self.df.index.name = 'index'
        self.df.to_csv("method_3 of " + "stsb_converted" + ".tsv", sep="\t")


if __name__ == '__main__':
    obj = Convert(path)
    obj.mnli()
