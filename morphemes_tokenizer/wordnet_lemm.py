# nltk.download('wordnet')
import pandas as pd
from nltk.stem import WordNetLemmatizer


def inflectional_finder(word):
    inflectional_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.inflectional.v1.tsv',
                                  sep='\t')
    inflectional_df.columns = ['word', 'inflectional_word', 'pos', 'affix']
    answer_df = inflectional_df[inflectional_df.eq(word).any(axis=1)]
    if answer_df.empty:
        return None, None
    else:
        df = inflectional_df.loc[inflectional_df['inflectional_word'] == word]
        affix = df.affix.to_string().split()[1].split('|')[1]
        pos = df.pos.to_string().split()
        idx = int(pos[0])
        return idx, affix


def derivational_finder(word):
    inflectional_df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/Tokenizers/data/eng/eng.derivational.v1.tsv',
                                  sep='\t')
    inflectional_df.columns = ['word', 'derivational_word', 'pos_0', 'pos_1', 'affix', 'strategy']

    answer_df = inflectional_df[inflectional_df.eq(word).any(axis=1)]
    if answer_df.empty:
        return None, None, None
    else:
        df = inflectional_df.loc[inflectional_df['derivational_word'] == word]
        print(df)
        affix = df.affix.to_string().split()[1]
        strategy = df.strategy.to_string().split()
        idx = int(strategy[0])
        strategy_affix = strategy[1]
        return idx, affix, strategy_affix


if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    # print(lemmatizer.lemmatize("eliminability"))
    word = "ozonising"
    idx, affix = inflectional_finder(word)

    a, b, c = derivational_finder(word)
    print(a, b, c)
