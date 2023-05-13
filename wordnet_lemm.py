import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("eliminability"))
