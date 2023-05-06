# test_word.py

import pprint
from datetime import datetime

import morphemes_lib as morphemes


def find(word):
    # nltk.download('wordnet')
    startTime = datetime.now()

    pp = pprint.PrettyPrinter()

    results = morphemes.discover_segments(word)

    if morphemes.format_results(results, "") == word:
        final_results = morphemes.generate_final_results(results)

    else:
        final_results = morphemes.format_results(results, "+"), "failed"

    pp.pprint(final_results)
    timeElapsed = datetime.now() - startTime
    print('script: time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))

    return final_results


if __name__ == '__main__':
    find("deconstructed")
