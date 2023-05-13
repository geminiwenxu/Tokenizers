# test_word.py

import pprint
from datetime import datetime

import morphemes_lib as morphemes

pp = pprint.PrettyPrinter()


def find(word):
    # nltk.download('wordnet')
    startTime = datetime.now()

    results = morphemes.discover_segments(word)

    if morphemes.format_results(results, "") == word:
        final_results = morphemes.generate_final_results(results)

    else:
        final_results = morphemes.format_results(results, "+"), "failed"

    # pp.pprint(final_results)
    # timeElapsed = datetime.now() - startTime
    # print('script: time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))

    return results, final_results


if __name__ == '__main__':
    word = "Luxembourger"
    print("length of the input word: ", len(word))
    results, final_result = find(word)
    pp.pprint(results)
    print("----------------------------")
    pp.pprint(final_result)
    strategies = ["prefix", "suffix", "root"]
    for strategy in strategies:
        if strategy in results:
            strategy_dict = results[strategy][0]['all_entries']
            print(strategy_dict)
            for affix, meaning in strategy_dict.items():
                print(f"this is information about {strategy}:")
                pp.pprint(affix)
                pp.pprint(strategy_dict.get(affix)["meaning"])
