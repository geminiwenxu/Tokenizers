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
    word = "cats"
    print("length of the input word: ", len(word))
    results, final_result = find(word)
    # pp.pprint(results)
    pp.pprint(final_result)
    strategies = ["prefix", "suffix", "root"]
    for strategy in strategies:
        if strategy in results:
            prefix_dict = results[strategy][0]['all_entries']
            for prefix, meaning in prefix_dict.items():
                print(f"this is information about {strategy}:")
                pp.pprint(prefix)
                pp.pprint(prefix_dict.get(prefix)["meaning"])
