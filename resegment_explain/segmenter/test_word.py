# test_word.py

import json
import sys
import pprint
import morphemes_lib as morphemes
from datetime import datetime

startTime = datetime.now()

pp = pprint.PrettyPrinter()

word = "apple"

results = morphemes.discover_segments(word)
# pp.pprint(results)
if __name__ == '__main__':
	if morphemes.format_results(results, "") == word:
		final_results = morphemes.generate_final_results(results)
		# print(final_results)
		pp.pprint(final_results)
	else:
		print(morphemes.format_results(results, "+"), "failed")

	timeElapsed=datetime.now()-startTime
	print('script: time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
