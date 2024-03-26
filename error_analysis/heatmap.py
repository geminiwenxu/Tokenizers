# df = pd.DataFrame([[1, 0.7917, 0.7015, 0.7955], [0, 0, 0.7917, 0.7955], [0, 0, 0, 0.7015]],
#                   columns=["baseline", "method 1", "method 2", "method 3"])
# print(df)
# corr = np.corrcoef(df)
# print(corr)
# # corr = [[1, 0.7917, 0.7015, 0.7955], [0, 0, 0.7917, 0.7955], [0, 0, 0, 0.7015]]
# mask = np.triu(np.zeros_like(df))
# print("----------")
# print(mask)
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
#     ax = sns.heatmap(df, mask=mask, vmax=.3, square=True, cmap="YlGnBu", annot=True)
#     plt.show()
dict = {'und-': {'loc': 'prefix', 'root': {
    'forms': [{'root': 'und-', 'form': 'und', 'loc': 'prefix', 'attach_to': [], 'category': ''},
              {'loc': 'suffix', 'root': '-und', 'form': 'und'}, {'loc': 'embedded', 'root': '-und-', 'form': 'und'}],
    'meaning': ['wave'], 'origin': 'Latin', 'etymology': 'unda',
    'examples': ['abound', 'abundance', 'abundant', 'inundant', 'inundate', 'inundation', 'redound', 'redundancy',
                 'redundant', 'superabound', 'superabundance', 'superabundant', 'surround', 'undine', 'undulant',
                 'undulate', 'undulation', 'undulatory', 'undulatus', 'undulose']}, 'form': 'und', 'len': 3,
                          'meaning': ['wave'], 'category': '', 'priority': 'highest'}, 'un-neg': {'loc': 'prefix',
                                                                                                  'root': {'forms': [
                                                                                                      {'root': 'un-',
                                                                                                       'form': 'un',
                                                                                                       'loc': 'prefix',
                                                                                                       'attach_to': [
                                                                                                           'adjective',
                                                                                                           'participle'],
                                                                                                       'category': 'preposition-like'}],
                                                                                                           'meaning': [
                                                                                                               'not',
                                                                                                               'negate',
                                                                                                               'negation',
                                                                                                               'not',
                                                                                                               'opposite'],
                                                                                                           'theme': 'negation',
                                                                                                           'origin': 'Latin',
                                                                                                           'etymology': 'ūnus, unius',
                                                                                                           'examples': [
                                                                                                               'undo',
                                                                                                               'unequal',
                                                                                                               'unfashionable',
                                                                                                               'unhappy',
                                                                                                               'unusual',
                                                                                                               'unfinished',
                                                                                                               'unfriendly',
                                                                                                               'undone',
                                                                                                               'unknown',
                                                                                                               'unsuitable']},
                                                                                                  'form': 'un',
                                                                                                  'len': 2,
                                                                                                  'meaning': ['not',
                                                                                                              'negate',
                                                                                                              'negation',
                                                                                                              'not',
                                                                                                              'opposite'],
                                                                                                  'category': 'preposition-like'},
                 'un-reverse': {'loc': 'prefix', 'root': {'forms': [
                     {'root': 'un-', 'form': 'un', 'loc': 'prefix', 'attach_to': ['verb', 'noun'],
                      'category': 'preposition-like'}], 'meaning': ['reverse', 'deprive', 'release'], 'origin': '',
                                                          'etymology': '',
                                                          'examples': ['undo', 'unlock', 'unstable', 'untie',
                                                                       'unwilling']}, 'form': 'un', 'len': 2,
                                'meaning': ['reverse', 'deprive', 'release'], 'category': 'preposition-like'}}
strategy_dict = {'-able-able': {'loc': 'suffix', 'root': {
    'forms': [{'root': '-able', 'form': 'able', 'pos': 'adjective', 'type': 'derivational', 'loc': 'suffix'}],
    'meaning': ['able to', 'fit to', 'having the quality of', 'capable of being', 'worthy'], 'origin': '',
    'etymology': '', 'examples': ['capable', 'agreeable', 'edible', 'visible', 'comfortable', 'portable']},
                                'form': 'able', 'len': 4,
                                'meaning': ['able to', 'fit to', 'having the quality of', 'capable of being', 'worthy'],
                                'priority': 'highest'}, '-ble-ble': {'loc': 'suffix', 'root': {
    'forms': [{'root': '-ble', 'form': 'ble', 'loc': 'suffix'}],
    'meaning': ['having the quality of', 'capable of being'], 'origin': '', 'etymology': '',
    'examples': ['humble', 'tumble']}, 'form': 'ble', 'len': 3,
                                                                     'meaning': ['having the quality of',
                                                                                 'capable of being']},
                 '-e-e': {'loc': 'suffix',
                          'root': {'forms': [{'root': '-e', 'form': 'e', 'loc': 'suffix'}], 'meaning': ['form noun'],
                                   'origin': '',
                                   'etymology': '', 'examples': ['chorale', 'finale', 'locale', 'rationale']},
                          'form': 'e',
                          'len': 1, 'meaning': ['form noun']}, '-le-le': {'loc': 'suffix', 'root': {
        'forms': [{'loc': 'suffix', 'root': '-le', 'form': 'le'}], 'meaning': ['form noun', 'form verb'], 'origin': '',
        'examples': ['handle', 'saddle', 'shuttle', 'sickle', 'thimble', 'whistle', 'babble', 'crackle', 'hobble',
                     'mingle', 'paddle', 'prattle', 'sparkle', 'tangle', 'tinkle', 'wriggle'], 'theme': 'grammatical'},
                                                                          'form': 'le', 'len': 2,
                                                                          'meaning': ['form noun', 'form verb']}}
if __name__ == '__main__':
    strategy_dict = {'ed-, es-': {'loc': 'embedded', 'root': {
        'forms': [{'root': 'ed-', 'form': 'ed', 'loc': 'prefix', 'attach_to': [], 'category': ''},
                  {'root': 'es-', 'form': 'es', 'loc': 'prefix', 'attach_to': [], 'category': ''},
                  {'loc': 'embedded', 'root': '-ed-', 'form': 'ed'}, {'loc': 'embedded', 'root': '-es-', 'form': 'es'}],
        'meaning': ['eat'], 'origin': 'Latin', 'etymology': 'edere, esus',
        'examples': ['comedo', 'comestible', 'edacity', 'edibility', 'edible', 'escarole', 'esculent', 'esurience',
                     'esurient', 'inedia', 'inedible', 'inescate', 'inescation', 'obese', 'obesity']}, 'form': 'es',
                         'len': 2, 'meaning': ['eat'], 'priority': 'highest'}}
    for key, value in strategy_dict.items():
        form = strategy_dict[key]["form"]
        meaning = strategy_dict[key]["meaning"][0]
        print(form,"|", meaning)
