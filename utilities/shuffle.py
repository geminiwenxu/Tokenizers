from random import sample

import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def shuffle():
    config = get_config('/../config/config.yaml')
    ccnews_enwiki_path = resource_filename(__name__, config['train']['path'])

    file = open(ccnews_enwiki_path)
    text_list = file.readlines()

    shuffled = sample(text_list, k=len(text_list))
    with open('shuffled_ccnews_enwiki.txt', 'w') as fp:
        for line in shuffled:
            fp.write(line)


if __name__ == '__main__':
    shuffle()
