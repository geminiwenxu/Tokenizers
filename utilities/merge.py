import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
cc_news_path = resource_filename(__name__, config['cc_news']['path'])
enwiki_path = resource_filename(__name__, config['enwiki']['path'])


def merge_files():
    # Python program to
    # demonstrate merging
    # of two files

    data = data2 = ""

    # Reading data from file1
    with open(cc_news_path) as fp:
        data = fp.read()

    # Reading data from file2
    with open(enwiki_path) as fp:
        data2 = fp.read()

    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2

    with open('ccnews_enwiki.txt', 'w') as fp:
        fp.write(data)


if __name__ == '__main__':
    merge_files()
