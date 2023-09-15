import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def extract():
    config = get_config('/../config/config.yaml')
    cc_news_path = resource_filename(__name__, config['cc_news']['path'])
    f = open(cc_news_path, "r")
    out = open("../data/raw/extracted", "w")
    i = 0
    for x in f:
        print(x)
        out.write(x)
        i += 1
        if i == 1000:
            break


if __name__ == '__main__':
    extract()
