import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
file_path = resource_filename(__name__, config['train']['path'])


def remove():
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open("../yourfile.txt", "w") as f:
        for line in lines:
            if line.strip(
                    "\n") != "yalamberpaviskandharbalambahritihumatijitedastigalinjapushkasuyarmapapabunkaswanandasthunkojinghrinanelukathorthokovermagujapushkarkeshusujasansagunamkhimbupatukagasti":
                print(line)
                f.write(line)


if __name__ == '__main__':
    remove()
