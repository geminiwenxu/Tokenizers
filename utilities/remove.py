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
    with open("newfile.txt", "w") as f:
        for line in lines:
            tokens = line.split()
            # if len(tokens) != 1:
            #     f.write(line)
            # else:
            #     if len(tokens[0]) < 150:
            #         f.write(line)
            #     else:
            #         print(tokens)

            if line.strip(
                    "\n") != "appalarajupetaarikatotabusayavalasachandapuramchintalavalasaduppalapudienubaruvugollapetaitlamamidipallejannivalasakondakenguvakondapalavalasakotasirlamkottakkilollarapadumamidivalasamarrivalasamulachelagammutcherlavalasanaiduvalasanarasapurampataregapedachelagamramabhadrapuramravivalasarompallerompallivalasasamarthula":
                f.write(line)


if __name__ == '__main__':
    remove()
