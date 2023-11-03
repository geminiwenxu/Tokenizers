import matplotlib.pyplot as plt
import numpy as np


def plot(log_history, epoch, actual_task, model):
    ls_train_loss = []
    ls_train_accuracy = []
    ls_loss = []
    ls_eval_loss = []
    ls_eval_accuracy = []
    for e in range(0, 60, 3):
        train_loss = log_history[e]["train_loss"]
        train_accuracy = log_history[e]["train_accuracy"]
        loss = log_history[e + 1]["loss"]
        eval_loss = log_history[e + 2]["eval_loss"]
        eval_accuracy = log_history[e + 2]["eval_accuracy"]
        ls_train_loss.append(train_loss)
        ls_train_accuracy.append(train_accuracy)
        ls_loss.append(loss)
        ls_eval_loss.append(eval_loss)
        ls_eval_accuracy.append(eval_accuracy)
    fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(10, 6))
    axs1.plot(np.arange(1, 21), np.array(ls_train_loss), "r", label='Training Loss')
    axs1.plot(np.arange(1, 21), np.array(ls_eval_loss), "b", label='Validation Loss')
    axs1.set_xlabel('Epochs')
    axs1.set_ylabel('Loss')
    axs1.legend()
    axs2.plot(np.arange(1, 21), np.array(ls_train_accuracy), "r", label='Training Accuracy')
    axs2.plot(np.arange(1, 21), np.array(ls_eval_accuracy), "b", label='Validation Accuracy')
    axs2.set_xlabel('Epochs')
    axs2.set_ylabel('Accuracy')
    axs2.legend()
    plt.tight_layout()
    plt.savefig(f'{model + actual_task}.png')


if __name__ == '__main__':
    log_histroy = [{'train_loss': 0.6961346864700317, 'train_accuracy': 0.49133858267716535, 'train_runtime': 1.896,
                    'train_samples_per_second': 334.91, 'train_steps_per_second': 21.097, 'epoch': 1.0, 'step': 40},
                   {'loss': 0.725, 'learning_rate': 3.8040365472055704e-05, 'epoch': 1.0, 'step': 40},
                   {'eval_loss': 0.7055951356887817, 'eval_accuracy': 0.43661971830985913, 'eval_runtime': 0.2932,
                    'eval_samples_per_second': 242.118, 'eval_steps_per_second': 17.051, 'epoch': 1.0, 'step': 40},
                   {'train_loss': 0.6923832297325134, 'train_accuracy': 0.510236220472441, 'train_runtime': 1.8686,
                    'train_samples_per_second': 339.82, 'train_steps_per_second': 21.406, 'epoch': 2.0, 'step': 80},
                   {'loss': 0.7018, 'learning_rate': 3.6038240973526455e-05, 'epoch': 2.0, 'step': 80},
                   {'eval_loss': 0.6928262710571289, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.3,
                    'eval_samples_per_second': 236.656, 'eval_steps_per_second': 16.666, 'epoch': 2.0, 'step': 80},
                   {'train_loss': 0.6907603144645691, 'train_accuracy': 0.5118110236220472, 'train_runtime': 1.8828,
                    'train_samples_per_second': 337.257, 'train_steps_per_second': 21.245, 'epoch': 3.0, 'step': 120},
                   {'loss': 0.7063, 'learning_rate': 3.4036116474997206e-05, 'epoch': 3.0, 'step': 120},
                   {'eval_loss': 0.7012796998023987, 'eval_accuracy': 0.5352112676056338, 'eval_runtime': 0.3485,
                    'eval_samples_per_second': 203.755, 'eval_steps_per_second': 14.349, 'epoch': 3.0, 'step': 120},
                   {'train_loss': 0.689856767654419, 'train_accuracy': 0.5496062992125984, 'train_runtime': 1.8487,
                    'train_samples_per_second': 343.483, 'train_steps_per_second': 21.637, 'epoch': 4.0, 'step': 160},
                   {'loss': 0.702, 'learning_rate': 3.2033991976467964e-05, 'epoch': 4.0, 'step': 160},
                   {'eval_loss': 0.713426947593689, 'eval_accuracy': 0.2676056338028169, 'eval_runtime': 0.2958,
                    'eval_samples_per_second': 240.054, 'eval_steps_per_second': 16.905, 'epoch': 4.0, 'step': 160},
                   {'train_loss': 0.6881913542747498, 'train_accuracy': 0.5149606299212598, 'train_runtime': 1.8827,
                    'train_samples_per_second': 337.273, 'train_steps_per_second': 21.246, 'epoch': 5.0, 'step': 200},
                   {'loss': 0.6999, 'learning_rate': 3.0031867477938716e-05, 'epoch': 5.0, 'step': 200},
                   {'eval_loss': 0.7102630734443665, 'eval_accuracy': 0.5070422535211268, 'eval_runtime': 0.2862,
                    'eval_samples_per_second': 248.045, 'eval_steps_per_second': 17.468, 'epoch': 5.0, 'step': 200},
                   {'train_loss': 0.6950893402099609, 'train_accuracy': 0.5086614173228347, 'train_runtime': 1.9334,
                    'train_samples_per_second': 328.43, 'train_steps_per_second': 20.689, 'epoch': 6.0, 'step': 240},
                   {'loss': 0.7055, 'learning_rate': 2.8029742979409464e-05, 'epoch': 6.0, 'step': 240},
                   {'eval_loss': 0.6935649514198303, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.2525,
                    'eval_samples_per_second': 281.188, 'eval_steps_per_second': 19.802, 'epoch': 6.0, 'step': 240},
                   {'train_loss': 0.7709738612174988, 'train_accuracy': 0.5086614173228347, 'train_runtime': 1.8595,
                    'train_samples_per_second': 341.487, 'train_steps_per_second': 21.511, 'epoch': 7.0, 'step': 280},
                   {'loss': 0.6935, 'learning_rate': 2.602761848088022e-05, 'epoch': 7.0, 'step': 280},
                   {'eval_loss': 0.7281943559646606, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.2532,
                    'eval_samples_per_second': 280.426, 'eval_steps_per_second': 19.748, 'epoch': 7.0, 'step': 280},
                   {'train_loss': 0.6951360106468201, 'train_accuracy': 0.5086614173228347, 'train_runtime': 1.8521,
                    'train_samples_per_second': 342.854, 'train_steps_per_second': 21.597, 'epoch': 8.0, 'step': 320},
                   {'loss': 0.7158, 'learning_rate': 2.402549398235097e-05, 'epoch': 8.0, 'step': 320},
                   {'eval_loss': 0.6869811415672302, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.2569,
                    'eval_samples_per_second': 276.353, 'eval_steps_per_second': 19.461, 'epoch': 8.0, 'step': 320},
                   {'train_loss': 0.6888261437416077, 'train_accuracy': 0.5181102362204725, 'train_runtime': 1.9013,
                    'train_samples_per_second': 333.984, 'train_steps_per_second': 21.038, 'epoch': 9.0, 'step': 360},
                   {'loss': 0.7095, 'learning_rate': 2.2023369483821725e-05, 'epoch': 9.0, 'step': 360},
                   {'eval_loss': 0.7155113816261292, 'eval_accuracy': 0.36619718309859156, 'eval_runtime': 0.2598,
                    'eval_samples_per_second': 273.286, 'eval_steps_per_second': 19.245, 'epoch': 9.0, 'step': 360},
                   {'train_loss': 0.6985875368118286, 'train_accuracy': 0.5086614173228347, 'train_runtime': 1.882,
                    'train_samples_per_second': 337.409, 'train_steps_per_second': 21.254, 'epoch': 10.0, 'step': 400},
                   {'loss': 0.6998, 'learning_rate': 2.0021244985292476e-05, 'epoch': 10.0, 'step': 400},
                   {'eval_loss': 0.6917583346366882, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.3749,
                    'eval_samples_per_second': 189.4, 'eval_steps_per_second': 13.338, 'epoch': 10.0, 'step': 400},
                   {'train_loss': 0.689294159412384, 'train_accuracy': 0.5322834645669291, 'train_runtime': 1.8768,
                    'train_samples_per_second': 338.333, 'train_steps_per_second': 21.312, 'epoch': 11.0, 'step': 440},
                   {'loss': 0.7091, 'learning_rate': 1.8019120486763227e-05, 'epoch': 11.0, 'step': 440},
                   {'eval_loss': 0.7399232387542725, 'eval_accuracy': 0.29577464788732394, 'eval_runtime': 0.2597,
                    'eval_samples_per_second': 273.428, 'eval_steps_per_second': 19.255, 'epoch': 11.0, 'step': 440},
                   {'train_loss': 0.7077211737632751, 'train_accuracy': 0.5086614173228347, 'train_runtime': 1.9503,
                    'train_samples_per_second': 325.583, 'train_steps_per_second': 20.509, 'epoch': 12.0, 'step': 480},
                   {'loss': 0.7073, 'learning_rate': 1.6016995988233982e-05, 'epoch': 12.0, 'step': 480},
                   {'eval_loss': 0.7308531999588013, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.2883,
                    'eval_samples_per_second': 246.251, 'eval_steps_per_second': 17.342, 'epoch': 12.0, 'step': 480},
                   {'train_loss': 0.6841725707054138, 'train_accuracy': 0.5448818897637795, 'train_runtime': 1.9026,
                    'train_samples_per_second': 333.753, 'train_steps_per_second': 21.024, 'epoch': 13.0, 'step': 520},
                   {'loss': 0.6943, 'learning_rate': 1.4014871489704732e-05, 'epoch': 13.0, 'step': 520},
                   {'eval_loss': 0.7343922257423401, 'eval_accuracy': 0.19718309859154928, 'eval_runtime': 0.2857,
                    'eval_samples_per_second': 248.531, 'eval_steps_per_second': 17.502, 'epoch': 13.0, 'step': 520},
                   {'train_loss': 0.6854202747344971, 'train_accuracy': 0.5165354330708661, 'train_runtime': 1.8665,
                    'train_samples_per_second': 340.212, 'train_steps_per_second': 21.431, 'epoch': 14.0, 'step': 560},
                   {'loss': 0.6891, 'learning_rate': 1.2012746991175485e-05, 'epoch': 14.0, 'step': 560},
                   {'eval_loss': 0.7692744135856628, 'eval_accuracy': 0.49295774647887325, 'eval_runtime': 0.2899,
                    'eval_samples_per_second': 244.927, 'eval_steps_per_second': 17.248, 'epoch': 14.0, 'step': 560},
                   {'train_loss': 0.679984986782074, 'train_accuracy': 0.5401574803149606, 'train_runtime': 1.8655,
                    'train_samples_per_second': 340.389, 'train_steps_per_second': 21.442, 'epoch': 15.0, 'step': 600},
                   {'loss': 0.6887, 'learning_rate': 1.0010622492646238e-05, 'epoch': 15.0, 'step': 600},
                   {'eval_loss': 0.7799317240715027, 'eval_accuracy': 0.22535211267605634, 'eval_runtime': 0.2632,
                    'eval_samples_per_second': 269.764, 'eval_steps_per_second': 18.997, 'epoch': 15.0, 'step': 600},
                   {'train_loss': 0.6759399175643921, 'train_accuracy': 0.5590551181102362, 'train_runtime': 1.855,
                    'train_samples_per_second': 342.312, 'train_steps_per_second': 21.563, 'epoch': 16.0, 'step': 640},
                   {'loss': 0.6896, 'learning_rate': 8.008497994116991e-06, 'epoch': 16.0, 'step': 640},
                   {'eval_loss': 0.8080962896347046, 'eval_accuracy': 0.16901408450704225, 'eval_runtime': 0.2747,
                    'eval_samples_per_second': 258.505, 'eval_steps_per_second': 18.205, 'epoch': 16.0, 'step': 640},
                   {'train_loss': 0.6725960373878479, 'train_accuracy': 0.5590551181102362, 'train_runtime': 2.2598,
                    'train_samples_per_second': 280.999, 'train_steps_per_second': 17.701, 'epoch': 17.0, 'step': 680},
                   {'loss': 0.6803, 'learning_rate': 6.0063734955877425e-06, 'epoch': 17.0, 'step': 680},
                   {'eval_loss': 0.8530024290084839, 'eval_accuracy': 0.15492957746478872, 'eval_runtime': 0.2849,
                    'eval_samples_per_second': 249.245, 'eval_steps_per_second': 17.552, 'epoch': 17.0, 'step': 680},
                   {'train_loss': 0.670089066028595, 'train_accuracy': 0.5511811023622047, 'train_runtime': 1.9709,
                    'train_samples_per_second': 322.185, 'train_steps_per_second': 20.295, 'epoch': 18.0, 'step': 720},
                   {'loss': 0.6898, 'learning_rate': 4.0042489970584956e-06, 'epoch': 18.0, 'step': 720},
                   {'eval_loss': 0.8887796998023987, 'eval_accuracy': 0.18309859154929578, 'eval_runtime': 0.2589,
                    'eval_samples_per_second': 274.201, 'eval_steps_per_second': 19.31, 'epoch': 18.0, 'step': 720},
                   {'train_loss': 0.6675313711166382, 'train_accuracy': 0.5622047244094488, 'train_runtime': 1.8719,
                    'train_samples_per_second': 339.227, 'train_steps_per_second': 21.369, 'epoch': 19.0, 'step': 760},
                   {'loss': 0.681, 'learning_rate': 2.0021244985292478e-06, 'epoch': 19.0, 'step': 760},
                   {'eval_loss': 0.8977072238922119, 'eval_accuracy': 0.15492957746478872, 'eval_runtime': 0.2883,
                    'eval_samples_per_second': 246.283, 'eval_steps_per_second': 17.344, 'epoch': 19.0, 'step': 760},
                   {'train_loss': 0.666363000869751, 'train_accuracy': 0.5669291338582677, 'train_runtime': 1.9518,
                    'train_samples_per_second': 325.334, 'train_steps_per_second': 20.493, 'epoch': 20.0, 'step': 800},
                   {'loss': 0.6774, 'learning_rate': 0.0, 'epoch': 20.0, 'step': 800},
                   {'eval_loss': 0.9055594801902771, 'eval_accuracy': 0.16901408450704225, 'eval_runtime': 0.2825,
                    'eval_samples_per_second': 251.359, 'eval_steps_per_second': 17.701, 'epoch': 20.0, 'step': 800},
                   {'train_runtime': 543.7057, 'train_samples_per_second': 23.358, 'train_steps_per_second': 1.471,
                    'total_flos': 491947270760340.0, 'train_loss': 0.6982946062088012, 'epoch': 20.0, 'step': 800},
                   {'eval_loss': 0.6928262710571289, 'eval_accuracy': 0.5633802816901409, 'eval_runtime': 0.2755,
                    'eval_samples_per_second': 257.723, 'eval_steps_per_second': 18.15, 'epoch': 20.0, 'step': 800}]
    plot(log_histroy, epoch=20, actual_task="test", model="model")
