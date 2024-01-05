import matplotlib.pyplot as plt
import numpy as np


def plot(log_history, actual_task, model):
    ls_train_loss = []
    ls_train_accuracy = []
    ls_train_pearson = []
    ls_train_spearmanr = []
    ls_train_matt = []
    ls_loss = []
    ls_eval_loss = []
    ls_eval_accuracy = []
    ls_eval_pearson = []
    ls_eval_spearmanr = []
    ls_eval_matt = []
    for e in range(0, 15, 3):
        train_loss = log_history[e]["train_loss"]
        ls_train_loss.append(train_loss)
        train_accuracy = log_history[e]["train_accuracy"]
        ls_train_accuracy.append(train_accuracy)

        # train_pearson = log_history[e]["train_pearson"]
        # train_spearmanr = log_history[e]["train_spearmanr"]
        # ls_train_pearson.append(train_pearson)
        # ls_train_spearmanr.append(train_spearmanr)

        """cola"""
        # train_matt = log_history[e]["train_matthews_correlation"]
        # ls_train_matt.append(train_matt)

        loss = log_history[e + 1]["loss"]
        ls_loss.append(loss)

        eval_loss = log_history[e + 2]["eval_loss"]
        ls_eval_loss.append(eval_loss)
        eval_accuracy = log_history[e + 2]["eval_accuracy"]
        ls_eval_accuracy.append(eval_accuracy)

        # eval_pearson = log_history[e + 2]["eval_pearson"]
        # eval_spearmanr = log_history[e + 2]["eval_spearmanr"]
        # ls_eval_pearson.append(eval_pearson)
        # ls_eval_spearmanr.append(eval_spearmanr)

        """cola"""
        # eval_matt = log_history[e + 2]["eval_matthews_correlation"]
        # ls_eval_matt.append(eval_matt)

    fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(10, 6))
    axs1.plot(np.arange(1, 6), np.array(ls_train_loss), "r", label='Training Loss')
    axs1.plot(np.arange(1, 6), np.array(ls_eval_loss), "b", label='Validation Loss')
    axs1.set_xlabel('Epochs')
    axs1.set_ylabel('Loss')
    axs1.legend()
    axs2.plot(np.arange(1, 6), np.array(ls_train_accuracy), "r", label='Training Accuracy')
    axs2.plot(np.arange(1, 6), np.array(ls_eval_accuracy), "b", label='Validation Accuracy')
    axs2.set_xlabel('Epochs')
    axs2.set_ylabel('Accuracy')
    axs2.legend()
    plt.tight_layout()
    plt.savefig(f'{model + actual_task}.png')


if __name__ == '__main__':
    log_history = [{'train_loss': 0.31559547781944275, 'train_accuracy': 0.8862776354589486, 'train_runtime': 4402.5305,
                    'train_samples_per_second': 89.199, 'train_steps_per_second': 1.394, 'epoch': 1.0, 'step': 6136},
                   {'loss': 0.5289, 'learning_rate': 3.0997533538961934e-05, 'epoch': 1.0, 'step': 6136},
                   {'eval_loss': 0.4237290024757385, 'eval_accuracy': 0.8335028478437754, 'eval_runtime': 110.3774,
                    'eval_samples_per_second': 89.076, 'eval_steps_per_second': 1.395, 'epoch': 1.0, 'step': 6136},
                   {'train_loss': 0.18579238653182983, 'train_accuracy': 0.9409908785796863, 'train_runtime': 4408.504,
                    'train_samples_per_second': 89.078, 'train_steps_per_second': 1.392, 'epoch': 2.0, 'step': 12272},
                   {'loss': 0.346, 'learning_rate': 2.324815015422145e-05, 'epoch': 2.0, 'step': 12272},
                   {'eval_loss': 0.4336816668510437, 'eval_accuracy': 0.8403173311635476, 'eval_runtime': 109.9674,
                    'eval_samples_per_second': 89.408, 'eval_steps_per_second': 1.4, 'epoch': 2.0, 'step': 12272},
                   {'train_loss': 0.09058939665555954, 'train_accuracy': 0.9742858452465228, 'train_runtime': 4398.4442,
                    'train_samples_per_second': 89.282, 'train_steps_per_second': 1.395, 'epoch': 3.0, 'step': 18408},
                   {'loss': 0.2194, 'learning_rate': 1.5498766769480967e-05, 'epoch': 3.0, 'step': 18408},
                   {'eval_loss': 0.4790451228618622, 'eval_accuracy': 0.8395036615134256, 'eval_runtime': 110.1574,
                    'eval_samples_per_second': 89.254, 'eval_steps_per_second': 1.398, 'epoch': 3.0, 'step': 18408},
                   {'train_loss': 0.044961072504520416, 'train_accuracy': 0.9881742389903795,
                    'train_runtime': 4394.2806, 'train_samples_per_second': 89.367, 'train_steps_per_second': 1.396,
                    'epoch': 4.0, 'step': 24544},
                   {'loss': 0.1334, 'learning_rate': 7.749383384740484e-06, 'epoch': 4.0, 'step': 24544},
                   {'eval_loss': 0.5847989320755005, 'eval_accuracy': 0.8395036615134256, 'eval_runtime': 110.0414,
                    'eval_samples_per_second': 89.348, 'eval_steps_per_second': 1.399, 'epoch': 4.0, 'step': 24544},
                   {'train_loss': 0.029908405616879463, 'train_accuracy': 0.9923224226003433,
                    'train_runtime': 4394.8308, 'train_samples_per_second': 89.355, 'train_steps_per_second': 1.396,
                    'epoch': 5.0, 'step': 30680}, {'loss': 0.0824, 'learning_rate': 0.0, 'epoch': 5.0, 'step': 30680},
                   {'eval_loss': 0.7235627174377441, 'eval_accuracy': 0.8379780309194467, 'eval_runtime': 110.0408,
                    'eval_samples_per_second': 89.349, 'eval_steps_per_second': 1.399, 'epoch': 5.0, 'step': 30680},
                   {'train_runtime': 84443.5398, 'train_samples_per_second': 23.252, 'train_steps_per_second': 0.363,
                    'total_flos': 5.166258268431053e+17, 'train_loss': 0.2620114034282182, 'epoch': 5.0, 'step': 30680},
                   {'eval_loss': 0.4336816668510437, 'eval_accuracy': 0.8403173311635476, 'eval_runtime': 109.322,
                    'eval_samples_per_second': 89.936, 'eval_steps_per_second': 1.409, 'epoch': 5.0, 'step': 30680}]
    plot(log_history, actual_task="mnli-mismatched", model="baseline")
