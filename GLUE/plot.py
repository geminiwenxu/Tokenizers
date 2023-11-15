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
    log_history = [{'train_loss': 0.31705278158187866, 'train_accuracy': 0.8855926376743689, 'train_runtime': 4417.7522,
                    'train_samples_per_second': 88.892, 'train_steps_per_second': 1.389, 'epoch': 1.0, 'step': 6136},
                   {'loss': 0.5303, 'learning_rate': 3.0997533538961934e-05, 'epoch': 1.0, 'step': 6136},
                   {'eval_loss': 0.4378598928451538, 'eval_accuracy': 0.8266938359653592, 'eval_runtime': 111.5141,
                    'eval_samples_per_second': 88.016, 'eval_steps_per_second': 1.381, 'epoch': 1.0, 'step': 6136},
                   {'train_loss': 0.1887846291065216, 'train_accuracy': 0.9395521285860525, 'train_runtime': 4516.4912,
                    'train_samples_per_second': 86.948, 'train_steps_per_second': 1.359, 'epoch': 2.0, 'step': 12272},
                   {'loss': 0.3462, 'learning_rate': 2.324815015422145e-05, 'epoch': 2.0, 'step': 12272},
                   {'eval_loss': 0.4418899118900299, 'eval_accuracy': 0.8336220071319409, 'eval_runtime': 110.6874,
                    'eval_samples_per_second': 88.673, 'eval_steps_per_second': 1.391, 'epoch': 2.0, 'step': 12272},
                   {'train_loss': 0.09265873581171036, 'train_accuracy': 0.9747416616161874, 'train_runtime': 4471.191,
                    'train_samples_per_second': 87.829, 'train_steps_per_second': 1.372, 'epoch': 3.0, 'step': 18408},
                   {'loss': 0.2221, 'learning_rate': 1.5498766769480967e-05, 'epoch': 3.0, 'step': 18408},
                   {'eval_loss': 0.4812075197696686, 'eval_accuracy': 0.8370860927152318, 'eval_runtime': 113.348,
                    'eval_samples_per_second': 86.592, 'eval_steps_per_second': 1.359, 'epoch': 3.0, 'step': 18408},
                   {'train_loss': 0.04564801976084709, 'train_accuracy': 0.9879959867787789, 'train_runtime': 4484.6115,
                    'train_samples_per_second': 87.567, 'train_steps_per_second': 1.368, 'epoch': 4.0, 'step': 24544},
                   {'loss': 0.1337, 'learning_rate': 7.749383384740484e-06, 'epoch': 4.0, 'step': 24544},
                   {'eval_loss': 0.6015509366989136, 'eval_accuracy': 0.8385124808965868, 'eval_runtime': 110.8681,
                    'eval_samples_per_second': 88.529, 'eval_steps_per_second': 1.389, 'epoch': 4.0, 'step': 24544},
                   {'train_loss': 0.030065622180700302, 'train_accuracy': 0.9923631659629948,
                    'train_runtime': 4456.5595, 'train_samples_per_second': 88.118, 'train_steps_per_second': 1.377,
                    'epoch': 5.0, 'step': 30680}, {'loss': 0.0836, 'learning_rate': 0.0, 'epoch': 5.0, 'step': 30680},
                   {'eval_loss': 0.7392398118972778, 'eval_accuracy': 0.8365766683647479, 'eval_runtime': 110.6909,
                    'eval_samples_per_second': 88.67, 'eval_steps_per_second': 1.391, 'epoch': 5.0, 'step': 30680},
                   {'train_runtime': 85171.5929, 'train_samples_per_second': 23.054, 'train_steps_per_second': 0.36,
                    'total_flos': 5.166258268431053e+17, 'train_loss': 0.2631927370869663, 'epoch': 5.0, 'step': 30680},
                   {'eval_loss': 0.6015509366989136, 'eval_accuracy': 0.8385124808965868, 'eval_runtime': 110.2284,
                    'eval_samples_per_second': 89.042, 'eval_steps_per_second': 1.397, 'epoch': 5.0, 'step': 30680}]
    plot(log_history, actual_task="mnli_matched", model="modified")
