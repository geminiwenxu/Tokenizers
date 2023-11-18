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
    for e in range(0, 9, 3):
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
    axs1.plot(np.arange(1, 4), np.array(ls_train_loss), "r", label='Training Loss')
    axs1.plot(np.arange(1, 4), np.array(ls_eval_loss), "b", label='Validation Loss')
    axs1.set_xlabel('Epochs')
    axs1.set_ylabel('Loss')
    axs1.legend()
    axs2.plot(np.arange(1, 4), np.array(ls_train_accuracy), "r", label='Training Accuracy')
    axs2.plot(np.arange(1, 4), np.array(ls_eval_accuracy), "b", label='Validation Accuracy')
    axs2.set_xlabel('Epochs')
    axs2.set_ylabel('Accuracy')
    axs2.legend()
    plt.tight_layout()
    plt.savefig(f'{model + actual_task}.png')


if __name__ == '__main__':
    log_history = [
        {'train_loss': 0.25399497151374817, 'train_accuracy': 0.9083969465648855, 'train_f1': 0.9342980054751663,
         'train_runtime': 44.6864, 'train_samples_per_second': 82.083, 'train_steps_per_second': 5.147, 'epoch': 1.0,
         'step': 230}, {'loss': 0.5228, 'learning_rate': 2.66949933137233e-05, 'epoch': 1.0, 'step': 230},
        {'eval_loss': 0.36715832352638245, 'eval_accuracy': 0.8455882352941176, 'eval_f1': 0.893760539629005,
         'eval_runtime': 5.0021, 'eval_samples_per_second': 81.566, 'eval_steps_per_second': 5.198, 'epoch': 1.0,
         'step': 230},
        {'train_loss': 0.06595389544963837, 'train_accuracy': 0.9839149400218102, 'train_f1': 0.9880880274581062,
         'train_runtime': 44.7489, 'train_samples_per_second': 81.969, 'train_steps_per_second': 5.14, 'epoch': 2.0,
         'step': 460}, {'loss': 0.2457, 'learning_rate': 1.334749665686165e-05, 'epoch': 2.0, 'step': 460},
        {'eval_loss': 0.3579009175300598, 'eval_accuracy': 0.8676470588235294, 'eval_f1': 0.9042553191489361,
         'eval_runtime': 5.0024, 'eval_samples_per_second': 81.56, 'eval_steps_per_second': 5.197, 'epoch': 2.0,
         'step': 460},
        {'train_loss': 0.03270208090543747, 'train_accuracy': 0.9920937840785169, 'train_f1': 0.9941567600241789,
         'train_runtime': 44.7383, 'train_samples_per_second': 81.988, 'train_steps_per_second': 5.141, 'epoch': 3.0,
         'step': 690}, {'loss': 0.0761, 'learning_rate': 0.0, 'epoch': 3.0, 'step': 690},
        {'eval_loss': 0.5425792932510376, 'eval_accuracy': 0.8676470588235294, 'eval_f1': 0.9065743944636677,
         'eval_runtime': 5.0066, 'eval_samples_per_second': 81.493, 'eval_steps_per_second': 5.193, 'epoch': 3.0,
         'step': 690}, {'train_runtime': 600.5511, 'train_samples_per_second': 18.323, 'train_steps_per_second': 1.149,
                        'total_flos': 2895274053181440.0, 'train_loss': 0.2815390766530797, 'epoch': 3.0, 'step': 690},
        {'eval_loss': 0.3579009175300598, 'eval_accuracy': 0.8676470588235294, 'eval_f1': 0.9042553191489361,
         'eval_runtime': 4.9394, 'eval_samples_per_second': 82.601, 'eval_steps_per_second': 5.264, 'epoch': 3.0,
         'step': 690}]
    plot(log_history, actual_task="mrpc", model="method_0")
