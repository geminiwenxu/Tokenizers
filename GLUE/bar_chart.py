import matplotlib.pyplot as plt
import numpy as np

modified_f1 = [0.8892575694732477, 0.8841088674275681, 0.8781549173194081, 0.8860125260960334]
baseline_f1 = [0.872549019607843, 0.8504504504504504, 0.8756302521008402, 0.881656804733728]
modified_acc = [0.8452173913043478, 0.8469565217391304, 0.8376811594202899, 0.8417391304347827]
baseline_acc = [0.8191304347826087, 0.807536231884058, 0.8284057971014492, 0.8376811594202899]
if __name__ == '__main__':
    langs = [1, 42, 1234, 1337]

    species = ("1", "42", "1234", "1337")
    f1 = {
        'method_1_f1': (0.8892575694732477, 0.8841088674275681, 0.8781549173194081, 0.8860125260960334),
        'method_0_f1': (0.872549019607843, 0.8504504504504504, 0.8756302521008402, 0.881656804733728)
    }
    acc = {
        'method_1_acc': (0.8452173913043478, 0.8469565217391304, 0.8376811594202899, 0.8417391304347827),
        'method_0_acc': (0.8191304347826087, 0.807536231884058, 0.8284057971014492, 0.8376811594202899)
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(12, 6))

    for attribute, measurement in acc.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Random seed')
    ax.set_title('Accuracy')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    plt.savefig('acc')
