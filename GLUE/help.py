import random

import datasets
import numpy as np
import pandas as pd


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    return df


if __name__ == '__main__':
    predictions = np.matrix([[3, 2, 1], [5, 14, 1], [7, 8, 9]])
    print(predictions)
    result = np.argmax(predictions, axis=1)
    print(result)

    result = predictions[:, 0]
    print(result)
