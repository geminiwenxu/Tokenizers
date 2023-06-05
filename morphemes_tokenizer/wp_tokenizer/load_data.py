from datasets import *


def load_data(file_path):
    dataset = load_dataset("text", data_files=file_path, split="train")
    d = dataset.train_test_split(test_size=0.1)
    return d["train"], d["test"]


def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)


if __name__ == '__main__':
    pass
