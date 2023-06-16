from approach_1.wp_tokenizer.load_data import load_data
from approach_1.wp_tokenizer.my_tokenizer import train_tokenizer


def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    tokenizer = train_tokenizer()
    max_length = 512
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                     max_length=max_length, return_special_tokens_mask=True)


def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    tokenizer = train_tokenizer()
    return tokenizer(examples["text"], return_special_tokens_mask=True)


def prepare_dataset(data_train, data_test):
    truncate_longer_samples = True
    # the encode function will depend on the truncate_longer_samples variable
    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
    # tokenizing the train dataset
    train_dataset = data_train.map(encode, batched=True)
    # tokenizing the testing dataset
    test_dataset = data_test.map(encode, batched=True)
    if truncate_longer_samples:
        # remove other columns and set input_ids and attention_mask as PyTorch tensors
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        # remove other columns, and remain them as Python lists
        test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
        train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    return train_dataset, test_dataset


# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= max_length:
#         total_length = (total_length // max_length) * max_length
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
#         for k, t in concatenated_examples.items()
#     }
#     return result
#
#
# if not truncate_longer_samples:
#     train_dataset = train_dataset.map(group_texts, batched=True,
#                                       desc=f"Grouping texts in chunks of {max_length}")
#     test_dataset = test_dataset.map(group_texts, batched=True,
#                                     desc=f"Grouping texts in chunks of {max_length}")
#     # convert them from lists to torch tensors
#     train_dataset.set_format("torch")
#     test_dataset.set_format("torch")
if __name__ == '__main__':
    data_train, data_test = load_data()
    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
    ]
    encode_with_truncation(data_train, special_tokens)
    prepare_dataset(data_train, data_test, special_tokens)
