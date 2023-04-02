import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        return {key: tensor[i] for key, tensor in self.encodings.items()}


def mask_encodings(data_path, tokenizer, max_len):
    with open(data_path, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')

    batch = tokenizer(lines, max_length=max_len, padding='max_length', truncation=True)
    labels = torch.tensor([x for x in batch.input_ids])
    mask = torch.tensor([x for x in batch['attention_mask']])

    input_ids = labels.detach().clone()  # torch.Size([6000, 512])

    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < 0.15) * (input_ids != 1) * (input_ids != 2)  # torch.Size([6000, 512]) boolean arr
    for i in range(input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        input_ids[i, selection] = 3

    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings
