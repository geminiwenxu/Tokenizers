import os

import torch
import yaml
from pkg_resources import resource_filename
from tqdm.auto import tqdm
from transformers import AdamW

from bert_tokenizer import build_tokenizer
from build_model import build_model
from prepare_data import Dataset, mask_encodings

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
data_path = resource_filename(__name__, config['train']['path'])
batch_size = config['batch_size']
epochs = config['epoch']
lr = config['learning_rate']
max_len = config['max_len']
vocab_size = config['vocab_size']
hidden_size = config['hidden_size']
hidden_layer = config['hidden_layer']
attention_heads = config['attention_heads']
typo_size = config['typo_size']


def main():
    tokenizer = build_tokenizer(max_len)
    encodings = mask_encodings(data_path, tokenizer, max_len)
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = build_model(vocab_size, max_len, hidden_size, hidden_layer, attention_heads, typo_size)

    model.train()
    optim = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
    model.save_pretrained('./pretrained_tokenizer')


if __name__ == '__main__':
    main()
