import torch
from tqdm.auto import tqdm
from transformers import AdamW

from bert_tokenizer import build_tokenizer
from build_model import build_model
from data_collection import collect_data
from prepare_data import Dataset, mask_encodings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    collect_data()
    tokenizer = build_tokenizer()
    encodings = mask_encodings(tokenizer)
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    model = build_model()

    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    epochs = 1
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
    model.save_pretrained('./liberto')


if __name__ == '__main__':
    main()
