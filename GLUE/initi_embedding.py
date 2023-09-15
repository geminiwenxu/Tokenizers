import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

if __name__ == '__main__':
    # print(model)
    # These are the input token embeddings
    print(model.get_input_embeddings())

    # Initialize new embeddings with your vocabulary size
    model.embeddings.word_embeddings = torch.nn.Embedding(123, 768, padding_idx=0)

    # Disable gradients on all model parameters
    for p in model.parameters(): p.requires_grad = False

    # Enable gradients only for new word embeddings
    model.embeddings.word_embeddings.requires_grad_()

    # Then train your model, some epochs should be enough
    model.train()  # TODO

    # Optionally: fine-tune your model after adjusting the embeddings
    # Then re-enable the gradients on the original model
    for p in model.parameters(): p.requires_grad = True
    # Train, maybe for one epoch or so
    model.train()  # TODO
