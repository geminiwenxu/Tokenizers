import torch
from transformers import RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling


def build_model(vocab_size, max_len, hidden_size, hidden_layer, attention_heads, typo_size):
    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        hidden_size=hidden_size,
        num_hidden_layers=hidden_layer,
        num_attention_heads=attention_heads,
        typo_vocab_size=typo_size
    )

    model = RobertaForMaskedLM(config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model
