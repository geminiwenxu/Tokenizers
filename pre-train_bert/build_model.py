import torch
from transformers import RobertaConfig, RobertaForMaskedLM


def build_model():
    config = RobertaConfig(
        vocab_size=30_522,
        max_position_embeddings=514,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        typo_vocab_size=1
    )

    model = RobertaForMaskedLM(config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model
