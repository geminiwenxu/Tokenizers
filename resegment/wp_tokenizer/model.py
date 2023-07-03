from transformers import BertConfig, BertForMaskedLM


# initialize the model with the config
def build_model(vocab_size, max_length):
    model_config = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                              vocab_size=vocab_size,
                              max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)
    return model