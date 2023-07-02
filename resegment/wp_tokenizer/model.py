from transformers import BertConfig, BertForMaskedLM


# initialize the model with the config
def build_model(vocab_size, max_length):
    model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)
    return model
