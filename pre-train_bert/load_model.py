import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForTokenClassification


if __name__ == '__main__':
  output_model_file = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/pytorch_model.bin"
  output_config_file = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/config.json"
  output_vocab_file = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/pytorch_model.bin"
  config = RobertaConfig.from_json_file(output_config_file)
  model = RobertaForTokenClassification(config)
  state_dict = torch.load(output_model_file)
  model.state_dict()
  model.load_state_dict(state_dict)
  # tokenizer = RobertaTokenizer(output_vocab_file)