from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification


def model():
    output_model_file = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/pytorch_model.bin"
    output_config_file = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/config.json"
    output_vocab_file = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/pytorch_model.bin"
    folder_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/pre-train_bert/liberto/"

    config = RobertaConfig.from_json_file(output_config_file)
    tokenizer = RobertaTokenizer.from_pretrained(folder_path)
    print(tokenizer.tokenize("hello, world!"))
    model = RobertaForSequenceClassification.from_pretrained(folder_path)
    print(model(**tokenizer("hello, world!", return_tensors="pt")))
    return model
