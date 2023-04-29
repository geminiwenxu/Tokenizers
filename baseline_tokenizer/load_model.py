import os

from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification

if __name__ == '__main__':
    model_path = "pretrained-bert"
    # load the model checkpoint
    model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "checkpoint-35000"), use_auth_token=True)
    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print(tokenizer.tokenize("rumination, unilateral, undesirable, antisocial"))
    print(model(**tokenizer("hello, world", return_tensors="pt")))
    input = tokenizer("hello, world", return_tensors="pt")
    print(input)
    print(os.path.join(model_path, "checkpoint-35000"))
    # output = model(**tokenizer("hello, world", return_tensors="pt"))
    # print(output['logits'].shape)
    # print(output)
    # logits(prediction_scores of language modeling head, scores for each vocabulary token before softmax
    # ) shape: batch_size*sequence_length*config.vocab_size = [1,5, 30522]
