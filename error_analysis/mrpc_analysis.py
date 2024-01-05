from resegment_explain.tokenization_bert_modified import ModifiedBertTokenizer

if __name__ == '__main__':
    model_checkpoint = "bert-base-cased"
    modified_tokenizer = ModifiedBertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    f = open(
        "/Users/geminiwenxu/PycharmProjects/Tokenizers/error_analysis/modified_correct_classification_of mrpc.txt",
        "r")
    lines = f.readlines()
    for line in lines:
        if line.startswith('Sentence 1'):
            sen_1 = line.replace("Sentence 1: ", "")
            modified_tokenizer(sen_1, return_tensors="pt")
        elif line.startswith('Sentence 2'):
            sen_2 = line.replace("Sentence 2: ", "")
            modified_tokenizer(sen_2, return_tensors="pt")
