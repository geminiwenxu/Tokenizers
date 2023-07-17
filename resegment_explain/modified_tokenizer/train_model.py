# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
import yaml
from pkg_resources import resource_filename
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from resegment_explain.modified_tokenizer.load_data import load_data, dataset_to_text
from resegment_explain.modified_tokenizer.model import build_model
from resegment_explain.modified_tokenizer.prepare_dataset import prepare_dataset
from resegment_explain.tokenization_bert_modified import ModifiedBertTokenizer


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../../config/config.yaml')
file_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/data/raw/shuffled_ccnews_enwiki.txt"
vocab_size = 30_522
max_length = 512
epoch = 1
batch_size = 2
data_train, data_test = load_data(file_path)
dataset_to_text(data_train, "train.txt")
dataset_to_text(data_test, "test.txt")
model_path = "pretrained_tokenizer"


def training():
    modified_tokenizer = ModifiedBertTokenizer(
        vocab_file="/Users/geminiwenxu/PycharmProjects/Tokenizers/data/pretrained_tokenizer/vocab.txt")
    tokenizer = modified_tokenizer.from_pretrained(
        "/Users/geminiwenxu/PycharmProjects/Tokenizers/data/pretrained_tokenizer")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    model = build_model(vocab_size, max_length)
    # model = model.to("cuda:0")
    training_args = TrainingArguments(
        output_dir=model_path,  # output directory to where save model checkpoint
        evaluation_strategy="steps",  # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        num_train_epochs=epoch,  # number of training epochs, feel free to tweak
        per_device_train_batch_size=batch_size,  # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=64,  # evaluation batch size
        logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    # initialize the trainer and pass everything to it
    train_dataset, test_dataset = prepare_dataset(data_train, data_test)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    # train the model
    for i in tqdm(trainer.train()):
        print(i)


if __name__ == '__main__':
    # training()
    modified_tokenizer = ModifiedBertTokenizer(
        vocab_file="/Users/geminiwenxu/PycharmProjects/Tokenizers/data/pretrained_tokenizer/vocab.txt")
    print(modified_tokenizer)
    tokenizer = modified_tokenizer.from_pretrained(
        "/Users/geminiwenxu/PycharmProjects/Tokenizers/data/pretrained_tokenizer")