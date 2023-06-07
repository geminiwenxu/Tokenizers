# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
import yaml
from pkg_resources import resource_filename
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from baseline_tokenizer.load_data import load_data, dataset_to_text
from baseline_tokenizer.model import build_model
from baseline_tokenizer.my_tokenizer import train_tokenizer
from baseline_tokenizer.prepare_dataset import prepare_dataset


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
file_path = resource_filename(__name__, config['train']['path'])
print(file_path)
vocab_size = config['vocab_size']
max_length = config['max_length']

data_train, data_test = load_data()
dataset_to_text(data_train, "train.txt")
dataset_to_text(data_test, "test.txt")


def training():
    tokenizer = train_tokenizer()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    model_path = "greedy_tokenizer"
    model = build_model(vocab_size, max_length)
    training_args = TrainingArguments(
        output_dir=model_path,  # output directory to where save model checkpoint
        evaluation_strategy="steps",  # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        num_train_epochs=1,  # number of training epochs, feel free to tweak
        per_device_train_batch_size=10,  # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=64,  # evaluation batch size
        logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    # initialize the trainer and pass everything to it
    train_dataset, test_dataset = prepare_dataset(data_train, data_test)
    print(train_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    # train the model
    trainer.train()


if __name__ == '__main__':
    training()
