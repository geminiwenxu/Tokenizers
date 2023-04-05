# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from load_data import load_data, dataset_to_text
from load_model import build_model
from prepare_dataset import prepare_dataset
from tokenizer import train_tokenizer


def main():
    data_train, data_test = load_data()
    dataset_to_text(data_train, "train.txt")
    dataset_to_text(data_test, "test.txt")

    tokenizer = train_tokenizer()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    model_path = "pretrained-bert"
    model = build_model()
    training_args = TrainingArguments(
        output_dir=model_path,  # output directory to where save model checkpoint
        evaluation_strategy="steps",  # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        num_train_epochs=10,  # number of training epochs, feel free to tweak
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
    main()
