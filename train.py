#!/usr/bin/env python
import argparse
import csv
import logging
import os

from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import csv

def load_tsv_dataset(file_path):
    """
    Load a TSV file with two columns: input_text and target_text.
    Returns a Hugging Face Dataset.
    """
    # Increase the CSV field size limit to handle very large fields.
    csv.field_size_limit(sys.maxsize)
    
    data = {"input_text": [], "target_text": []}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue  # Skip lines that don't have exactly 2 columns
            data["input_text"].append(row[0])
            data["target_text"].append(row[1])
    return Dataset.from_dict(data)

def tokenize_function(examples, tokenizer, max_source_length=128, max_target_length=128):
    """
    Tokenize the inputs and targets.
    """
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
    # Use the tokenizer as target tokenizer for the labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser(description="Fine-tune mBART for text generation/augmentation on Urdu")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training TSV file")
    parser.add_argument("--model_name", type=str, default="facebook/mbart-large-50-many-to-many-mmt", help="Pretrained mBART model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_source_length", type=int, default=128, help="Max length for input sequences")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max length for target sequences")
    args = parser.parse_args()

    # Load mBART tokenizer and model
    # After loading the tokenizer
    tokenizer = MBart50Tokenizer.from_pretrained(args.model_name)
    print("Original language mapping:", tokenizer.lang_code_to_id)

    # If Urdu is not present, add it manually by assigning it the same id as Hindi (hi_IN)
    if "ur_PK" not in tokenizer.lang_code_to_id:
        tokenizer.lang_code_to_id["ur_PK"] = tokenizer.lang_code_to_id["hi_IN"]

    # Now, explicitly set both source and target languages to our Urdu code
    tokenizer.src_lang = "ur_PK"
    tokenizer.tgt_lang = "ur_PK"

    # And update the model's forced beginning-of-sentence token
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id["ur_PK"]

    print("Updated language mapping:", tokenizer.lang_code_to_id)

    # Load the dataset from the TSV file
    logger.info("Loading training data from %s", args.train_file)
    train_dataset = load_tsv_dataset(args.train_file)

    # Tokenize the dataset
    logger.info("Tokenizing the dataset...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_source_length, args.max_target_length),
        batched=True,
    )
    
    print(tokenized_train)
    

    # Prepare a data collator that dynamically pads the inputs and labels to the maximum length in the batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        predict_with_generate=True,
        fp16=True,  # Enable mixed precision if supported
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model and tokenizer
    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
