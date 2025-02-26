#!/usr/bin/env python
import argparse
import csv
import logging
import sys
import math
import numpy as np
import torch

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

# -----------------------------------------------------------------------------
# Data loading and noise functions
# -----------------------------------------------------------------------------

def load_tsv_dataset(file_path):
    """
    Load a TSV file where each line contains two columns:
      - The first column is a label (e.g. "Sarcastic" or "Non-Sarcastic") which is ignored.
      - The second column is the Urdu text.
    Returns a Hugging Face Dataset with a single "text" field.
    """
    csv.field_size_limit(sys.maxsize)
    data = {"text": []}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue  # Skip lines without at least 2 columns
            # Only use the second column (index 1) for the text.
            data["text"].append(row[1])
    return Dataset.from_dict(data)

def apply_noise_to_ids(token_ids, drop_prob, pad_token_id, bos_token_id, eos_token_id):
    """
    Applies simple noise by randomly dropping tokens (except BOS, EOS, and PAD).
    This simulates a denoising objective.
    """
    if len(token_ids) <= 2:
        return token_ids

    noised = [token_ids[0]]  # Always keep BOS
    for token in token_ids[1:-1]:
        if token in {bos_token_id, eos_token_id, pad_token_id}:
            noised.append(token)
        else:
            if np.random.rand() > drop_prob:
                noised.append(token)
    noised.append(token_ids[-1])  # Always keep EOS

    # Ensure there's at least one non-special token
    if len(noised) < 3:
        noised.insert(1, token_ids[1])
    return noised

def tokenize_and_noise(examples, tokenizer, max_length, noise_drop_prob):
    """
    Tokenize the Urdu text and create a noisy version for input.
    The target remains the original (tokenized) sentence.
    """
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    original_ids = tokenized["input_ids"]

    noised_input_ids = []
    for ids in original_ids:
        noised = apply_noise_to_ids(
            ids,
            drop_prob=noise_drop_prob,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        noised_input_ids.append(noised)

    # Build an attention mask for the noised input (lengths may vary)
    noised_attention_mask = [[1] * len(seq) for seq in noised_input_ids]

    tokenized["input_ids"] = noised_input_ids
    tokenized["attention_mask"] = noised_attention_mask
    tokenized["labels"] = original_ids  # Target is the original text
    return tokenized

# -----------------------------------------------------------------------------
# Main training script (denoising for Urdu sarcasm augmentation)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Denoising fine-tuning for Urdu sarcasm augmentation"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to a TSV file containing label and Urdu text (labels in first column are ignored)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="Pretrained mBART model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--noise_drop_prob",
        type=float,
        default=0.3,
        help="Probability of dropping tokens to create noise (denoising)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Load the tokenizer and model, set up Urdu language codes
    # -----------------------------------------------------------------------------
    tokenizer = MBart50Tokenizer.from_pretrained(args.model_name)
    logger.info("Original language mapping: %s", tokenizer.lang_code_to_id)

    if "ur_PK" not in tokenizer.lang_code_to_id:
        tokenizer.lang_code_to_id["ur_PK"] = tokenizer.lang_code_to_id.get("hi_IN", 99)
    tokenizer.src_lang = "ur_PK"
    tokenizer.tgt_lang = "ur_PK"

    model = MBartForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id["ur_PK"]

    # -----------------------------------------------------------------------------
    # Load and prepare the dataset (apply tokenization and noise)
    # -----------------------------------------------------------------------------
    logger.info("Loading training data from %s", args.train_file)
    train_dataset = load_tsv_dataset(args.train_file)
    logger.info("Tokenizing data and applying noise for denoising training...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_and_noise(
            examples, tokenizer, args.max_length, args.noise_drop_prob
        ),
        batched=True,
    )

    # -----------------------------------------------------------------------------
    # Prepare the data collator and training arguments
    # -----------------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=3000,
        evaluation_strategy="no",
        predict_with_generate=True,
        fp16=True,
        report_to=["tensorboard"],
    )

    # -----------------------------------------------------------------------------
    # Initialize Trainer and start training
    # -----------------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
