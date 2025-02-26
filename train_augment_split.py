#!/usr/bin/env python
import argparse
import csv
import logging
import os
import sys
import math
import numpy as np
import torch

import evaluate
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data loading and noise functions
# -----------------------------------------------------------------------------

def load_tsv_dataset(file_path):
    """
    Load a TSV file where each line contains two columns:
      - First column: label (e.g. "Sarcastic" or "Non-Sarcastic") [ignored]
      - Second column: Urdu text.
    Returns a Hugging Face Dataset with a single "text" field.
    """
    csv.field_size_limit(sys.maxsize)
    data = {"text": []}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue  # Skip lines without at least two columns.
            # Use only the second column.
            data["text"].append(row[1])
    return Dataset.from_dict(data)

def apply_noise_to_ids(token_ids, drop_prob, pad_token_id, bos_token_id, eos_token_id):
    """
    Applies simple noise by randomly dropping tokens (except BOS, EOS, and PAD).
    This simulates a denoising objective.
    """
    if len(token_ids) <= 2:
        return token_ids

    noised = [token_ids[0]]  # Always keep BOS.
    for token in token_ids[1:-1]:
        if token in {bos_token_id, eos_token_id, pad_token_id}:
            noised.append(token)
        else:
            if np.random.rand() > drop_prob:
                noised.append(token)
    noised.append(token_ids[-1])  # Always keep EOS.

    # Ensure there's at least one non-special token.
    if len(noised) < 3:
        noised.insert(1, token_ids[1])
    return noised

def tokenize_and_noise(examples, tokenizer, max_length, noise_drop_prob, apply_noise=True):
    """
    Tokenizes the Urdu text.
    If apply_noise is True, creates a noised version for input while keeping
    the original tokenized sentence as the target.
    """
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    original_ids = tokenized["input_ids"]

    if apply_noise:
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
        # Build attention masks for the noised inputs.
        noised_attention_mask = [[1] * len(seq) for seq in noised_input_ids]
        tokenized["input_ids"] = noised_input_ids
        tokenized["attention_mask"] = noised_attention_mask

    # Set the target as the original tokenized sequence.
    tokenized["labels"] = original_ids
    return tokenized

# -----------------------------------------------------------------------------
# Main training script (denoising for Urdu sarcasm augmentation)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Denoising fine-tuning for Urdu sarcasm augmentation with train/val/test splits and checkpoint resume"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to a TSV file containing label and Urdu text (first column label is ignored)",
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
        help="Directory to save the fine-tuned model (and checkpoints)",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenization")
    parser.add_argument("--noise_drop_prob", type=float, default=0.3, help="Token drop probability for noise")
    # Optional split sizes (as ratios) for train, val, test. Default: 80/10/10.
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Load the tokenizer and model; set up Urdu language codes
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
    # Load the dataset and split into train/val/test
    # -----------------------------------------------------------------------------
    logger.info("Loading dataset from %s", args.train_file)
    full_dataset = load_tsv_dataset(args.train_file)
    total = len(full_dataset)
    test_val_size = args.val_ratio + args.test_ratio

    # First split off test+val from train.
    split_ds = full_dataset.train_test_split(test_size=test_val_size, seed=42)
    train_dataset = split_ds["train"]
    # Then split the combined test+val into validation and test.
    val_test_ds = split_ds["test"].train_test_split(
        test_size=args.test_ratio / test_val_size, seed=42
    )
    val_dataset = val_test_ds["train"]
    test_dataset = val_test_ds["test"]

    logger.info("Dataset sizes: Train=%d, Val=%d, Test=%d", len(train_dataset), len(val_dataset), len(test_dataset))

    # -----------------------------------------------------------------------------
    # Tokenize datasets.
    # For training, we apply noise; for validation/test, we do not.
    # -----------------------------------------------------------------------------
    logger.info("Tokenizing training data with noise...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_and_noise(examples, tokenizer, args.max_length, args.noise_drop_prob, apply_noise=True),
        batched=True,
    )
    logger.info("Tokenizing validation data without noise...")
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_and_noise(examples, tokenizer, args.max_length, args.noise_drop_prob, apply_noise=False),
        batched=True,
    )
    logger.info("Tokenizing test data without noise...")
    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_noise(examples, tokenizer, args.max_length, args.noise_drop_prob, apply_noise=False),
        batched=True,
    )

    # -----------------------------------------------------------------------------
    # Define evaluation metric (using sacreBLEU as a proxy for reconstruction quality)
    # -----------------------------------------------------------------------------
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
        return {"bleu": result["score"]}

    # -----------------------------------------------------------------------------
    # Set up training arguments: evaluate and save every epoch.
    # Only keep the latest checkpoint by setting save_total_limit=1.
    # -----------------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        fp16=True,
        logging_steps=100,
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # -----------------------------------------------------------------------------
    # Check for existing checkpoint and resume if found
    # -----------------------------------------------------------------------------
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            logger.info("Resuming training from checkpoint: %s", last_checkpoint)
        else:
            logger.info("No checkpoint found in %s. Training from scratch.", args.output_dir)

    # -----------------------------------------------------------------------------
    # Training loop with per-epoch evaluation
    # -----------------------------------------------------------------------------
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # After training, evaluate on the validation set.
    logger.info("Final evaluation on validation set:")
    metrics = trainer.evaluate(eval_dataset=tokenized_val)
    logger.info("Validation metrics: %s", metrics)

if __name__ == "__main__":
    main()
