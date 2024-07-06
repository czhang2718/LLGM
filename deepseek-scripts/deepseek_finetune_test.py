import os
from datetime import datetime
from pathlib import Path
import secrets

from deepseek_download import get_classification_model, get_dataset

from modal_setup import (
    app,
    axolotl_image,
    HOURS,
    MINUTES,
    VOLUME_CONFIG,
)

GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:2")
if len(GPU_CONFIG.split(":")) <= 1:
    N_GPUS = int(os.environ.get("N_GPUS", 2))
    GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"
SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "a10g:1")

@app.function(
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
    _allow_background_volume_commits=True,
)
def train():
    from transformers import TrainingArguments, Trainer
    import torch
    import numpy as np
    import evaluate

    config, model_classify, tokenizer = get_classification_model.local()

    small_train_dataset, small_eval_dataset = get_dataset.local(tokenizer)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="/test_trainer", 
        eval_strategy="epoch",
        auto_find_batch_size=True
    )

    trainer = Trainer(
        model=model_classify,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
