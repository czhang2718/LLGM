import modal

from deepseek_params import model_name

from modal_setup import (
    app,
    axolotl_image,
    HOURS,
    MINUTES,
    VOLUME_CONFIG,
)

@app.function(image=axolotl_image, timeout=30 * MINUTES, volumes=VOLUME_CONFIG)
def launch():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} ...")
        snapshot_download(model_name)

        print("Committing /pretrained directory (no progress bar) ...")
        VOLUME_CONFIG["/pretrained"].commit()

@app.function(image=axolotl_image, timeout=30 * MINUTES, volumes=VOLUME_CONFIG)
def get_classification_model():
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    import torch

    import sys
    sys.path.append("/pretrained/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/")
    from e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11 import modeling_deepseek

    model_base = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )
    config.hidden_size = 100

    model_classify = modeling_deepseek.DeepseekV2ForSequenceClassification(config)
    model_classify.model = model_base

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    return config, model_classify, tokenizer

@app.function(image=axolotl_image, timeout=30 * MINUTES, volumes=VOLUME_CONFIG)
def get_dataset(tokenizer):
    from datasets import load_dataset

    dataset_name = "AnkitAI/CategorizedTextReviews"

    def tokenize_function(examples):
        return tokenizer(examples["review"], padding="max_length", truncation=True)

    dataset = load_dataset(dataset_name)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

    return small_train_dataset, small_eval_dataset
