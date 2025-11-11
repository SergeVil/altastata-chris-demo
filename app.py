#!/usr/bin/env python

"""AltaStata ChRIS demo plugin - Simple PyTorch training example."""

import importlib
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms as transforms
from chris_plugin import chris_plugin
from torch.utils.data import DataLoader, SubsetRandomSampler

from altastata.altastata_functions import AltaStataFunctions
from altastata.altastata_pytorch_dataset import (
    AltaStataPyTorchDataset,
    register_altastata_functions_for_pytorch,
)

from model_trainer import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY,
    ModelTrainer,
    set_random_seed,
    split_indices,
    write_summary,
)


# Configuration
FILE_PATTERN = "*.png"
CONFIG_MODULE = "altastata_config"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Command line arguments
parser = ArgumentParser(description="AltaStata PyTorch Training Demo")
parser.add_argument("--data-root", required=True, help="AltaStata path containing training images")
parser.add_argument("--model-output", help="AltaStata path to save trained model")
parser.add_argument("--summary-filename", default="training_summary.json", help="Training summary filename")


@chris_plugin(
    parser=parser,
    title="AltaStata PyTorch Training Demo",
    category="Learning",
    min_memory_limit="4Gi",
    min_cpu_limit="2000m",
    min_gpu_limit=0,
)
def main(options, inputdir: Path, outputdir: Path):
    """Main training pipeline - loads data from AltaStata, trains model, saves results."""
    print("\n" + "=" * 40)
    print("🚀 AltaStata ChRIS Demo - PyTorch Trainer")
    print("=" * 40 + "\n")

    # 1. Setup: Set random seed for reproducibility
    set_random_seed()

    # 2. Load credentials from config file
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(inputdir))

    config = importlib.import_module(CONFIG_MODULE)
    user_properties = config.user_properties
    private_key = config.private_key
    password = os.getenv("ALTASTATA_ACCOUNT_PASSWORD")
    account_id = config.account_id

    if not password:
        raise RuntimeError("ALTASTATA_ACCOUNT_PASSWORD environment variable is required")

    # 3. Connect to AltaStata
    logging.info("Connecting to AltaStata...")
    functions = AltaStataFunctions.from_credentials(user_properties, private_key, port=25333, callback_server_port=None)
    functions.set_password(password)
    register_altastata_functions_for_pytorch(functions, account_id)
    logging.info("Connected to AltaStata account: %s", account_id)

    # 4. Prepare image transforms (data augmentation for training, none for validation)
    base_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        *base_transform.transforms
    ])

    # 5. Load datasets from AltaStata
    logging.info("Loading dataset from: %s", options.data_root)
    dataset_params = {
        "account_id": account_id,
        "root_dir": options.data_root,
        "file_pattern": FILE_PATTERN,
    }
    train_dataset = AltaStataPyTorchDataset(transform=train_transform, **dataset_params)
    validation_dataset = AltaStataPyTorchDataset(transform=base_transform, **dataset_params)

    dataset_size = len(train_dataset)
    logging.info("Found %s images in dataset", dataset_size)

    # 6. Split dataset into training and validation sets
    train_indices, val_indices = split_indices(dataset_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=0,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=0,
    )

    # 7. Train the model
    logging.info("Starting model training...")
    trainer = ModelTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        patience=PATIENCE,
    )
    best_state, metrics = trainer.train()

    # 8. Save training summary
    outputdir.mkdir(parents=True, exist_ok=True)
    write_summary(options.summary_filename, metrics, dataset_size)
    logging.info("Training summary saved to %s", options.summary_filename)

    # 9. Upload model to AltaStata (if output path provided)
    if best_state and options.model_output:
        logging.info("Uploading model to AltaStata: %s", options.model_output)
        train_dataset.save_model(best_state, options.model_output)
        logging.info("Model successfully uploaded to AltaStata")

    # 10. Cleanup
    functions.shutdown()
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
