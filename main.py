import sys
import json
import pandas as pd
from dataset import ActionDataset, get_data_transforms, get_aug_transforms
from torchmetrics import F1Score
import torch
from torch import nn, optim
from models import *
from torch.utils.data import DataLoader
from trainer import Trainer
from eval import Evaluator


def main():
    if len(sys.argv) != 2:
        print("Invalid arguments. Usage: python script.py <config.json>")
        sys.exit(1)

    config_file = "cfg/" + sys.argv[1]

    try:
        # Load JSON configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        # Load CSV data
        csv_data = pd.read_csv("data/data.csv")

        # Prepare datasets
        if config["data"]["aug"] == True:
            transforms = get_aug_transforms()
        else:
            transforms = get_data_transforms()

        train_ds = ActionDataset(csv_data, split="train", transform=transforms)
        val_ds = ActionDataset(csv_data, split="val", transform=transforms)
        test_ds = ActionDataset(csv_data, split="test", transform=transforms)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        # Define loss functions, metrics, and model
        loss_fns = [
            nn.CrossEntropyLoss(),
            nn.BCEWithLogitsLoss(pos_weight=train_ds.wts),
        ]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = [
            F1Score(task="multiclass", num_classes=40, average="micro").to(device),
            F1Score(task="binary", average="micro").to(device),
        ]

        if config["model"]["type"] == "baseline":
            model = BaselineModel(
                finetune=config["training"]["finetune"],
                name=config["model"]["name"],
                in_channels=config["model"]["in_channels"],
                dropout_rate=config["model"]["dropout_rate"],
            )
        elif config["model"]["type"] == "attention":
            model = AttentionModel(
                finetune=config["training"]["finetune"],
                name=config["model"]["name"],
                reduction_ratio=config["model"]["reduction_ratio"],
                kernel_size=config["model"]["kernel_size"],
                dropout_rate=config["model"]["dropout_rate"],
                in_channels=config["model"]["in_channels"],
            )

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

        # Create Trainer and start training
        trainer = Trainer(
            dataloaders=dataloaders,
            model=model,
            loss_fns=loss_fns,
            optimizer=optimizer,
            metrics=metrics,
            device=device,
            plot=config["training"]["plot"],
            save=config["training"]["save"],
        )
        trainer.fit(epochs=config["training"]["epochs"])
        trainer.plot_history()

        evaluator = Evaluator(
            model,
            dataloaders["test"],
            device,
            action_mapping=train_ds.action_encoder.classes_,
            person_mapping=train_ds.person_encoder.classes_,
            loss_fns=loss_fns,
            metrics=metrics,
            save=config["eval"]["save"],
            plot=config["eval"]["plot"],
        )

        evaluator.eval()
        evaluator.heatmap()

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except KeyError as e:
        print(f"Error: Missing key in configuration: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
