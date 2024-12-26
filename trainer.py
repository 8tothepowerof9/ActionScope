import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchmetrics import F1Score
from dataset import ActionDataset, get_data_transforms
from models.baseline import BaselineModel
import pandas as pd
import matplotlib.pyplot as plt
import time
from utils.callbacks import EarlyStopping


class Trainer:
    def __init__(
        self,
        dataloaders,
        model,
        loss_fns,
        optimizer,
        metrics,
        device,
        scheduler=None,
        save=True,
        plot=True,
    ):
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fns = loss_fns
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.scheduler = scheduler  # Does not work with ReduceLROnPlateau
        self.save = save
        self.plot = plot
        self.save_dir = "results/training/"

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_action_score": [],
            "val_action_score": [],
            "train_person_score": [],
            "val_person_score": [],
        }

    def __train__(self):
        start = time.time()
        size = len(self.dataloaders["train"].dataset)
        num_batches = len(self.dataloaders["train"])
        total_loss = 0

        # Metrics
        action_score = self.metrics[0]
        person_score = self.metrics[1]
        action_score.reset()
        person_score.reset()

        self.model.train()

        for batch, (img, action, person) in enumerate(self.dataloaders["train"]):
            img, action, person = (
                img.to(self.device),
                action.to(self.device).long(),
                person.to(self.device).float(),
            )
            action_pred, person_pred = self.model(img)
            person_pred = person_pred.squeeze(1)

            action_loss = self.loss_fns[0](action_pred, action)
            person_loss = self.loss_fns[1](person_pred, person)
            loss = action_loss + person_loss
            total_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            action_score(action_pred.argmax(dim=1), action)
            person_score(torch.sigmoid(person_pred).round(), person)

            # Logging
            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(img)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if self.scheduler:
            self.scheduler.step()
        avg_loss = total_loss / num_batches
        train_action_score = action_score.compute().item()
        train_person_score = person_score.compute().item()

        # Save metrics to history
        self.history["train_loss"].append(avg_loss)
        self.history["train_action_score"].append(train_action_score)
        self.history["train_person_score"].append(train_person_score)

        end = time.time()

        print(
            f"Train Summary: \n Avg Loss: {avg_loss:.4f} | Action F1: {train_action_score:.4f} | Person F1: {train_person_score:.4f} | Time taken: {end-start:.4f}s"
        )

    def __validation__(self):
        start = time.time()
        num_batches = len(self.dataloaders["val"])
        total_loss = 0

        # Metrics
        action_score = self.metrics[0]
        person_score = self.metrics[1]
        action_score.reset()
        person_score.reset()

        self.model.eval()

        with torch.no_grad():
            for img, action, person in self.dataloaders["val"]:
                img, action, person = (
                    img.to(self.device),
                    action.to(self.device).long(),
                    person.to(self.device).float(),
                )
                action_pred, person_pred = self.model(img)
                person_pred = person_pred.squeeze(1)

                action_loss = self.loss_fns[0](action_pred, action)
                person_loss = self.loss_fns[1](person_pred, person)
                loss = action_loss + person_loss
                total_loss += loss

                # Update metrics
                action_score(action_pred.argmax(dim=1), action)
                person_score(torch.sigmoid(person_pred).round(), person)

        avg_loss = total_loss / num_batches
        val_action_score = action_score.compute().item()
        val_person_score = person_score.compute().item()

        # Save metrics to history
        self.history["val_loss"].append(avg_loss.item())
        self.history["val_action_score"].append(val_action_score)
        self.history["val_person_score"].append(val_person_score)

        end = time.time()

        print(
            f"Validation Summary: \n Avg Loss: {avg_loss:.4f} | Action F1: {val_action_score:.4f} | Person F1: {val_person_score:.4f} | Time taken: {end-start:.4f}s"
        )

    def fit(self, epochs):
        early_stopper = EarlyStopping(patience=3, min_delta=0.001)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.__train__()
            self.__validation__()

            # Get the last validation lost in history
            if early_stopper.early_stop(self.history["val_loss"][-1], self.model):
                print("Early Stopped!")
                break

        # Save best model
        if self.save:
            torch.save(
                early_stopper.best_model_state,
                "models/checkpoints/" + self.model.name + ".pt",
            )
            print("Model checkpoint saved!")

        if self.save:
            pd.DataFrame(self.history).to_csv(
                f"logs/{self.model.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")

        return self.history

    def plot_history(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot losses
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend()

        # Plot F1 Scores
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_action_score"], label="Train Action Score")
        plt.plot(
            epochs, self.history["val_action_score"], label="Validation Action Score"
        )
        plt.plot(epochs, self.history["train_person_score"], label="Train Person Score")
        plt.plot(
            epochs, self.history["val_person_score"], label="Validation Person Score"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("History")
        plt.legend()
        plt.tight_layout()

        if self.save:
            plt.savefig(self.save_dir + f"{self.model.name}.png")

        if self.plot:
            plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/data.csv")

    train_ds = ActionDataset(data, split="train", transform=get_data_transforms())
    val_ds = ActionDataset(data, split="val", transform=get_data_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
    )

    dataloaders = {"train": train_loader, "val": val_loader}

    loss_fns = [nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]
    device = "cuda"
    metrics = [
        F1Score(task="multiclass", num_classes=40, average="micro").to(device),
        F1Score(task="binary", average="micro").to(device),
    ]
    model = BaselineModel(finetune=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(
        dataloaders=dataloaders,
        model=model,
        loss_fns=loss_fns,
        optimizer=optimizer,
        metrics=metrics,
        device=device,
    )
    trainer.fit(epochs=10, plot=True)
