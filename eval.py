import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import random
import time


class Evaluator:
    def __init__(
        self,
        model,
        dataloader,
        device,
        action_mapping,
        person_mapping,
        loss_fns,
        metrics,
        save=True,
        plot=True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.save_dir = "results/"
        self.save = save
        self.plot = plot
        self.device = device
        self.action_mapping = action_mapping
        self.person_mapping = person_mapping
        self.loss_fns = loss_fns
        self.metrics = metrics

    def heatmap(self):
        self.model.eval()
        all_action_preds, all_action_labels = [], []
        all_person_preds, all_person_labels = [], []

        with torch.no_grad():
            for img, action, person in self.dataloader:
                img, action, person = (
                    img.to(self.device),
                    action.to(self.device).long(),
                    person.to(self.device).float(),
                )
                action_pred, person_pred = self.model(img)
                action_pred, person_pred = action_pred.argmax(
                    dim=1
                ), person_pred.squeeze(1)

                all_action_preds.extend(action_pred.cpu().numpy())
                all_action_labels.extend(action.cpu().numpy())
                all_person_preds.extend(
                    torch.sigmoid(person_pred).round().cpu().numpy()
                )
                all_person_labels.extend(person.cpu().numpy())

        action_cmat = confusion_matrix(all_action_labels, all_action_preds)
        person_cmat = confusion_matrix(all_person_labels, all_person_preds)

        # Retrieve class names from mappings
        action_classes = [
            self.action_mapping[i] for i in range(len(self.action_mapping))
        ]
        person_classes = [
            self.person_mapping[i] for i in range(len(self.person_mapping))
        ]

        # Plot action and person confusion matrices
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(
            action_cmat,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[0],
            xticklabels=action_classes,
            yticklabels=action_classes,
        )
        ax[0].set_title("Action Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")

        sns.heatmap(
            person_cmat,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[1],
            xticklabels=person_classes,
            yticklabels=person_classes,
        )
        ax[1].set_title("Person Confusion Matrix")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")

        if self.save:
            plt.savefig(self.save_dir + f"{self.model.name}_confusion_matrix.png")

        if self.plot:
            plt.show()

    def eval(self):
        start = time.time()

        test_loss = 0
        num_batches = len(self.dataloader)
        # Metrics
        action_score = self.metrics[0]
        person_score = self.metrics[1]
        action_score.reset()
        person_score.reset()
        self.model.eval()

        with torch.no_grad():
            for img, action, person in self.dataloader:
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
                test_loss += loss

                # Update metrics
                action_score(action_pred.argmax(dim=1), action)
                person_score(torch.sigmoid(person_pred).round(), person)

        avg_loss = test_loss / num_batches
        test_action_score = action_score.compute().item()
        test_person_score = person_score.compute().item()

        end = time.time()

        print(
            f"Test Summary: \n Avg Loss: {avg_loss:.4f} | Action F1: {test_action_score:.4f} | Person F1: {test_person_score:.4f} | Time taken: {end-start:.4f}s"
        )

    def predict_action(self, action_class):
        # Accept only idx for action_class
        self.model.eval()

        with torch.no_grad():
            for img, action, _ in self.dataloader:
                # Filter out images with the given action class
                match_idx = torch.where(action == action_class)[0]
                if len(match_idx) == 0:
                    continue

                # Select a random matching sample
                idx = random.choice(match_idx)
                sample_img = img[idx].unsqueeze(0).to(self.device)
                action = action[idx].unsqueeze(0).to(self.device).long()

                # Predict
                action_pred, _ = self.model(sample_img)
                action_pred = action_pred.argmax(dim=1).item()

                # Plot the sample with the prediction
                img = img.squeeze(0).permute(1, 2, 0).numpy()
                plt.imshow(img)
                plt.axis("off")
                plt.title(
                    f"Predicted: {self.action_mapping[action_pred]} \n Actual: {self.action_mapping[action.item()]}"
                )

                if self.plot:
                    plt.show()

                if self.save:
                    plt.savefig(
                        self.save_dir + f"{self.model.name}_action_{action_class}.png"
                    )

                return  # Exit after the first match

    def predict_person(self, person_class):
        # Given a MoreThanOnePerson status (0 or 1), the model predicts and plots a sample
        self.model.eval()

        with torch.no_grad():
            for img, _, person in self.dataloader:
                # Filter out person class
                match_idx = torch.where(person == person_class)[0]
                if len(match_idx) == 0:
                    continue

                # Select a random matching sample
                idx = random.choice(match_idx)
                sample_img = img[idx].unsqueeze(0).to(self.device)
                person = person[idx].unsqueeze(0).to(self.device)

                # Predict on random sample
                _, person_pred = self.model(sample_img)
                person_pred = person_pred.squeeze(1)

                # Plot sample and prediction
                img = img.squeeze(0).permute(1, 2, 0).numpy()
                plt.imshow(img)
                plt.axis("off")
                plt.title(
                    f"Predicted: {self.person_mapping[person_pred]} \n Actual: {self.person_mapping[person.item()]}"
                )

                if self.plot:
                    plt.show()

                if self.save:
                    plt.savefig(
                        self.save_dir + f"{self.model_name}_person_{person_class}.png"
                    )

    def action_activation_map(self, action):
        # Given an action class, plots the activation map of the model for that class
        pass

    def person_activation_map(self, person_num):
        # Given a MoreThanOnePerson status, plots the activation map of the model for that status
        pass
