import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


WIDTH, HEIGHT = 224, 224


class ActionDataset(Dataset):
    """
    This class process raw data frame data and split it into train, validation and test sets.
    """

    def __init__(
        self,
        data,
        split="train",
        transform=None,
        action_transform=None,
        person_transform=None,
    ):
        # Preprocess data
        self.split = split
        self.__preprocess_data__(data)

        self.img_dir = "data/Images/"
        self.transform = transform
        self.action_transform = action_transform
        self.person_transform = person_transform

        if split == "train":
            self.__handle_imbalance__()
            self.__cal_class_weights__()

    def __handle_imbalance__(self):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(
            self.data.drop("Class", axis=1), self.data["Class"]
        )
        data = pd.concat(
            [
                pd.DataFrame(
                    X_resampled, columns=self.data.drop("Class", axis=1).columns
                ),
                pd.DataFrame(y_resampled, columns=["Class"]),
            ],
            axis=1,
        )
        self.data = data[["FileName", "Class", "MoreThanOnePerson"]]

    def __cal_class_weights__(self):
        counts = np.bincount(self.data["MoreThanOnePerson"])
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        weights = weights / weights.sum()
        self.wts = weights[1] / weights[0]

    def __preprocess_data__(self, data):
        self.action_encoder = LabelEncoder()
        self.person_encoder = LabelEncoder()
        data["Class"] = self.action_encoder.fit_transform(data["Class"])
        data["MoreThanOnePerson"] = self.person_encoder.fit_transform(
            data["MoreThanOnePerson"]
        )

        # Split data
        train, test = train_test_split(data, test_size=0.15, random_state=42)
        train, val = train_test_split(train, test_size=0.1765, random_state=42)

        if self.split == "train":
            self.data = train
        elif self.split == "val":
            self.data = val
        elif self.split == "test":
            self.data = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = decode_image(os.path.join(self.img_dir, img_path))
        image = to_pil_image(image)
        action = self.data.iloc[idx, 1]
        person = self.data.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        if self.action_transform:
            action = self.action_transform(action)

        if self.person_transform:
            person = self.person_transform(person)

        return image, action, person


def get_data_transforms():
    return transforms.Compose(
        [transforms.Resize((WIDTH, HEIGHT)), transforms.ToTensor()]
    )


def get_aug_transforms():
    return transforms.Compose(
        [
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(3),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
        ]
    )


def get_target_transforms():
    return None


if __name__ == "__main__":
    data = pd.read_csv("data/data.csv")
    dataset = ActionDataset(
        data,
        split="train",
        transform=get_data_transforms(),
        action_transform=get_target_transforms(),
        person_transform=get_target_transforms(),
    )
    print(dataset.action_encoder.classes_)

    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True
    )

    for img, action, person in dataloader:
        print(img.shape)
        print(action.shape)
        print(person.shape)
        break
