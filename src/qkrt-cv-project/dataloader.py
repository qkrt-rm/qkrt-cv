import os
from torch.utils.data import DataLoader
from dataset import armourDataset  # Import the correct class


class armourData:
    def __init__(self, data_dir=[], img_dir="images", label_dir="labels",
                 train_file="train_plates.txt", test_file="test_plates.txt", dataset=[], train=False,
                 transform_list=None):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.train_file = train_file
        self.test_file = test_file
        self.training_data = []
        self.test_data = []
        self.transform_list = transform_list

        if dataset:
            if train:
                self.training_data = dataset
            else:
                self.test_data = dataset
        elif not data_dir:
            print("Error: A dataset or a data directory is required")
            exit()

    def train_loader(self, training_data=[], batch_size=64):
        if not training_data:
            if not self.training_data:
                training_data = armourDataset(
                    data_dir=self.data_dir,
                    img_dir=self.img_dir,
                    label_dir=self.label_dir,
                    file_list=self.train_file,
                    transform_list=self.transform_list
                )
            else:
                training_data = self.training_data

        return DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    def test_loader(self, test_data=[], batch_size=64):
        if not test_data:
            if not self.test_data:
                test_data = armourDataset(
                    data_dir=self.data_dir,
                    img_dir=self.img_dir,
                    label_dir=self.label_dir,
                    file_list=self.test_file,
                    transform_list=self.transform_list
                )
            else:
                test_data = self.test_data

        return DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
