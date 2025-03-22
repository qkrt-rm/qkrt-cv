import os
import torch
import random
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as Ft
import torchvision.transforms as transforms


class armourDataset(Dataset):
    def __init__(self, data_dir, train=True, img_dir="images", label_dir="labels",
                 file_list=None, transform_list=None, label_transform=None, noise_threshold=0.5):
        self.img_dir = os.path.join(data_dir, img_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.train = train
        self.transform_list = transform_list or [transforms.Compose([])]
        self.label_transform = label_transform
        self.noise_threshold = noise_threshold

        if file_list:
            with open(os.path.join(data_dir, file_list), "r") as f:
                self.image_files = [line.strip() for line in f.readlines()]
        else:
            self.image_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(self.img_dir)
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ]

        self.original_len = len(self.image_files)

    def __len__(self):
        return len(self.image_files) * len(self.transform_list)

    def __getitem__(self, idx):
        image_index = idx % self.original_len
        transform_index = idx // self.original_len

        img_name = self.image_files[image_index]
        img_path = self._find_image_path(os.path.join(self.img_dir, img_name))
        label_path = os.path.join(self.label_dir, f"{img_name}.txt")

        image = read_image(img_path).float() / 255.0
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    vals = list(map(int, line.strip().split()))
                    if len(vals) < 9:
                        continue
                    cls = vals[0]
                    coords = vals[1:]
                    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                    xs, ys = zip(*points)
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    boxes.append(bbox)
                    labels.append(cls)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            return None  # Skip image with no boxes

        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        image, boxes = self.resize(image, boxes, (227, 227))
        image, boxes, labels, occlusion_flags = self.add_random_noise(image, boxes, labels)

        if self.transform_list:
            image = self.transform_list[transform_index](image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "occluded": occlusion_flags
        }

        return image, target

    def resize(self, image, boxes, target_size=(227, 227)):
        h, w = image.shape[1], image.shape[2]
        image = Ft.resize(image, target_size)

        if boxes.numel() > 0:
            h_scale = target_size[0] / h
            w_scale = target_size[1] / w
            boxes[:, [0, 2]] *= w_scale
            boxes[:, [1, 3]] *= h_scale

        return image, boxes

    def add_random_noise(self, image, boxes, labels, mean=0.0, std=0.2):
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise
        noisy_image = noisy_image.clamp(0.0, 1.0)

        occlusion_flags = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                occlusion_flags.append(True)
                continue
            cropped_original = image[:, y1:y2, x1:x2]
            cropped_noisy = noisy_image[:, y1:y2, x1:x2]
            if cropped_original.numel() == 0:
                occlusion_flags.append(True)
                continue
            diff = torch.abs(cropped_noisy - cropped_original).mean().item()
            occlusion_flags.append(diff > self.noise_threshold)

        return noisy_image, boxes, labels, occlusion_flags

    def _find_image_path(self, base_path):
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            full_path = base_path + ext
            if os.path.exists(full_path):
                return full_path
        raise FileNotFoundError(f"No image file found for base path: {base_path}")
