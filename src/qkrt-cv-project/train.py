import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ObjectDetectionModel
from dataset import armourDataset

DATA_DIR = "dataset"
TRAIN_FILE = "train_plates.txt"
TEST_FILE = "test_plates.txt"
IMG_DIR = "images"
LABEL_DIR = "labels"
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_CLASSES = 3  # background + class1 + class2

# Optional: re-split train/test
def split_dataset(data_dir, output_train_file="train_plates.txt", output_test_file="test_plates.txt", test_ratio=0.1):
    image_dir = os.path.join(data_dir, IMG_DIR)
    image_filenames = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
    random.shuffle(image_filenames)

    split_idx = int(len(image_filenames) * (1 - test_ratio))
    train_set = image_filenames[:split_idx]
    test_set = image_filenames[split_idx:]

    with open(os.path.join(data_dir, output_train_file), "w") as f:
        for name in train_set:
            f.write(f"{name}\n")
    with open(os.path.join(data_dir, output_test_file), "w") as f:
        for name in test_set:
            f.write(f"{name}\n")
    print(f"Dataset split: {len(train_set)} train, {len(test_set)} test")

split_dataset(DATA_DIR, TRAIN_FILE, TEST_FILE)

# Transforms
transform_list = [
    transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
    transforms.Compose([transforms.GaussianBlur(3)])
]

# Dataloader
train_dataset = armourDataset(
    data_dir=DATA_DIR,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
    file_list=TRAIN_FILE,
    transform_list=transform_list
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: [sample for sample in x if sample is not None]  # skip None entries
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ObjectDetectionModel(num_classes=NUM_CLASSES).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training Loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        if not batch:
            continue
        images, targets = zip(*batch)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    lr_scheduler.step()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")

# Save model
model.save("fasterrcnn_model.pth")
print("Model saved to models/fasterrcnn_model.pth")
