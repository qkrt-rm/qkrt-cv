import torch
from torch import nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import os


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def save(self, name: str) -> None:
        os.makedirs("models", exist_ok=True)
        torch.save(self.state_dict(), os.path.join("models", name))

    def load(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
            return True
        return False
