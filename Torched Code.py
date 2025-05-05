import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import numpy as np

class DigitBBoxDataset(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        self.dataset = MNIST(root=root, train=train, download=True)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Random bbox position
        xmin = np.random.randint(0, 48)
        ymin = np.random.randint(0, 48)
        xmax = xmin + 28
        ymax = ymin + 28

        # Create blank canvas and paste image
        canvas = Image.new("L", (75, 75), color=0)
        canvas.paste(img, (xmin, ymin))

        if self.transform:
            canvas = self.transform(canvas)

        # Normalize bbox
        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32) / 75.0

        # One-hot encode label
        label_onehot = torch.zeros(10)
        label_onehot[label] = 1.0

        return canvas, (label_onehot, bbox)

import torch.nn as nn

class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()

        # Shared CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0),  # (75 - 3 + 1) = 73 -> 36 after pool
            nn.ReLU(),
            nn.AvgPool2d(2),  # 73 -> 36

            nn.Conv2d(16, 32, kernel_size=3),  # 36 - 3 + 1 = 34 -> 17
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),  # 17 - 3 + 1 = 15 -> 7
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # Flatten and dense shared input
        self.flatten = nn.Flatten()

        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Linear(128, 10)

        # Bounding box head
        self.bbox_regressor = nn.Linear(128, 4)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dense(x)

        class_output = self.classifier(x)
        bbox_output = self.bbox_regressor(x)

        return class_output, bbox_output

import torch
import torch.nn.functional as F
import os

class Trainer:
    def __init__(self, model, device, save_path="models"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.MSELoss()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_cls_loss, total_bbox_loss = 0.0, 0.0

        for images, (labels, bboxes) in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            bboxes = bboxes.to(self.device)

            self.optimizer.zero_grad()
            class_logits, bbox_preds = self.model(images)

            cls_loss = self.cls_criterion(class_logits, torch.argmax(labels, dim=1))
            bbox_loss = self.bbox_criterion(bbox_preds, bboxes)

            total_loss = cls_loss + bbox_loss
            total_loss.backward()
            self.optimizer.step()

            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()

        return total_cls_loss / len(dataloader), total_bbox_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        bbox_loss_total = 0.0

        with torch.no_grad():
            for images, (labels, bboxes) in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                class_logits, bbox_preds = self.model(images)
                pred_labels = torch.argmax(class_logits, dim=1)
                true_labels = torch.argmax(labels, dim=1)

                correct += (pred_labels == true_labels).sum().item()
                total += labels.size(0)

                bbox_loss = self.bbox_criterion(bbox_preds, bboxes)
                bbox_loss_total += bbox_loss.item()

        accuracy = correct / total
        avg_bbox_loss = bbox_loss_total / len(dataloader)
        return accuracy, avg_bbox_loss

    def fit(self, train_loader, val_loader, epochs=10):
        best_acc = 0.0
        for epoch in range(epochs):
            cls_loss, bbox_loss = self.train_one_epoch(train_loader)
            val_acc, val_bbox_loss = self.validate(val_loader)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss - Class: {cls_loss:.4f}, BBox: {bbox_loss:.4f}")
            print(f"Val   Loss - Acc: {val_acc:.4f}, BBox: {val_bbox_loss:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pth"))
                print("✅ Best model saved.")

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import ImageDraw

class Utils:
    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        return interArea / (boxAArea + boxBArea - interArea + 1e-5)

    @staticmethod
    def draw_bounding_box(image_tensor, bbox, color='red'):
        image = ToPILImage()(image_tensor.cpu()).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size
        xmin, ymin, xmax, ymax = [int(v * w) for v in bbox]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        return image

    @staticmethod
    def display_batch(images, pred_labels, true_labels, pred_bboxes, true_bboxes):
        n = min(3, len(images))
        fig, axs = plt.subplots(1, n, figsize=(15, 5))
        for i in range(n):
            image = images[i]
            true_box = true_bboxes[i]
            pred_box = pred_bboxes[i]

            img = Utils.draw_bounding_box(image, true_box, color='green')  # ground truth
            img = Utils.draw_bounding_box(transforms.ToTensor()(img), pred_box, color='red')  # prediction

            axs[i].imshow(img)
            axs[i].set_title(f"True: {true_labels[i]}, Pred: {pred_labels[i]}")
            axs[i].axis("off")
        plt.tight_layout()
        plt.show()

import torch
import os

class InferenceEngine:
    def __init__(self, model_class, model_path, device):
        self.device = device
        self.model = model_class().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"🔍 Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    def predict_batch(self, dataloader, num_samples=3):
        self.model.eval()
        images_out, true_labels_out, pred_labels_out = [], [], []
        true_bboxes_out, pred_bboxes_out = [], []

        with torch.no_grad():
            for images, (labels, bboxes) in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                class_logits, bbox_preds = self.model(images)

                preds = torch.argmax(class_logits, dim=1)
                true = torch.argmax(labels, dim=1)

                images_out.extend(images.cpu())
                true_labels_out.extend(true.cpu().numpy())
                pred_labels_out.extend(preds.cpu().numpy())

                true_bboxes_out.extend(bboxes.cpu().numpy())
                pred_bboxes_out.extend(bbox_preds.cpu().numpy())

                if len(images_out) >= num_samples:
                    break

        Utils.display_batch(
            images_out[:num_samples],
            pred_labels_out[:num_samples],
            true_labels_out[:num_samples],
            pred_bboxes_out[:num_samples],
            true_bboxes_out[:num_samples]
        )

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import all classes
# from your_module import DigitBBoxDataset, DigitModel, Trainer, InferenceEngine, Utils

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = DigitBBoxDataset(train=True, transform=transform)
    val_dataset = DigitBBoxDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model and trainer
    model = DigitModel()
    trainer = Trainer(model, device=device, save_path="models")

    # Train the model
    trainer.fit(train_loader, val_loader, epochs=10)

    # Inference
    infer = InferenceEngine(DigitModel, model_path="models/best_model.pth", device=device)
    infer.predict_batch(val_loader, num_samples=3)

if __name__ == "__main__":
    main()
