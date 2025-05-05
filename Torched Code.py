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

        # Random position for placing the digit in a 75x75 canvas
        xmin = np.random.randint(0, 48)
        ymin = np.random.randint(0, 48)
        xmax = xmin + 28
        ymax = ymin + 28

        # Create 75x75 canvas
        canvas = Image.new("L", (75, 75), color=0)
        canvas.paste(img, (xmin, ymin))

        if self.transform:
            canvas = self.transform(canvas)

        # Normalize bbox coordinates to [0, 1] range
        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32) / 75.0

        # One-hot encode label
        label_onehot = torch.zeros(10)
        label_onehot[label] = 1.0

        return canvas, (label_onehot, bbox)

    @staticmethod
    def to_numpy(dataloader, num_batches=1):
        images, labels, bboxes = [], [], []
        for i, (x, (y, bb)) in enumerate(dataloader):
            images.extend(x.numpy())
            labels.extend(torch.argmax(y, dim=1).numpy())
            bboxes.extend(bb.numpy())
            if i + 1 >= num_batches:
                break
        return np.array(images), np.array(labels), np.array(bboxes)

import torch.nn as nn

class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # 75 → 73
            nn.ReLU(),
            nn.AvgPool2d(2),                 # 73 → 36

            nn.Conv2d(16, 32, kernel_size=3),  # 36 → 34
            nn.ReLU(),
            nn.AvgPool2d(2),                   # 34 → 17

            nn.Conv2d(32, 64, kernel_size=3),  # 17 → 15
            nn.ReLU(),
            nn.AvgPool2d(2)                    # 15 → 7
        )

        # Shared dense layer
        self.flatten = nn.Flatten()
        self.shared_dense = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Linear(128, 10)

        # Bounding box regression head
        self.bbox_regressor = nn.Linear(128, 4)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.shared_dense(x)

        class_output = self.classifier(x)
        bbox_output = self.bbox_regressor(x)

        return class_output, bbox_output

import torch
import torch.nn as nn
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

        # History tracking
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'train_bbox_loss': [],
            'val_bbox_loss': []
        }

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_cls_loss, total_bbox_loss = 0.0, 0.0
        correct, total = 0, 0

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

            # Metrics
            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            preds = torch.argmax(class_logits, dim=1)
            true = torch.argmax(labels, dim=1)
            correct += (preds == true).sum().item()
            total += labels.size(0)

        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_bbox_loss = total_bbox_loss / len(dataloader)
        acc = correct / total

        return avg_cls_loss, avg_bbox_loss, acc

    def validate(self, dataloader):
        self.model.eval()
        total_cls_loss, total_bbox_loss = 0.0, 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, (labels, bboxes) in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                class_logits, bbox_preds = self.model(images)

                cls_loss = self.cls_criterion(class_logits, torch.argmax(labels, dim=1))
                bbox_loss = self.bbox_criterion(bbox_preds, bboxes)

                total_cls_loss += cls_loss.item()
                total_bbox_loss += bbox_loss.item()
                preds = torch.argmax(class_logits, dim=1)
                true = torch.argmax(labels, dim=1)
                correct += (preds == true).sum().item()
                total += labels.size(0)

        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_bbox_loss = total_bbox_loss / len(dataloader)
        acc = correct / total

        return avg_cls_loss, avg_bbox_loss, acc

    def fit(self, train_loader, val_loader, epochs=10):
        best_acc = 0.0

        for epoch in range(epochs):
            train_cls_loss, train_bbox_loss, train_acc = self.train_one_epoch(train_loader)
            val_cls_loss, val_bbox_loss, val_acc = self.validate(val_loader)

            # Store history
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['train_bbox_loss'].append(train_bbox_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['val_bbox_loss'].append(val_bbox_loss)
            self.history['val_acc'].append(val_acc)

            # Display progress
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train - Acc: {train_acc:.4f}, Cls Loss: {train_cls_loss:.4f}, BBox Loss: {train_bbox_loss:.4f}")
            print(f"Val   - Acc: {val_acc:.4f}, Cls Loss: {val_cls_loss:.4f}, BBox Loss: {val_bbox_loss:.4f}")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pth"))
                print("✅ Best model saved.")

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image, ImageDraw, ImageFont

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
    def draw_bounding_boxes_on_image_array(image_tensor, boxes, colors=None, labels=None):
        image = ToPILImage()(image_tensor.cpu()).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size

        if colors is None:
            colors = ['red'] * len(boxes)
        if labels is None:
            labels = [None] * len(boxes)

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = [int(v * w) for v in box]
            draw.rectangle([xmin, ymin, xmax, ymax], outline=colors[i], width=2)
            if labels[i]:
                draw.text((xmin + 2, ymax + 2), labels[i], fill=colors[i])

        return image

    @staticmethod
    def display_batch(images, pred_labels, true_labels, pred_bboxes, true_bboxes, iou_scores=None, max_display=10):
        n = min(max_display, len(images))
        fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(18, 8))
        axs = axs.flatten()

        for i in range(n):
            image = images[i]
            true_box = true_bboxes[i]
            pred_box = pred_bboxes[i]
            true_label = str(true_labels[i])
            pred_label = str(pred_labels[i])

            drawn = Utils.draw_bounding_boxes_on_image_array(
                image_tensor=image,
                boxes=[true_box, pred_box],
                colors=['green', 'red'],
                labels=[f"True: {true_label}", f"Pred: {pred_label}"]
            )

            axs[i].imshow(drawn)
            axs[i].axis("off")

            # IoU Title
            if iou_scores is not None:
                iou = iou_scores[i]
                axs[i].set_title(f"IoU: {iou:.2f}", color='green' if iou >= 0.5 else 'red')
            else:
                axs[i].set_title(f"{true_label} vs {pred_label}", color='black' if true_label == pred_label else 'red')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(14, 4))

        plt.subplot(1, 3, 1)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history['train_cls_loss'], label='Train Cls Loss')
        plt.plot(history['val_cls_loss'], label='Val Cls Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(history['train_bbox_loss'], label='Train BBox Loss')
        plt.plot(history['val_bbox_loss'], label='Val BBox Loss')
        plt.title('Bounding Box Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_digits_from_local_fonts(n):
        font_labels = []
        img = Image.new('L', (75 * n, 75), color=(0,))
        font1_path = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf/DejaVuSansMono-Oblique.ttf")
        font2_path = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf/STIXGeneral.ttf")
        font1 = ImageFont.truetype(font1_path, 25)
        font2 = ImageFont.truetype(font2_path, 25)
        draw = ImageDraw.Draw(img)

        for i in range(n):
            font_labels.append(i % 10)
            font = font1 if i < 10 else font2
            draw.text((75 * i, 0), str(i % 10), fill=(255,), font=font)

        font_digits = np.array(img.getdata()).reshape((75, 75 * n)) / 255.0
        font_digits = np.reshape(font_digits, (n, 75, 75, 1))

        return font_digits, np.array(font_labels)

import torch
import os
import numpy as np

class InferenceEngine:
    def __init__(self, model_class, model_path, device):
        self.device = device
        self.model = model_class().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"🔍 Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

    def predict_batch(self, dataloader, num_samples=10, display=True):
        self.model.eval()
        images_out, true_labels_out, pred_labels_out = [], [], []
        true_bboxes_out, pred_bboxes_out, iou_scores = [], [], []

        with torch.no_grad():
            for images, (labels, bboxes) in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                class_logits, bbox_preds = self.model(images)
                preds = torch.argmax(class_logits, dim=1)
                true = torch.argmax(labels, dim=1)

                # Move everything to CPU for visualization
                images_out.extend(images.cpu())
                true_labels_out.extend(true.cpu().numpy())
                pred_labels_out.extend(preds.cpu().numpy())
                true_bboxes_out.extend(bboxes.cpu().numpy())
                pred_bboxes_out.extend(bbox_preds.cpu().numpy())

                if len(images_out) >= num_samples:
                    break

        images_out = images_out[:num_samples]
        true_labels_out = true_labels_out[:num_samples]
        pred_labels_out = pred_labels_out[:num_samples]
        true_bboxes_out = true_bboxes_out[:num_samples]
        pred_bboxes_out = pred_bboxes_out[:num_samples]

        for pb, tb in zip(pred_bboxes_out, true_bboxes_out):
            iou_scores.append(Utils.compute_iou(pb, tb))

        if display:
            Utils.display_batch(
                images_out,
                pred_labels_out,
                true_labels_out,
                pred_bboxes_out,
                true_bboxes_out,
                iou_scores
            )

        return {
            "images": images_out,
            "true_labels": true_labels_out,
            "pred_labels": pred_labels_out,
            "true_bboxes": true_bboxes_out,
            "pred_bboxes": pred_bboxes_out,
            "iou_scores": iou_scores
        }

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Datasets and Dataloaders
    train_dataset = DigitBBoxDataset(train=True, transform=transform)
    val_dataset = DigitBBoxDataset(train=False, transform=transform)
    test_dataset = DigitBBoxDataset(train=False, transform=transform)  # Could replace with real test split

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = DigitModel()
    trainer = Trainer(model, device=device, save_path="models")

    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Loaded saved model.")
    else:
        print("🚀 Starting training...")
        trainer.fit(train_loader, val_loader, epochs=10)
        torch.save(model.state_dict(), model_path)

    # Plot training history
    Utils.plot_training_history(trainer.history)

    # Inference and Evaluation
    print("🔎 Running inference on test data...")
    infer_engine = InferenceEngine(DigitModel, model_path=model_path, device=device)
    infer_engine.predict_batch(test_loader, num_samples=10, display=True)

    # Optional: Visualize synthetic digits
    print("🧪 Visualizing font-generated synthetic digits...")
    font_digits, font_labels = Utils.create_digits_from_local_fonts(10)
    fig, axs = plt.subplots(1, 10, figsize=(18, 3))
    for i in range(10):
        axs[i].imshow(font_digits[i].squeeze(), cmap='gray')
        axs[i].set_title(f"Label: {font_labels[i]}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
