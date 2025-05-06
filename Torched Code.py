import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import numpy as np
import random

from pycocotools.coco import COCO
from torch.utils.data import Dataset
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoDetectionDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            boxes.append(bbox)
            labels.append(ann['category_id'])

        if self.transforms:
            transformed = self.transforms(image=np.array(img), bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.ids)

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms():
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_detection_model(num_classes=91):  # 80 COCO classes + background
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

import torch
import os

class DetectionTrainer:
    def __init__(self, model, device, save_path="models"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def fit(self, train_loader, val_loader=None, epochs=10):
        best_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else 0.0

            self.history['train_loss'].append(train_loss)
            if val_loader:
                self.history['val_loss'].append(val_loss)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            if val_loader:
                print(f"Val   Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pth"))
                    print("✅ Best model saved.")

import torch
import os
from utils import DetectionUtils  # Assuming utility class is called DetectionUtils

class InferenceEngine:
    def __init__(self, model_class, model_path, device, num_classes=11):
        self.device = device
        self.model = model_class(num_classes=num_classes).to(device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"🔍 Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    def predict_batch(self, dataloader, score_threshold=0.5, num_samples=10, display=True):
        self.model.eval()

        all_images = []
        all_true_boxes = []
        all_true_labels = []
        all_pred_boxes = []
        all_pred_labels = []
        all_scores = []
        iou_scores = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for i in range(len(images)):
                    img = images[i].cpu()
                    tgt = targets[i]
                    out = outputs[i]

                    # Apply score threshold
                    keep = out['scores'] >= score_threshold
                    boxes = out['boxes'][keep].cpu()
                    labels = out['labels'][keep].cpu()
                    scores = out['scores'][keep].cpu()

                    all_images.append(img)
                    all_true_boxes.append(tgt['boxes'])
                    all_true_labels.append(tgt['labels'])
                    all_pred_boxes.append(boxes)
                    all_pred_labels.append(labels)
                    all_scores.append(scores)

                    # IoU per predicted box (against matched GT if available)
                    ious = DetectionUtils.compute_iou_set(boxes, tgt['boxes'])
                    iou_scores.append(ious)

                    if len(all_images) >= num_samples:
                        break
                if len(all_images) >= num_samples:
                    break

        if display:
            DetectionUtils.display_detections(
                all_images, all_pred_boxes, all_pred_labels,
                all_true_boxes, all_true_labels, all_scores, iou_scores
            )

        return {
            "images": all_images,
            "true_boxes": all_true_boxes,
            "true_labels": all_true_labels,
            "pred_boxes": all_pred_boxes,
            "pred_labels": all_pred_labels,
            "scores": all_scores,
            "ious": iou_scores
        }
    def predict_with_uncertainty(self, image_tensor, T=10):
        self.model.eval()
        enable_dropout(self.model)

        preds = [self.model([image_tensor.to(self.device)])[0] for _ in range(T)]
        
        boxes = torch.stack([p['boxes'] for p in preds])
        scores = torch.stack([p['scores'] for p in preds])
        labels = torch.stack([p['labels'] for p in preds])

        mean_boxes = boxes.mean(dim=0)
        std_boxes = boxes.std(dim=0)

        mean_scores = scores.mean(dim=0)
        std_scores = scores.std(dim=0)

        return {
            "mean_boxes": mean_boxes.cpu(),
            "std_boxes": std_boxes.cpu(),
            "labels": labels[0].cpu(),
            "mean_scores": mean_scores.cpu(),
            "std_scores": std_scores.cpu()
        }

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import ImageDraw, ImageFont

class DetectionUtils:
    @staticmethod
    def compute_iou(boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        return interArea / (areaA + areaB - interArea + 1e-5)

    @staticmethod
    def compute_iou_set(pred_boxes, true_boxes):
        ious = []
        for pred in pred_boxes:
            max_iou = 0.0
            for gt in true_boxes:
                iou = DetectionUtils.compute_iou(pred, gt)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
        return ious

    @staticmethod
    def draw_boxes(image_tensor, boxes, labels=None, colors=None):
        image = ToPILImage()(image_tensor).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size

        if colors is None:
            colors = ['red'] * len(boxes)
        if labels is None:
            labels = [""] * len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=2)
            if labels[i]:
                draw.text((x1 + 2, y1 + 2), str(labels[i]), fill=colors[i])

        return image

    @staticmethod
    def display_detections(images, pred_boxes, pred_labels, true_boxes, true_labels, scores, ious, max_display=8):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib import cm

        n = min(len(images), max_display)
        fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(20, 10))
        axs = axs.flatten()

        for i in range(n):
            img = images[i].permute(1, 2, 0).numpy()
            fig_i = axs[i]
            fig_i.imshow(img)
            fig_i.axis('off')

            for box, label, score in zip(pred_boxes[i], pred_labels[i], scores[i]):
                x1, y1, x2, y2 = map(int, box)
                color = cm.viridis(score.item())
                fig_i.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                linewidth=2, edgecolor=color, facecolor='none'))
                fig_i.text(x1, y1, f'{label.item()} {score.item():.2f}', color='white', bbox=dict(facecolor='black'))

            fig_i.set_title(f'IoU Mean: {np.mean(ious[i]):.2f}' if ious[i] else "No match")

        plt.tight_layout()
        plt.show()

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import MultiDigitDetectionDataset
from model import get_detection_model
from trainer import DetectionTrainer
from inference import InferenceEngine

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', type=str, default='path/to/train2017')
    parser.add_argument('--val_img_dir', type=str, default='path/to/val2017')
    parser.add_argument('--train_ann', type=str, default='path/to/annotations/instances_train2017.json')
    parser.add_argument('--val_ann', type=str, default='path/to/annotations/instances_val2017.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # Dataset and Transforms
    train_dataset = CocoDetectionDataset(
        root=args.train_img_dir,
        annFile=args.train_ann,
        transforms=get_train_transforms()
    )
    val_dataset = CocoDetectionDataset(
        root=args.val_img_dir,
        annFile=args.val_ann,
        transforms=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = get_detection_model(num_classes=91)  # COCO has 80 classes + background
    trainer = DetectionTrainer(model, device=device, save_path="models")

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("✅ Loaded saved model.")
    else:
        print("🚀 Starting training...")
        trainer.fit(train_loader, val_loader, epochs=args.epochs)
        torch.save(model.state_dict(), args.model_path)

    # Inference with batch display
    print("🔎 Running inference on validation data...")
    infer_engine = InferenceEngine(get_detection_model, model_path=args.model_path, device=device, num_classes=91)
    infer_engine.predict_batch(val_loader, score_threshold=0.5, num_samples=8, display=True)

    # Optional: Inference with uncertainty on a single image
    print("🧠 Predicting with uncertainty...")
    sample_image, _ = val_dataset[0]
    result = infer_engine.predict_with_uncertainty(sample_image, T=10)

    print("📊 Uncertainty Summary:")
    print("Mean Scores:", result["mean_scores"])
    print("Score Std Devs:", result["std_scores"])

if __name__ == "__main__":
    main()
