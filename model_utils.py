import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import Counter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    idx_to_class = checkpoint["idx_to_class"]
    if isinstance(idx_to_class, dict):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    model = build_model(num_classes=len(idx_to_class))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    img_size = checkpoint.get("img_size", 224)
    transform = get_transform(img_size)

    return model, transform, idx_to_class


def predict_image(pil_image, model, transform, idx_to_class, device):
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()
    label = idx_to_class[pred_idx]

    return label, confidence, probs.squeeze().cpu().tolist()


def extract_frames(video_path, sample_every=30, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % sample_every == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)

            if len(frames) >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frames


def predict_video(video_path, model, transform, idx_to_class, device):
    frames = extract_frames(video_path, sample_every=30, max_frames=20)

    if not frames:
        return {
            "final_label": "Could not read video",
            "confidence": 0.0,
            "frame_predictions": []
        }

    frame_predictions = []

    for frame in frames:
        label, confidence, probs = predict_image(frame, model, transform, idx_to_class, device)
        frame_predictions.append({
            "label": label,
            "confidence": confidence
        })

    labels = [item["label"] for item in frame_predictions]
    counts = Counter(labels)
    final_label = counts.most_common(1)[0][0]

    selected_confidences = [
        x["confidence"] for x in frame_predictions if x["label"] == final_label
    ]
    final_confidence = sum(selected_confidences) / len(selected_confidences)

    return {
        "final_label": final_label,
        "confidence": final_confidence,
        "frame_predictions": frame_predictions
    }