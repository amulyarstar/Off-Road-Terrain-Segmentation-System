import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# CONFIG (MATCH TRAINING)
# =========================
n_classes = 11   # ✅ MUST MATCH YOUR TRAINED MODEL

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    600: 6,   # ✅ Unknown class
    700: 7,
    800: 8,
    7100: 9,
    10000: 10
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Unknown',
    'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_PALETTE = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [0, 200, 0],
    [210, 180, 140],
    [139, 90, 43],
    [128, 128, 0],
    [255, 165, 0],   # Unknown
    [139, 69, 19],
    [169, 169, 169],
    [160, 82, 45],
    [135, 206, 235],
], dtype=np.uint8)


# =========================
# DATASET
# =========================
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.ids = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        img = Image.open(os.path.join(self.image_dir, id_)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, id_))

        mask = np.array(mask)
        new_mask = np.zeros_like(mask)

        for k, v in value_map.items():
            new_mask[mask == k] = v

        mask = Image.fromarray(new_mask.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask) * 255

        return img, mask.long(), id_


# =========================
# MODEL (same as training)
# =========================
class BetterHead(torch.nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU()
        )

        self.out = torch.nn.Conv2d(128, n_classes, 1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.out(x)


# =========================
# METRIC
# =========================
def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(n_classes):
        p = (pred == cls)
        t = (target == cls)

        inter = (p & t).sum().float()
        union = (p | t).sum().float()

        if union == 0:
            continue

        ious.append((inter / union).item())

    return np.mean(ious) if ious else 0.0


# =========================
# INFERENCE
# =========================
def forward_pass(model, backbone, imgs):
    with torch.no_grad():
        feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(feats)
        logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
    return logits


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = "../Offroad_Segmentation_Training_Dataset/val"
    model_path = "./best_model.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    dataset = MaskDataset(data_dir, transform, mask_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Load backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    # Get embedding dim
    sample_img, _, _ = dataset[0]
    with torch.no_grad():
        feat = backbone.forward_features(sample_img.unsqueeze(0).to(device))["x_norm_patchtokens"]
    emb_dim = feat.shape[2]

    # Load model
    model = BetterHead(emb_dim, n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    print("\nRunning evaluation...")

    ious = []

    for imgs, labels, _ in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.squeeze(1).long().to(device)

        outputs = forward_pass(model, backbone, imgs)

        iou = compute_iou(outputs, labels)
        ious.append(iou)

    print("\n======================")
    print(f"FINAL IoU: {np.mean(ious):.4f}")
    print("======================")


if __name__ == "__main__":
    main()