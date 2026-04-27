import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
from tqdm import tqdm

# ================= CONFIG =================
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 25
IMG_SIZE = (224, 224)   # MUST be multiple of 14

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ================= LABEL MAP =================
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8,
    7100: 9, 10000: 10
}
n_classes = len(value_map)

CLASS_WEIGHTS = torch.tensor(
    [0.5,1.5,2,1.5,2,3,6,8,4,1,1],
    dtype=torch.float32
)

# ================= DATA =================
def convert_mask(mask):
    arr = np.array(mask)
    new = np.zeros_like(arr, dtype=np.uint8)
    for k,v in value_map.items():
        new[arr == k] = v
    return Image.fromarray(new)

class DatasetSeg(Dataset):
    def __init__(self, root, augment=False):
        self.img_dir = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")
        self.ids = os.listdir(self.img_dir)
        self.augment = augment

        self.norm = transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )

        self.color = transforms.ColorJitter(0.3,0.3,0.3,0.1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]

        img = Image.open(os.path.join(self.img_dir,name)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.mask_dir,name)))

        img = img.resize(IMG_SIZE)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            img = self.color(img)

        img = transforms.ToTensor()(img)
        img = self.norm(img)

        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

# ================= MODEL =================
class BetterHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(128, n_classes, 1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.reshape(B, H, W, C).permute(0,3,1,2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.out(x)

# ================= METRIC =================
def compute_iou(pred, target):
    pred = torch.argmax(pred,1)
    ious = []
    for c in range(n_classes):
        inter = ((pred==c)&(target==c)).sum().float()
        union = ((pred==c)|(target==c)).sum().float()
        if union>0:
            ious.append((inter/union).item())
    return np.mean(ious)

# ================= LOAD DATA =================
base = os.getcwd()

train_dir = os.path.join(base,"..","Offroad_Segmentation_Training_Dataset","train")
val_dir   = os.path.join(base,"..","Offroad_Segmentation_Training_Dataset","val")

train_ds = DatasetSeg(train_dir,augment=True)
val_ds   = DatasetSeg(val_dir)

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE)

# ================= BACKBONE =================
backbone = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
backbone.to(device)

# 🔥 PARTIAL UNFREEZE (IMPORTANT)
for name, param in backbone.named_parameters():
    param.requires_grad = False

for name, param in backbone.named_parameters():
    if "blocks.10" in name or "blocks.11" in name:
        param.requires_grad = True

# get embedding size
sample = next(iter(train_loader))[0].to(device)
feat = backbone.forward_features(sample)['x_norm_patchtokens']
embed_dim = feat.shape[-1]

model = BetterHead(embed_dim,n_classes).to(device)

# optimizer
optimizer = optim.AdamW(
    list(model.parameters()) + list(backbone.parameters()),
    lr=LR,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

# ================= TRAIN =================
best_iou = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs,masks in tqdm(train_loader):
        imgs,masks = imgs.to(device),masks.to(device)

        feats = backbone.forward_features(imgs)['x_norm_patchtokens']

        out = model(feats)
        out = F.interpolate(out,size=IMG_SIZE)

        loss = criterion(out,masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    ious = []

    with torch.no_grad():
        for imgs,masks in val_loader:
            imgs,masks = imgs.to(device),masks.to(device)

            feats = backbone.forward_features(imgs)['x_norm_patchtokens']
            out = model(feats)
            out = F.interpolate(out,size=IMG_SIZE)

            iou = compute_iou(out,masks)
            ious.append(iou)

    val_iou = np.mean(ious)

    print(f"\nEpoch {epoch+1}")
    print("Loss:", total_loss)
    print("Val IoU:", val_iou)

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(),"best_model.pth")
        print("🔥 Saved best model")

print("Training done")