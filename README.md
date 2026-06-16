# Offroad Semantic Segmentation 🚜🌲

An end-to-end computer vision pipeline designed for autonomous navigation in unstructured, offroad environments. This project focuses on pixel-level classification of complex terrain features—such as dirt tracks, grass, rocks, and obstacles—to enable safe path planning for autonomous ground vehicles (AGVs).

---

## 🚀 Features

* **Multi-Class Terrain Classification:** Segments complex outdoor scenes into distinct operational zones (e.g., drivable path, low vegetation, hazardous obstacles, background).
* **Real-Time Inference Capability:** Optimized for deployment on edge devices with lightweight backbone support.
* **Robust Performance:** High Intersection-over-Union (IoU) metrics on unstructured paths with variable lighting and shadows.
* **Custom Evaluation Framework:** Includes comprehensive metrics monitoring including Mean IoU (mIoU), Pixel Accuracy, and Precision-Recall tracking.

---

## 🛠️ Tech Stack & Frameworks

* **Deep Learning Framework:** PyTorch / PyTorch Lightning
* **Computer Vision Libraries:** OpenCV, Albumentations
* **Model Architectures:** Unet++ / DeepLabV3+ / SegFormer
* **Experiment Tracking:** Weights & Biases (W&B) / TensorBoard

---

## 📊 Dataset & Preprocessing

Unstructured offroad environments present unique challenges like motion blur, dust, and dynamic illumination. The pipeline incorporates advanced augmentations to counter these:

### Augmentations

* Random Sun Flare
* Contrast Adjustments
* Elastic Transformations
* Coarse Dropout

### Target Classes

| Class ID | Description                            |
| -------- | -------------------------------------- |
| 0        | Background / Sky                       |
| 1        | Drivable Offroad Path (Dirt/Gravel)    |
| 2        | Low Vegetation / Grass                 |
| 3        | High Hazards (Rocks, Trees, Obstacles) |

---

## 📐 Pipeline Architecture

```text
[ Input Rugged Frame ]
          │
          ▼
[ Data Augmentation ]
          │
          ▼
[ Encoder (ResNet / EfficientNet) ]
          │
          ▼
     (Latent Features)
          │
          ▼
[ Decoder (U-Net / FPN) ]
          │
          ▼
[ Threshold / Argmax ]
          │
          ▼
[ Segmented Map Output ]
```

---

## ⚙️ Getting Started

### 1. Installation

```bash
git clone https://github.com/amulyarstar/offroad-segmentation.git
cd offroad-segmentation

pip install -r requirements.txt
```

### 2. Training the Model

```bash
python train.py --config config.yaml --batch_size 8 --epochs 50 --lr 0.001
```

### 3. Running Evaluation

```bash
python evaluate.py --weights checkpoints/best_model.pth --data_dir ./data/test
```

---

## 📈 Evaluation Metrics

The project utilizes standard semantic segmentation metrics to evaluate performance across highly imbalanced classes.

### Metrics Used

* **Mean Intersection over Union (mIoU)**
  Measures overlap between predicted and ground-truth segmentation masks.

* **Pixel Accuracy**
  Overall percentage of correctly classified pixels.

* **Dice Coefficient**
  Evaluates spatial overlap and improves segmentation boundary quality.

---

## 🤝 Contributing

Contributions to optimize network backbones, enhance inference speed, or integrate inference accelerators such as TensorRT are highly encouraged.

1. Fork the repository.
2. Create your feature branch:

```bash
git checkout -b feature/Optimization
```

3. Commit your changes:

```bash
git commit -m "Optimize encoder inference speed"
```

4. Push to GitHub:

```bash
git push origin feature/Optimization
```

5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License.

See the `LICENSE` file for more information.
