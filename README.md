# Anime vs Cartoon Image Classification

A comparative image classification project that distinguishes **anime** from **Western cartoon** images using three different machine learning approaches:

- **GIST Descriptor + SVM**
- **From-Scratch CNN**
- **EfficientNet-B0 Transfer Learning**

This project was completed for **COMP 9130: Applied Artificial Intelligence** as the final capstone project by **ImageVision**.

## Team Members
- Nicky Cheng
- Tanishq Rawat
- Vibhor Malik

## Project Overview

Anime and Western cartoons are both popular forms of illustrated media, but they differ in visual style. This project studies how well different machine learning approaches can classify an image into one of these two categories.

The main goal was to compare:
1. a **classical machine learning baseline**
2. a **CNN trained from scratch**
3. a **transfer learning model**

We also used **Grad-CAM** to understand what visual features the best model focuses on while making predictions.

## Problem Statement

Given an input image, classify it into one of the following classes:

- **Anime**
- **Cartoon**

The project also aims to identify which visual features help the model make this distinction.

## Dataset

We used the **Kaggle Anime and Cartoon Image Classification dataset**.

### Dataset Details
- **Total images:** 8,892
- **Anime images:** 4,447
- **Cartoon images:** 4,445
- **Class balance:** approximately balanced

The dataset includes:
- screenshots
- promotional art
- cover images

## Success Criteria

The project was considered successful if the final model achieved:

- **Accuracy greater than 85%** on the held-out test set

Additional evaluation metrics:
- Precision
- Recall
- Macro F1-score
- Confusion matrix

## Data Split

The dataset was split using **stratified sampling**:

- **Training:** 70% → 6,224 images
- **Validation:** 15% → 1,334 images
- **Test:** 15% → 1,334 images

All three models were evaluated on the **same held-out test set** for fair comparison.

## Methods

## 1. Baseline A: GIST Descriptor + SVM

This classical baseline uses the **GIST spatial envelope descriptor** to capture global scene layout.

### Key Details
- Images resized to **256 × 256**
- GIST computed over:
  - **4 × 4 spatial grid**
  - **4 scales**
  - **8 orientations**
- Final feature vector size: **512**
- Features standardized using **StandardScaler**
- Classifier: **SVM with RBF kernel**
- Hyperparameters tuned using **3-fold grid search**

## 2. Baseline B: From-Scratch CNN

A custom CNN was built and trained from scratch.

### Architecture
- 4 convolutional blocks
- Batch normalization
- ReLU activations
- Max pooling
- Adaptive average pooling
- Dropout-regularized classifier

### Training Details
- **Total parameters:** 389,890
- Optimizer: **AdamW**
- Learning rate: **1e-3**
- Weight decay: **1e-4**
- Label smoothing: **0.05**
- Scheduler: **Cosine annealing**
- Early stopping based on **validation macro F1**
- Maximum epochs: **15**

## 3. Proposed Model: EfficientNet-B0 Transfer Learning

The best-performing model used **EfficientNet-B0 pretrained on ImageNet**.

### Model Details
- Pretrained on **ImageNet**
- **Total parameters:** 4,010,110
- Final classifier replaced with:
  - Dropout (**p = 0.4**)
  - Linear layer for binary classification

### Two-Stage Training

#### Stage 1: Frozen Backbone
- Only classifier head trained
- Trainable parameters: **2,562**
- Epochs: **5**
- Learning rate: **1e-3**

#### Stage 2: Full Fine-Tuning
- Entire network unfrozen
- Backbone learning rate: **2e-5**
- Classifier learning rate: **2e-4**
- Up to **15 epochs**
- Early stopping patience: **4**

### Regularization and Augmentation
- RandomResizedCrop
- Horizontal flip
- Rotation (±10°)
- Color jitter
- Random erasing
- ImageNet normalization
- Mixed precision training (FP16)
- Class-weighted cross-entropy
- Label smoothing (**0.05**)

## Tools and Technologies

- **Python 3.10**
- **PyTorch 2.x**
- **scikit-learn**
- **PIL**
- **scikit-image**
- **Matplotlib**
- **Google Colab**
- **NVIDIA RTX PRO 6000 GPU**

## Results

## Overall Test Performance

| Model | Accuracy | Precision | Recall | Macro F1 |
|------|---------:|----------:|-------:|---------:|
| GIST + SVM | 0.6424 | 0.6427 | 0.6424 | 0.6423 |
| From-Scratch CNN | 0.8793 | 0.8821 | 0.8793 | 0.8791 |
| EfficientNet-B0 | **0.9490** | **0.9492** | **0.9490** | **0.9490** |

### Best Model
The **EfficientNet-B0 transfer learning model** achieved the best performance with:

- **94.90% accuracy**
- **0.9490 macro F1**

This exceeded the project success threshold of **85%** by nearly **10 percentage points**.

## Per-Class Results for EfficientNet-B0

| Class | Precision | Recall | F1-score | Support |
|------|----------:|-------:|---------:|--------:|
| Anime | 0.9573 | 0.9400 | 0.9486 | 667 |
| Cartoon | 0.9411 | 0.9580 | 0.9495 | 667 |

The model performed well on both classes and remained balanced across anime and cartoon predictions.

## Confusion Matrix Summary

### GIST + SVM
- 253 anime images misclassified as cartoon
- 224 cartoon images misclassified as anime

### From-Scratch CNN
- Better than GIST + SVM
- Still showed some imbalance in recall between classes

### EfficientNet-B0
- Only **68 total errors**
- 40 anime → cartoon
- 28 cartoon → anime

This shows that EfficientNet-B0 was much more accurate and balanced than the other two models.

## Training Behavior

The EfficientNet-B0 model improved significantly after moving from:

- **Stage 1: frozen backbone**
to
- **Stage 2: full fine-tuning**

Validation F1 improved from around **0.89** to **0.9475** after unfreezing, showing that fine-tuning pretrained features was very effective.

## Explainability with Grad-CAM

To understand what the best model was focusing on, we used **Grad-CAM** on the last convolutional block of EfficientNet-B0.

### Main Features the Model Focused On

#### 1. Eye Regions
- Anime eyes produced strong, concentrated activations
- Cartoon eyes usually produced weaker signals

#### 2. Line Work
- The model paid attention to line sharpness and boundary edges
- Anime often showed cleaner cel-shaded boundaries
- Cartoons often had bolder outlines

#### 3. Color Structure
- Cartoon images showed more diffuse activation across flatter color regions
- Anime images showed stronger activation near highlights and boundaries

This helped explain why the model could separate the two styles so effectively.

## Failure Analysis

Out of 1,334 test images, the EfficientNet-B0 model misclassified **68 images**, giving an error rate of **5.1%**.

### Main Failure Patterns

#### 1. Text Overlays
Promotional art and cover images with logos or text sometimes confused the model.

#### 2. Hybrid-Style Content
Some images visually sit between anime and cartoon styles, making them naturally harder to classify.

The project also noted that works like **Avatar: The Last Airbender** can fall near the boundary between the two styles, showing that this is not always a strict binary distinction.

## Boundary Cases

The least confident correct predictions were visually ambiguous. These boundary cases support the idea that anime and cartoon styles exist on a **continuum**, rather than as two completely separate groups.

## Confidence Calibration

The confidence analysis showed:

- Correct predictions mostly had **high confidence** (> 0.9)
- Wrong predictions were generally made with **lower confidence**

This suggests the model was reasonably well-calibrated.

## Why EfficientNet-B0 Performed Best

The performance ranking was:

**GIST + SVM < From-Scratch CNN < EfficientNet-B0**

### Reason
- **GIST + SVM** only captures global texture and spatial layout
- **CNN from scratch** can learn useful features, but the dataset is still limited for learning everything from zero
- **EfficientNet-B0** already has strong pretrained visual features from ImageNet, so it can focus on learning style-specific differences

This made transfer learning the most effective approach for this task.

## Limitations

Some limitations of the project include:

- The dataset includes promotional art and logo-heavy images
- Some images are hybrid in style
- Text overlays sometimes attract model attention
- The dataset mixes screenshots and cover art, which may affect consistency

A more curated dataset with only clean in-frame screenshots could improve results further.

## Future Work

Possible improvements for future versions of this project:

- Use **EfficientNet-B2** or **ConvNeXt-Tiny**
- Apply **test-time augmentation**
- Use **5-fold cross-validation**
- Build a cleaner dataset without promotional art
- Combine **classical features** (e.g. HOG, LBP) with deep learning features
- Test more hybrid-style shows and animations

## Team Contributions

| Member | Role | Responsibilities |
|------|------|------------------|
| Tanishq | Data Analyst / PM | GitHub, data preprocessing, GIST baseline, model selection |
| Nicky | Data Scientist | CNN baseline, augmentation pipeline, evaluation metrics |
| Vibhor | ML Engineer | EfficientNet architecture, Grad-CAM, failure analysis, visualizations |

## Conclusion

This project compared three approaches for anime vs cartoon image classification and found that **transfer learning with EfficientNet-B0** performed the best.

### Final Takeaways
- **GIST + SVM:** 64.24% accuracy
- **From-Scratch CNN:** 87.93% accuracy
- **EfficientNet-B0:** 94.90% accuracy

The project successfully met its goal and also used **Grad-CAM explainability** to show that the model mainly focuses on:

- facial features
- eyes
- line work
- color boundaries

Overall, the project demonstrates that transfer learning is the most effective approach for this binary style-classification problem.

## References

1. Chai, J., Ramesh, V., and Yeo, Y. *Are anime cartoons?* Stanford CS229 Machine Learning Project Report, 2016.  
2. Kaur, R., and Prakash, S. *When cartoon meets anime: Distinguishing animation styles with convolutional neural networks.* IEEE ICCCIS, 2020.  
3. Liu, C. et al. *Animated character style investigation with decision tree classification.* Symmetry, 2020.  
4. Lan, Z. et al. *Multi-label classification in anime illustrations based on hierarchical attribute relationships.* Sensors, 2023.  
5. Oliva, A., and Torralba, A. *Modeling the shape of the scene: A holistic representation of the spatial envelope.* International Journal of Computer Vision, 2001.  
6. Tan, M., and Le, Q. V. *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML, 2019.  
7. Selvaraju, R. R. et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV, 2017.  
8. Li, Y. et al. *A challenging benchmark of anime style recognition.* CVPR Workshop, 2022.  
9. Mittal, K. *Anime and Cartoon Image Classification.* Kaggle Dataset.
