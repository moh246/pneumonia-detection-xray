# FDA Submission

**Name:** Abdelrahman Ibrahim

**Device:** PneumoDetect-CXR, a pneumonia detection assistant for chest X-rays

## 1. General Information

**Intended use:** The algorithm is intended to assist radiologists in detecting pneumonia on frontal (PA or AP) digital chest X-rays. It is a computer-aided triage and screening tool. It flags studies likely to show pneumonia so they can be prioritised in the reading worklist. It does not replace the radiologist's diagnosis.

**Indications for use:**

- Men and women aged 1 to 100 years.
- Frontal chest radiographs (DX modality) taken in PA or AP position.
- Patients with or without other thoracic findings, including those commonly comorbid with pneumonia (infiltration, edema, effusion, atelectasis).
- Use in screening and in clinical settings where pneumonia is suspected, as a worklist-prioritisation aid for a radiologist.

**Device limitations:**

- Trained and validated only on the NIH ChestX-ray14 dataset. Performance on other scanners, hospitals or populations has not been established.
- Not intended for lateral views, other body parts or other modalities. These are rejected by the DICOM check.
- Pneumonia often appears together with infiltration, edema, effusion and atelectasis, which look similar on X-ray. These findings cause false positives, and when they are present the model cannot reliably tell pneumonia from them (see the subgroup table in section 3).
- Sensitivity is much lower on PA views (0.43) than on AP views (0.78).
- Labels were derived from radiology reports by NLP and are noisy. The model inherits this label noise.
- Few very young (<10) or very old (>90) patients were in the data, so performance for these groups is less certain.
- The VGG16-based network is best run on a GPU. Mean prediction time in the test environment was about 1 second per image.

**Clinical impact of performance:** At the chosen operating point the algorithm has a recall (sensitivity) of 0.624, a precision (PPV) of 0.340 and a specificity of 0.697, on a validation set with about 20% prevalence. In a triage setting, a false negative means a pneumonia study is not moved up the worklist, but a radiologist still reads it, so the diagnosis is delayed rather than missed. A false positive moves a normal study up the list and costs reading time. The threshold was therefore chosen to balance both (maximum F1), leaning toward catching pneumonia. The device should not be used to rule out pneumonia on its own.

## 2. Algorithm Design and Function

```
DICOM file from PACS / scanner
        │
        ▼
DICOM check: Modality = DX, Body part = CHEST, Position = PA/AP, else reject
        │
        ▼
Pre-processing: scale 0–1, standardise (training mean/std), resize 224×224, 3 channels
        │
        ▼
CNN inference: VGG16 (fine-tuned) + dense head → probability
        │
        ▼
Threshold: probability > 0.403 → "Pneumonia", else "No Pneumonia"
        │
        ▼
Radiologist: result used to prioritise the worklist; the radiologist makes the diagnosis
```

**DICOM checking steps:** The DICOM header is read with pydicom. The study is accepted only if Modality = DX, BodyPartExamined = CHEST and PatientPosition is PA or AP. Anything else is rejected with a message, and no prediction is made. A warning is printed if PatientAge is outside 1–100.

**Preprocessing steps:** The pixel array is scaled to the range 0–1 and standardised with the mean (0.4813) and standard deviation (0.2300) of the training images. It is then resized to 224 × 224 pixels, and the single gray channel is repeated three times to match the VGG16 input (1 × 224 × 224 × 3).

**CNN architecture:** VGG16 pre-trained on ImageNet, truncated at `block5_pool`, followed by a new classification head.

| Layer | Details | Trainable |
|---|---|---|
| VGG16 blocks 1–4 | ImageNet weights, 10 conv layers + 4 max-pool | No (frozen) |
| VGG16 block 5 | 3 conv layers + max-pool (7×7×512 output) | Yes (fine-tuned) |
| Flatten | 25,088 features | – |
| Dropout | rate 0.5 | – |
| Dense | 1024 units, ReLU | Yes |
| Dropout | rate 0.5 | – |
| Dense | 512 units, ReLU | Yes |
| Dropout | rate 0.3 | – |
| Dense | 256 units, ReLU | Yes |
| Dense (output) | 1 unit, sigmoid → probability of pneumonia | Yes |

## 3. Algorithm Training

**Parameters:**

- Augmentation, on training data only: horizontal flip, rotation (±10°), width and height shift (10%), zoom (10%), shear (0.05). No vertical flip, because upside-down chest images don't occur clinically. Validation data received no augmentation.
- Batch size: 32 for training. The validation set was evaluated as one batch.
- Optimizer: Adam, initial learning rate 1e-4, halved when validation loss plateaued for 3 epochs (minimum 1e-6). Loss: binary cross-entropy.
- Frozen layers: VGG16 blocks 1–4 (all layers up to `block4_pool`).
- Fine-tuned layers: `block5_conv1`, `block5_conv2`, `block5_conv3` (and `block5_pool`).
- Added layers: Flatten, Dropout(0.5), Dense(1024, ReLU), Dropout(0.5), Dense(512, ReLU), Dropout(0.3), Dense(256, ReLU), Dense(1, sigmoid).
- Best weights saved on minimum validation loss (0.526). Early stopping with patience 10. Training stopped after 18 epochs.

The training history, ROC curve, precision-recall curve and F1-versus-threshold plots are in `2_Build_and_train_model.ipynb`.

**Final threshold and explanation:** The final threshold is **0.403**. It maximises the F1 score on the validation set, balancing sensitivity against precision for a triage tool.

| Metric | Value |
|---|---|
| AUC (ROC) | 0.702 |
| Average precision | 0.376 |
| F1 score | 0.440 |
| Precision (PPV) | 0.340 |
| Recall (sensitivity) | 0.624 |
| Specificity | 0.697 |
| NPV | 0.881 |
| TP / FP / TN / FN | 179 / 348 / 800 / 108 |

For reference, the CheXNet study (Rajpurkar et al., 2017) reported an average F1 of 0.387 for practising radiologists on pneumonia detection from chest X-rays, which shows how hard the task is.

**Performance by subgroup** (validation set, at the final threshold):

| Subgroup | n | Pneumonia | Sensitivity | Specificity |
|---|---|---|---|---|
| Male | 860 | 189 | 0.656 | 0.677 |
| Female | 575 | 98 | 0.561 | 0.725 |
| PA view | 846 | 129 | 0.434 | 0.823 |
| AP view | 589 | 158 | 0.778 | 0.487 |
| Age 1–19 | 83 | 19 | 0.684 | 0.750 |
| Age 20–39 | 402 | 98 | 0.643 | 0.711 |
| Age 40–59 | 618 | 113 | 0.602 | 0.675 |
| Age 60–79 | 325 | 56 | 0.607 | 0.714 |
| Age 80–100 | 7 | 1 | – | – |
| With infiltration | 328 | 130 | 0.769 | 0.505 |
| With edema | 84 | 62 | 0.952 | 0.227 |
| With effusion | 172 | 48 | 0.729 | 0.452 |
| With atelectasis | 175 | 59 | 0.475 | 0.569 |
| No other finding | 712 | 64 | 0.438 | 0.802 |

The age 80–100 group is too small (7 images, 1 with pneumonia) to report meaningful rates.

## 4. Databases

The NIH ChestX-ray14 dataset has 112,120 frontal chest X-rays from 30,805 patients, each with up to 14 disease labels mined from the radiology reports. About 1.3% of images are labelled Pneumonia, usually together with other findings. Records with an invalid age (>100) were removed. The data was split by patient (80% / 20%, stratified on pneumonia), so no patient appears in both sets.

**Training dataset:** 2,286 images, of which 1,143 are pneumonia (50% prevalence). The set contains all pneumonia images from the training patients plus an equal random sample of non-pneumonia images, which may include other diseases. It covers both sexes, ages 1–100, and both PA and AP views. Balancing stops the network from learning to always predict the majority class.

**Validation dataset:** 1,435 images, of which 287 are pneumonia (20% prevalence). The set contains all pneumonia images from the held-out patients plus randomly sampled non-pneumonia images. The 20% prevalence approximates a clinical population where pneumonia is suspected, so precision and F1 are realistic. No augmentation was applied.

## 5. Ground Truth

Labels come from the NIH dataset. They were extracted from the associated radiology reports with natural-language processing, with a reported accuracy above 90%.

**Benefits:** this made it possible to label over 100,000 images cheaply, which a deep network needs. It reflects real clinical reporting, and each image carries all its co-occurring findings.

**Limitations:** the labels are not confirmed by an independent radiologist review, laboratory tests or clinical follow-up. Reports can be uncertain or incomplete, and pneumonia is a clinical diagnosis that can't always be made from an X-ray alone. Some labels are therefore wrong, which caps achievable performance and adds noise to the evaluation.

## 6. FDA Validation Plan

**Patient population:** Men and women aged 1–100 presenting for chest X-ray at one or more hospitals other than the training source. Studies are frontal (PA or AP) digital chest radiographs (DX), with a prevalence of pneumonia similar to the intended setting (about 20% where pneumonia is suspected). Patients with other thoracic findings, including infiltration, edema, effusion and atelectasis, should be included in proportions typical of routine practice. The data should cover different scanner manufacturers and include enough cases per age, sex and view subgroup to report subgroup performance.

**Ground truth acquisition:** The ideal gold standard combines imaging with clinical confirmation (for example sputum culture, laboratory results and clinical course). Because that is costly and often unavailable, the practical approach is a silver standard: each image is read independently by three board-certified radiologists, and the ground truth is the majority (or experience-weighted) vote. Readers see the images without the algorithm output.

**Algorithm performance standard:** The primary metric is the F1 score, reported with sensitivity, specificity, precision and AUC and their 95% confidence intervals. The algorithm should reach an F1 score at least as high as the average radiologist (0.387 in Rajpurkar et al., 2017) and show statistically non-inferior sensitivity. Performance should also be reported for each subgroup (age band, sex, PA/AP view and comorbid findings).
