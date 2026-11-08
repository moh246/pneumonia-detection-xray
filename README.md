# Pneumonia Detection from Chest X-rays

A pneumonia classifier for frontal chest X-rays on the NIH ChestX-ray14 dataset, from data exploration to DICOM inference, with an FDA-style write-up. Built for the Udacity AI for Healthcare Nanodegree.

## Result

On a validation set with about 20% pneumonia (1,435 images, split by patient):

| Metric | Value |
|---|---|
| AUC (ROC) | 0.702 |
| Average precision | 0.376 |
| F1 at threshold 0.403 | **0.440** |
| Recall (sensitivity) | 0.624 |
| Precision (PPV) | 0.340 |
| Specificity | 0.697 |

For context, the CheXNet paper (Rajpurkar et al., 2017) reports an average F1 of 0.387 for practising radiologists on this task. Pneumonia on X-ray is hard, and the labels themselves are noisy because they were mined from radiology reports by NLP.

## Notebooks

1. **`1_EDA.ipynb`**: data exploration. Demographics, view positions, how rare pneumonia is (about 1.3% of 112,120 images), which findings appear alongside it, and pixel-intensity distributions.
2. **`2_Build_and_train_model.ipynb`**: training.
   - Split **by patient** (80/20, stratified), so no patient appears in both sets.
   - Training set balanced 50/50. Validation set kept at about 20% pneumonia, closer to a clinical setting where pneumonia is suspected.
   - VGG16 pre-trained on ImageNet, with blocks 1–4 frozen, block 5 fine-tuned, and a new dense head.
   - Augmentation on training data only. Adam with learning-rate decay on plateau, and early stopping.
   - Threshold chosen to maximise F1 on validation, plus a **subgroup analysis** by sex, view, age and co-occurring findings.
3. **`3_Inference.ipynb`**: runs on DICOM files. It checks the header first (modality DX, body part CHEST, position PA or AP) and rejects anything else before predicting.

**[`FDA_submission.md`](FDA_submission.md)** covers intended use, limitations, algorithm design, training, ground truth and a validation plan.

## What the subgroup analysis shows

The model is much more sensitive on AP views (0.78) than on PA views (0.43). It also flags many studies with edema, infiltration or effusion as pneumonia, since these look alike on X-ray. Both are listed as limitations in the FDA document.

## Running it

The notebooks expect the NIH ChestX-ray14 data under `/data` (the Udacity workspace layout). Main libraries: Keras/TensorFlow, pydicom, scikit-learn, pandas.

## Author

Abdelrahman Ibrahim · [GitHub profile](https://github.com/moh246) · [LinkedIn](https://www.linkedin.com/in/abdelrahman-mohsen-4438991a8/)
