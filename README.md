# Report-Guided Weak Supervision for Multi-Label Knee MRI Abnormality Detection

## About the Project

This project develops a deep-learning framework for detecting multiple abnormalities from knee MRI examinations. It combines 2.5D MRI representations, pretrained visual encoders, attention-based multiple instance learning (MIL), and report-guided weak supervision. The aim is to make better use of a dataset in which only a small subset of studies has complete structured labels, while retaining an MRI-only prediction pipeline at inference time.

## Problem Statement

Knee MRI interpretation requires specialist radiological expertise, and manually annotating every examination for multiple abnormalities is time-consuming. The available dataset contains 4,407 training studies: 58 studies have complete structured labels for all 12 target findings, while 4,349 additional studies have radiology reports but do not have the complete structured label vector.

The project addresses the challenge of training a reliable multi-label study-level classifier when complete expert-labelled MRI data is scarce. It must also account for variable numbers of MRI series and slices within each examination, class imbalance, uncertainty in report-derived labels, and the risk of information leakage between training and evaluation studies.

## Proposed Solution

The proposed framework uses radiology reports as privileged information during training. A report teacher identifies findings and their states—Present, Absent, Uncertain, or Not Mentioned—and associates confidence scores and supporting evidence with extracted labels. These report-derived labels are intended to supplement the limited gold-labelled subset through fold-safe, confidence-aware weak supervision. Reports are not required when the trained model performs inference; the final prediction pathway uses MRI data only.

The visual pipeline organizes DICOM data by study and MRI series, orders and normalizes slices, and creates a 2.5D representation from the previous, centre, and next slices. Multiple model families are implemented and compared: EfficientNet-B0, ViT-S/16, and DINOv2-based multiple instance learning. The DINOv2 pathway includes slice-level attention, series-level attention, and their hierarchical combination to aggregate variable-length MRI examinations into 12 study-level predictions.

## Target Abnormalities

The model predicts the following 12 findings, with multiple findings permitted for one study:

1. Anterior cruciate ligament (ACL) injury
2. Medial collateral ligament (MCL) injury
3. Medial meniscus tear
4. Lateral meniscus tear
5. Medial-compartment osteoarthritis
6. Lateral-compartment osteoarthritis
7. Patellofemoral-compartment osteoarthritis
8. Joint effusion
9. Synovitis
10. Baker's cyst
11. Bone contusion
12. Fracture

## Implemented Notebook Pipeline

The project workflow is organized into 18 implemented notebooks:

| Notebook | File / Stage | Purpose | Status |
|---|---|---|---|
| 00 | `00-setup.ipynb` | Environment setup and reproducibility configuration | Implemented |
| 01 | `01-dataset-audit.ipynb` | Dataset inventory, integrity checks, and metadata audit | Implemented |
| 02 | `02-eda.ipynb` | Exploratory data analysis and label/data summaries | Implemented |
| 03 | `03-preprocessing.ipynb` | DICOM indexing, cleaning, normalization, resizing, and processed-volume metadata | Implemented |
| 04 | `04-2-5d.ipynb` | Previous/centre/next slice construction | Implemented |
| 05 | `05-cv-splits.ipynb` | Study-level five-fold gold-label split manifest | Implemented |
| 06 | `06-llm-report-teacher.ipynb` | Radiology-report teacher, weak-label extraction, and validation workflow | Implemented |
| 07 | `07-efficientnet.ipynb` | EfficientNet-B0 2.5D gold-label baseline | Implemented |
| 08 | `08-vit.ipynb` | ViT-S/16 2.5D gold-label baseline | Implemented |
| 09 | `09-dinov2.ipynb` | DINOv2 ViT-S/14 feature extraction and embedding validation | Implemented |
| 10 | `10-slice-mil.ipynb` | DINOv2 slice-attention multiple instance learning | Implemented |
| 11 | `11-series-mil.ipynb` | DINOv2 series-attention multiple instance learning | Implemented |
| 12 | `12-hierarchical-mil.ipynb` | Joint learned slice- and series-attention hierarchical MIL | Implemented |
| 13 | `13-weak-supervision.ipynb` | Confidence-weighted report-guided weak-supervision training | Implemented |
| 14 | `14-evaluation-calibration.ipynb` | Common evaluation protocol, calibration, and model comparison | Implemented |
| 15 | `15-explainability.ipynb` | Model explainability and attention/feature interpretation | Implemented |
| 16 | `16-ablation-studies.ipynb` | Ablation experiments for model components and supervision choices | Implemented |
| 17 | `17-error-analysis-robustness.ipynb` | Error analysis, failure cases, and robustness evaluation | Implemented |
| 18 | `18-mri-only-inference.ipynb` | Final MRI-only prediction and inference pipeline | Implemented |

## Experimental Setup

The gold-labelled experiments use the 58 studies with complete labels and the study-level five-fold manifest generated in Notebook 05. The fold sizes are 12, 12, 12, 11, and 11 studies. The corrected evaluation workflow selects the epoch using an inner split of the outer training studies, refits on the complete outer training fold, and evaluates the untouched outer fold. The DINOv2 MIL stages reuse cached embeddings and metadata rather than extracting the frozen visual features again.

Evaluation includes macro ROC-AUC, macro PR-AUC, precision, recall, F1-score, specificity, micro F1, Hamming loss, subset accuracy, per-finding metrics, out-of-fold predictions, and study-level error analysis. Fold-wise mean ± standard deviation must be distinguished from metrics calculated over pooled out-of-fold predictions.

## Repository and Data

The repository contains notebooks, source code, configuration, documentation, figures, and permitted compact experiment summaries. Raw DICOM files, patient-level clinical reports, processed MRI volumes, large embedding arrays, checkpoints, Kaggle credentials, API keys, and other sensitive or bulky artifacts are kept outside the public Git repository.

