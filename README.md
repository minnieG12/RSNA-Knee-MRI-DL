# Report-Guided Weak Supervision for Multi-Label Knee MRI Abnormality Detection

Deep Learning mini-project for ICT-4442 at Manipal Institute of Technology.

## 1. Project Overview

This project investigates whether radiology reports can provide useful **training-time weak supervision** for multi-label knee MRI abnormality detection when structured image labels are scarce.

The project is based on the RSNA Knee Abnormality Detection dataset:

- **4,407 training studies**
- **58 studies with complete structured labels for all 12 abnormalities**
- **4,349 report-only studies** with radiology reports but without the complete structured label vector
- MRI structure: **study → series → slices**
- Task: **12-label multi-label classification**
- Final deployment goal: **MRI-only inference**

Radiology reports are treated as privileged information during training. The report-teacher module generates weak/soft labels, while the final MRI model is intended to operate without reports.

## 2. Research Question

Can radiology reports be converted into reliable weak supervision that improves image-only knee MRI abnormality detection under extreme structured-label scarcity?

The project studies four connected questions:

1. Can report text provide useful labels for the 12 knee abnormalities?
2. How do CNN and Vision Transformer baselines perform under the same study-level protocol?
3. Does DINOv2 with hierarchical slice/series attention improve MRI representation?
4. Does confidence-aware report supervision improve the final MRI-only model?

## 3. Target Abnormalities

The model predicts 12 independent sigmoid outputs:

- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Medial OA
- Lateral OA
- PF OA
- Effusion
- Synovitis
- Baker's
- Contusion
- Fracture

## 4. Methodology

### MRI preprocessing

DICOM MRI series are:

1. loaded and ordered using acquisition metadata;
2. intensity-clipped and normalized;
3. resized to a fixed spatial resolution;
4. stored as preprocessed NumPy volumes;
5. converted into **2.5D neighbouring-slice inputs**.

For each selected slice:

```text
previous slice → channel 1
current slice  → channel 2
next slice     → channel 3
```

This provides local volumetric context while remaining compatible with pretrained 2D vision backbones.

### Vision baselines

**EfficientNet-B0**

A pretrained CNN processes each 2.5D instance. Instance logits are mean-pooled to obtain a study-level prediction.

**ViT-S/16**

A pretrained Vision Transformer uses the same 2.5D input, study-level aggregation, folds, loss and evaluation protocol as EfficientNet-B0.

### Proposed representation pipeline

**DINOv2 ViT-S/14 → Slice Attention MIL → Series Attention MIL → Study prediction**

DINOv2 first produces a 384-dimensional feature vector for each 2.5D MRI instance. Later notebooks use these features to model the natural hierarchy:

```text
Study
 ├── Series 1
 │    ├── 2.5D instance
 │    ├── 2.5D instance
 │    └── ...
 ├── Series 2
 │    └── ...
 └── ...
```

Attention is then applied first across slices and subsequently across series.

### Weak supervision

Radiology reports are used only during training:

```text
Radiology report
      ↓
Report teacher
      ↓
12 weak/soft labels + confidence + evidence
      ↓
Confidence-aware MRI training
      ↓
MRI-only inference
```

The report teacher is not required during final inference.

## 5. Evaluation Protocol

Because the structured gold subset is small, evaluation uses **study-level 5-fold cross-validation**.

The same folds and evaluation protocol are intended to be reused across the vision models.

Primary/common metrics:

- Macro ROC-AUC
- Macro PR-AUC
- Macro Precision
- Macro Recall / Sensitivity
- Macro F1
- Macro Specificity

Supporting diagnostics:

- Micro F1
- Hamming Loss
- Subset Accuracy
- Per-abnormality ROC-AUC and PR-AUC
- ROC curves
- Precision–Recall curves
- Training/validation curves
- Study-level error analysis

The project emphasizes **OOF (out-of-fold) predictions** for held-out evaluation.

## 6. Notebook Pipeline

```text
00  Setup / Environment
      ↓
01  Dataset Audit
      ↓
02  Exploratory Data Analysis
      ↓
03  DICOM Preprocessing
      ↓
04  2.5D Slice Construction
      ↓
05  Study-Level Cross-Validation
      ↓
06  Report Teacher / Weak Labels
      ↓
07  EfficientNet-B0 Baseline
      ↓
08  ViT-S/16 Baseline
      ↓
09  DINOv2 ViT-S/14 Feature Extraction
      ↓
10  Slice-Level Attention MIL
      ↓
11  Series-Level Attention MIL
      ↓
12  Hierarchical DINOv2 + MIL
      ↓
13  Confidence-Weighted Weak Supervision
      ↓
14  Common Evaluation / Calibration
      ↓
15  Explainability
      ↓
16  Ablation Studies
      ↓
17  Error Analysis / Robustness
      ↓
18  Final MRI-Only Inference
```

## 7. Current Implementation Status

| Notebook | Status |
|---|---|
| 00 Setup | Completed |
| 01 Audit | Completed |
| 02 EDA | Completed |
| 03 Preprocessing | Updated for the 58-study gold pipeline |
| 04 2.5D | Updated for 58 gold studies |
| 05 CV Splits | Updated for 58 gold studies |
| 06 Report Teacher | Development / weak-label stage |
| 07 EfficientNet-B0 | Completed 5-fold gold baseline |
| 08 ViT-S/16 | Completed 5-fold gold baseline |
| 09 DINOv2 | Completed full 58-study embedding extraction |
| 10–18 | Remaining research pipeline |

### Current preliminary gold-only results

These are **preliminary, unoptimized baseline results**, not final model claims.

| Model | Macro ROC-AUC (5-fold mean ± SD) |
|---|---:|
| EfficientNet-B0 | 0.5723 ± 0.0690 |
| ViT-S/16 | 0.5591 ± 0.0504 |
| DINOv2 | Feature extraction complete; predictive evaluation occurs in the later MIL stages |

DINOv2 Notebook 09 currently provides the feature representation used by the subsequent MIL models rather than a standalone 12-label classifier.

## 8. Repository Structure

A suggested project structure is:

```text
project-root/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 00_setup/
│   ├── 01_audit/
│   ├── 02_eda/
│   ├── 03_preprocessing/
│   ├── 04_2p5d/
│   ├── 05_cv/
│   ├── 06_report_teacher/
│   ├── 07_efficientnet/
│   ├── 08_vit/
│   ├── 09_dinov2/
│   ├── 10_slice_mil/
│   ├── 11_series_mil/
│   ├── 12_hierarchical_mil/
│   ├── 13_weak_supervision/
│   ├── 14_evaluation/
│   ├── 15_explainability/
│   ├── 16_ablations/
│   ├── 17_error_analysis/
│   └── 18_inference/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── error_analysis/
│
└── docs/
    ├── literature/
    ├── report/
    └── presentation/
```

The actual directory names can follow the team's existing repository structure.

## 9. Reproducibility

The experiments are designed around:

- fixed random seed;
- fixed study-level folds;
- shared preprocessing artifacts;
- consistent 2.5D construction;
- common evaluation metrics;
- saved OOF predictions;
- saved fold-level histories and checkpoints where appropriate.

For 07 and 08, the same 58 gold studies and 5-fold study-level protocol are used.

## 10. Data and Privacy

The RSNA MRI data and radiology reports are **not included in this repository**.

Do not commit:

- raw DICOM files;
- processed MRI volumes;
- radiology reports;
- `.npy` embedding files;
- model checkpoints;
- API keys or tokens;
- Kaggle credentials;
- private or identifying patient information.

Notebooks 00–03 should also remain private on the notebook platform while they directly expose dataset acquisition/preprocessing workflows.

## 11. Running the Project

The notebooks are intended to be executed in a GPU-enabled environment such as Kaggle.

Recommended order:

```text
00 → 01 → 02 → 03 → 04 → 05
                         ↓
                    06 → 07 → 08 → 09
                                  ↓
                         10 → 11 → 12
                                  ↓
                         13 → 14 → 15
                                  ↓
                         16 → 17 → 18
```

Notebooks 07–09 use the shared outputs from 03–05, so those artifacts must be generated consistently before running the full experiments.

## 12. Academic Integrity

The project follows the ICT-4442 requirement that external code, tutorials and LLM assistance be disclosed in the report and that results must not be fabricated.

Any external code or AI-assisted material should be documented in the final report according to the course guidelines.

## 13. Core References

1. Bien N, Rajpurkar P, Ball RL, et al. *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.* PLOS Medicine, 2018.
2. Qiu Z, Xie Z, Lin H, et al. *Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities.* Nature Communications, 2024.
3. Xie Z, Qiu Z, Li Y, et al. *Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI.* eClinicalMedicine, 2025.
4. Müller-Franzes G, Khader F, Siepmann R, et al. *Medical slice transformer for improved diagnosis and explainability on 3D medical images with DINOv2.* Scientific Reports, 2025.
5. Oquab M, Darcet T, Moutakanni T, et al. *DINOv2: Learning Robust Visual Features without Supervision.* Transactions on Machine Learning Research, 2024.
6. Ilse M, Tomczak JM, Welling M. *Attention-based Deep Multiple Instance Learning.* ICML, 2018.
7. Misera L, Müller-Franzes G, Truhn D, Kather JN. *Weakly Supervised Deep Learning in Radiology.* Radiology, 2024.
8. Bosma JS, Saha A, Hosseinzadeh M, et al. *Semisupervised Learning with Report-guided Pseudo Labels for Deep Learning-based Prostate Cancer Detection Using Biparametric MRI.* Radiology: Artificial Intelligence, 2023.

## 14. Project Goal

The final system aims to demonstrate that radiology reports can act as **training-time supervision** for an image-only knee MRI model, while hierarchical attention allows the model to focus on informative slices and MRI series.

The final deployment pathway is:

```text
MRI study
   ↓
preprocessing
   ↓
2.5D representation
   ↓
DINOv2 / hierarchical MIL
   ↓
12 abnormality probabilities
```

Reports are used to supervise training, not required for final MRI-only inference.
