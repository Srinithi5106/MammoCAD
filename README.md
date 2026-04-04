
# MammoCAD — AI-Powered Mammogram Analysis System

MammoCAD is a clinical decision-support web application that uses deep learning to assist radiologists and lab technicians in analysing mammogram images for early detection of breast cancer.

---

## Overview

MammoCAD combines a trained Convolutional Neural Network (EfficientNetB3) with a role-based clinical workflow to deliver real-time mammogram analysis, BI-RADS classification, and detailed radiomics-style feature profiling — all within a secure, dark-themed Streamlit interface.

---

## Features

- **Role-based access** — Separate dashboards for Doctors and Lab Assistants
- **AI Prediction** — Binary classification (Benign / Malignant) with probability scores
- **BI-RADS Classification** — Automatic assignment of BI-RADS categories (0–6)
- **Feature Extraction** — OpenCV-based radiomics feature extraction from mammogram images
- **Interactive Visualisations** — Probability gauge, radar chart, feature bar charts, BI-RADS distribution
- **Patient Management** — Full patient registration, history tracking, and multi-analysis records
- **PDF Report Generation** — Downloadable clinical reports with embedded charts
- **Doctor Dashboard** — Population statistics, trend analysis, scatter matrix, and case overview
- **User Registration** — In-app account creation for both roles

---

## Model Architecture

- **Base Model:** EfficientNetB3 (ImageNet pre-trained)
- **Training Dataset:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
  - Train: 1,233 benign + 966 malignant images
  - Test: 307 benign + 230 malignant images
- **Training Strategy:** Two-phase transfer learning (frozen base → fine-tuning top 30 layers)
- **Inference Format:** ONNX Runtime (lightweight, no TensorFlow dependency at runtime)
- **Input Size:** 224 × 224 RGB
- **Output:** Sigmoid probability of malignancy

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit 1.28 |
| AI Inference | ONNX Runtime |
| Model Training | TensorFlow / Keras (EfficientNetB3) |
| Image Processing | OpenCV, Pillow |
| Database | SQLite (via Python sqlite3) |
| Authentication | bcrypt password hashing |
| Visualisations | Plotly |
| PDF Reports | fpdf2 |
| Deployment | Hugging Face Spaces (Docker) |

---

## Default Credentials

| Role | Username | Password |
|---|---|---|
| Doctor | `doctor1` | `doc123` |
| Lab Assistant | `labtech1` | `lab123` |

You can also register a new account from the login page.

---

## Usage

### Lab Assistant Workflow
1. Login as Lab Assistant
2. Enter patient details (ID, name, age, contact, clinical history)
3. Upload a mammogram image (PNG/JPG)
4. Click **Run Analysis**
5. View prediction, BI-RADS category, probability scores, and feature charts

### Doctor Workflow
1. Login as Doctor
2. View dashboard overview — total cases, malignant/benign distribution, trends
3. Browse all patient records and their analyses
4. View detailed analysis with tabbed visualisations
5. Generate and download PDF reports

---

## BI-RADS Classification

| Category | Description | Malignancy Risk |
|---|---|---|
| BI-RADS 0 | Incomplete — additional imaging needed | — |
| BI-RADS 2 | Benign finding | 0% |
| BI-RADS 3 | Probably benign | < 2% |
| BI-RADS 4 | Suspicious | 15–30% |
| BI-RADS 5 | Highly suggestive of malignancy | > 95% |
| BI-RADS 6 | Known biopsy-proven malignancy | 100% |

---

## Important Disclaimer

> This application is an AI-assisted Computer-Aided Diagnosis (CAD) tool intended solely as a **decision-support system**. It does **not** replace clinical judgment, radiological interpretation, or histopathological confirmation. All findings must be reviewed and validated by a qualified medical professional. Do not make clinical decisions based solely on this system.

---

## Project Structure

```
MammoCAD/
├── app.py                  # Main Streamlit application
├── config.py               # Paths and constants
├── database.py             # SQLite database operations
├── predict.py              # ONNX inference + feature extraction
├── visualizations.py       # Plotly chart functions
├── report_generator.py     # PDF report generation
├── model_downloader.py     # Auto-download model from Google Drive
├── train_ai.py             # Model training script (local use)
├── prepare_dataset.py      # CBIS-DDSM dataset organizer
├── requirements.txt        # Python dependencies
├── Dockerfile              # HF Spaces deployment
└── assets/
    └── style.css           # Custom dark theme CSS
```

---
## 1.Login Page - Lab Assistant

<img width="400" height="380" alt="03login_lab_assistant" src="https://github.com/user-attachments/assets/a6a26e52-d3bd-4a8f-b455-0be8daa43ccf" />

## 2.Lab Assistant Dashboard
<img width="743" height="411" alt="04_labassistantdashboard_1" src="https://github.com/user-attachments/assets/7492a044-d575-439e-8968-49a0d7d57a33" />

<img width="904" height="419" alt="04_labassistantdashboard_2" src="https://github.com/user-attachments/assets/0b1958f2-5218-4a84-bc36-e94c01c9b6c1" />

<img width="478" height="343" alt="04_labassistantdashboard_3" src="https://github.com/user-attachments/assets/fb02f8a1-0bab-42da-bbe1-31d704c7e20e" />

<img width="925" height="398" alt="04_labassistantdashboard_4" src="https://github.com/user-attachments/assets/a7e873ec-928f-4c4b-b580-f73721e46545" />

## 1.Login Page - Doctor

<img width="373" height="379" alt="01login" src="https://github.com/user-attachments/assets/4b54951c-4239-46c6-9bbf-c9b375078fb1" />

## 2.Doctor Dashboard

<img width="914" height="436" alt="02doctordashboard_1" src="https://github.com/user-attachments/assets/a7c4a24c-aff2-48a7-ae74-edf36fe7a314" />

<img width="915" height="407" alt="02doctordashboard_2" src="https://github.com/user-attachments/assets/3ebdff3d-ccf9-44a5-acc0-2510dd329e53" />

<img width="927" height="416" alt="02doctordashboard_3" src="https://github.com/user-attachments/assets/44ab14b2-9c35-4aa4-8a52-238c4d77a55f" />

<img width="914" height="419" alt="02doctordashboard_4" src="https://github.com/user-attachments/assets/45bc8bed-fc9f-4b46-82d0-9a40493d3eb1" />

<img width="934" height="400" alt="02doctordashboard_5" src="https://github.com/user-attachments/assets/1967b40c-bf28-44b7-83ed-3a8ad6878088" />

<img width="920" height="404" alt="02doctordashboard_6" src="https://github.com/user-attachments/assets/de0b667e-07ec-45c7-9ad2-c66f54e847cb" />


## Developed By

Srinithi B

Dataset: [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) — Curated Breast Imaging Subset of DDSM
