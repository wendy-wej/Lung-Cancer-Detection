---
license: mit
---
# Symptom-based Lung Cancer Prediction Model

This repository contains a machine learning model developed to predict lung cancer risk based on patient symptoms and other health-related factors. The model uses an ensemble approach combining multiple classifiers to provide a robust and accurate prediction.

## Model Overview

The Symptom-based Lung Cancer Prediction Model is designed to assist in the early detection of lung cancer by analyzing key symptoms and behaviors associated with the disease. The model is trained on a dataset containing various features such as smoking habits, anxiety, chronic diseases, and other symptoms.

### Key Features:

- **Predictive Model**: Uses an ensemble of classifiers (`VotingClassifier`) for improved accuracy.
- **Input Features**: Includes a range of symptoms and behaviors (e.g., smoking, fatigue, coughing).
- **Output**: Predicts the likelihood of lung cancer (high risk or low risk).

## Dataset

The dataset includes the following features:

- `SMOKING`: Whether the patient smokes (0 = No, 1 = Yes)
- `YELLOW_FINGERS`: Presence of yellow fingers (0 = No, 1 = Yes)
- `ANXIETY`: Presence of anxiety (0 = No, 1 = Yes)
- `PEER_PRESSURE`: Influence of peer pressure (0 = No, 1 = Yes)
- `CHRONIC DISEASE`: Existence of any chronic disease (0 = No, 1 = Yes)
- `FATIGUE`: Experience of fatigue (0 = No, 1 = Yes)
- `ALLERGY`: Presence of any allergies (0 = No, 1 = Yes)
- `WHEEZING`: Presence of wheezing (0 = No, 1 = Yes)
- `ALCOHOL CONSUMING`: Whether the patient consumes alcohol (0 = No, 1 = Yes)
- `COUGHING`: Presence of coughing (0 = No, 1 = Yes)
- `SWALLOWING DIFFICULTY`: Difficulty in swallowing (0 = No, 1 = Yes)
- `CHEST PAIN`: Presence of chest pain (0 = No, 1 = Yes)

## Installation

To use the model, ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
