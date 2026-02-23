# Data Harmonization: PTB-XL to AAMI

## 1. The Challenge
MIT-BIH Arrhythmia Database (Source) uses AAMI Beat Classes:
*   **N**: Normal
*   **S**: Supraventricular Ectopic
*   **V**: Ventricular Ectopic
*   **F**: Fusion
*   **Q**: Unknown

PTB-XL (Target) uses SCP-ECG Statements (Recording Level).
Standard PTB-XL Superclasses (Wagner et al.):
*   **NORM**: Normal ECG
*   **MI**: Myocardial Infarction
*   **STTC**: ST/T Change
*   **CD**: Conduction Disturbance
*   **HYP**: Hypertrophy

## 2. The Harmonization Strategy (Corrected)
To strictly test generalization without label noise, we adopt a **Binary Classification** task.
**Goal**: Test "Abnormality Detection" robustness.
Mapping:
*   **N (MIT-BIH)** -> **NORM (PTB-XL)** (Class 0: Normal)
*   **S/V/F/Q (MIT-BIH)** -> **MI/STTC/CD/HYP (PTB-XL)** (Class 1: Abnormal)

This strategy avoids the invalid mapping of "Myocardial Infarction" to "Ventricular Ectopic beats". Instead, we ask: *Can a model trained to detect irregular beats (MIT-BIH) generalize to detecting pathology in a broad sense (PTB-XL)?*
*   **Hypothesis**: Yes. WavKAN's morphological awareness should make it better at distinguishing "Standard Sinus Rhythm" from "Anything Else" across domains.

## 3. Implementation
The `src/data_harmonization.py` script implements the standard aggregation of PTB-XL SCP codes into the 5 superclasses.
