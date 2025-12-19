# Quantum vs Classical Models for Advance-Fee Fraud (AFF) Detection

Author: Farheen Shaikh

This project implements and compares Quantum Neural Networks (QNN) with classical machine-learning models (Logistic Regression, SVM) for detecting Advance-Fee Fraud (AFF) using Inter-Payment Interval (IPI) as the primary feature.

This repository contains:

✔ QNN Training Script (PennyLane)

✔ Logistic Regression & SVM Training Script

✔ Combined Runner Script

✔ Example dataset (IPI + behavioral features)

✔ Confusion matrix plots

✔ Research paper source (optional)

## Installation
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

## Experimental Findings
- QNN performance plateaus as qubits increase (4–6)
- Optimization noise dominates over circuit expressivity
- Dense angle encoding improves stability but not parity with classical MLP
- Results align with known limitations of NISQ-era variational circuits
