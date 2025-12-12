# -------------------------------------------------------------
# run_all_models.py
# Runs QNN + LR + SVM and compares accuracy + inference time
# Author: Farheen Shaikh
# -------------------------------------------------------------

import os
import matplotlib.pyplot as plt

from train_svm_lr import train_logistic_regression, train_svm
from train_qnn import train_qnn


def main():
    os.makedirs("result", exist_ok=True)

    # --- Classical ---
    lr_model, lr_acc, lr_cm, lr_t, _ = train_logistic_regression("transactions.csv")
    svm_model, svm_acc, svm_cm, svm_t, _ = train_svm("transactions.csv")

    # --- Quantum ---
    qnn_model, qnn_acc, qnn_cm, qnn_t = train_qnn("transactions.csv")

    # Print summary
    print("\n==================== SUMMARY ====================")
    print(f"LR  Accuracy: {lr_acc:.3f} | Avg inference (sec/pred): {lr_t:.6f}")
    print(f"SVM Accuracy: {svm_acc:.3f} | Avg inference (sec/pred): {svm_t:.6f}")
    print(f"QNN Accuracy: {qnn_acc:.3f} | Avg inference (sec/pred): {qnn_t:.6f}")

    # Speed plot
    models = ["Logistic Regression", "SVM", "QNN (4-Qubit)"]
    times = [lr_t, svm_t, qnn_t]

    plt.figure(figsize=(7, 4))
    plt.bar(models, times)
    plt.title("Inference Time Comparison Across Models")
    plt.ylabel("Inference Time (seconds per prediction)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig("result/inference_time_comparison.png", dpi=300)
    plt.close()

    print("\nSaved: result/inference_time_comparison.png")
    print("Saved: result/cm_lr.png, result/cm_svm.png, result/cm_qnn.png")


if __name__ == "__main__":
    main()