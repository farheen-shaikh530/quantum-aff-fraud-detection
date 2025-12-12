# -------------------------------------------------------------
# train_qnn.py
# 4-Qubit Quantum Neural Network (PennyLane) for AFF Detection
# Uses same features as LR/SVM for fair comparison
# Author: Farheen Shaikh
# -------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


FEATURE_COLS = ["ipi_minutes", "Amount1", "ResponseTime", "MessageCount"]
LABEL_COL = "label"


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    # ipi_minutes (if missing)
    if "ipi_minutes" not in df.columns:
        if "scam_msg_time" in df.columns and "first_pay_time" in df.columns:
            df["scam_msg_time"] = pd.to_datetime(df["scam_msg_time"], errors="coerce")
            df["first_pay_time"] = pd.to_datetime(df["first_pay_time"], errors="coerce")
            df["ipi_minutes"] = (df["first_pay_time"] - df["scam_msg_time"]).dt.total_seconds() / 60.0
        else:
            raise ValueError("ipi_minutes missing and timestamps not found to compute it.")

    # Amount1
    if "Amount1" not in df.columns:
        if "amount" in df.columns:
            df["Amount1"] = np.log1p(pd.to_numeric(df["amount"], errors="coerce"))
        else:
            df["Amount1"] = 0.0

    # ResponseTime placeholder
    if "ResponseTime" not in df.columns:
        ipi = pd.to_numeric(df["ipi_minutes"], errors="coerce")
        df["ResponseTime"] = (0.7 * ipi + np.random.normal(0, 1.0, size=len(df))).clip(lower=0)

    # MessageCount placeholder
    if "MessageCount" not in df.columns:
        ipi = pd.to_numeric(df["ipi_minutes"], errors="coerce").fillna(ipi.median())
        ipi_max = ipi.max()
        if pd.isna(ipi_max) or ipi_max <= 0:
            df["MessageCount"] = 5
        else:
            msg = 10 - (ipi / ipi_max) * 10
            msg = msg.replace([np.inf, -np.inf], np.nan).fillna(5)
            df["MessageCount"] = msg.clip(lower=1).round().astype(int)

    return df


def _plot_cm(cm, title, out_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def train_qnn(csv_path="transactions.csv", layers=2, epochs=60, stepsize=0.15, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    df = _ensure_features(df)

    if LABEL_COL not in df.columns:
        raise ValueError(f"'{LABEL_COL}' column not found. Found: {df.columns.tolist()}")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    df_model = df[FEATURE_COLS + [LABEL_COL]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(df_model) == 0:
        raise ValueError("After cleaning, 0 rows remain. Fix NaNs in ipi_minutes/ResponseTime/etc.")

    X = df_model[FEATURE_COLS].values.astype(float)
    y = df_model[LABEL_COL].values.astype(int)

    # Normalize to [0,1] for angle encoding
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed, stratify=y
    )

    # PennyLane arrays
    X_train = pnp.array(X_train)
    X_test  = pnp.array(X_test)
    y_train = pnp.array(y_train)
    y_test  = pnp.array(y_test)

    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    def feature_map(x):
        for i in range(n_qubits):
            qml.RX(np.pi * x[i], wires=i)

    def ansatz(weights):
        for l in range(weights.shape[0]):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            # ring entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

    @qml.qnode(dev)
    def circuit(x, weights):
        feature_map(x)
        ansatz(weights)
        return qml.expval(qml.PauliZ(0))

    def forward_prob(Xb, weights):
        raw = pnp.array([circuit(x, weights) for x in Xb])  # [-1,1]
        return (raw + 1) / 2  # [0,1]

    def loss_mse(weights, Xb, yb):
        p = forward_prob(Xb, weights)
        return pnp.mean((p - yb) ** 2)

    def predict_labels(Xb, weights, threshold=0.5):
        p = forward_prob(Xb, weights)
        return (p >= threshold).astype(int)

    weights = pnp.array(
        rng.uniform(0, np.pi, size=(layers, n_qubits)),
        requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=stepsize)

    for ep in range(1, epochs + 1):
        weights = opt.step(lambda w: loss_mse(w, X_train, y_train), weights)
        if ep % 10 == 0 or ep == 1:
            pred_test = predict_labels(X_test, weights)
            acc_test = accuracy_score(np.array(y_test), np.array(pred_test))
            print(f"Epoch {ep:02d} | Test Acc: {acc_test:.3f} | Loss: {loss_mse(weights, X_train, y_train):.4f}")

    # Final evaluation
    y_pred = predict_labels(X_test, weights)
    acc = accuracy_score(np.array(y_test), np.array(y_pred))
    cm = confusion_matrix(np.array(y_test), np.array(y_pred))

    # Inference time per prediction (QNN simulation)
    t0 = time.perf_counter()
    _ = predict_labels(X_test, weights)
    t1 = time.perf_counter()
    avg_infer = (t1 - t0) / len(X_test)

    # Save confusion matrix plot
    _plot_cm(cm, "QNN Confusion Matrix (4-Qubit)", "result/cm_qnn.png")

    print("\nFinal QNN Test Accuracy:", acc)
    print("Avg inference time (sec/pred):", avg_infer)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(np.array(y_test), np.array(y_pred), digits=3))

    # Return "model object" = weights + scaler (enough for reuse)
    qnn_model = {"weights": weights, "scaler": scaler, "features": FEATURE_COLS}
    return qnn_model, acc, cm, avg_infer


if __name__ == "__main__":
    train_qnn("transactions.csv")