# -------------------------------------------------------------
# train_svm_lr.py
# Logistic Regression & SVM training for AFF Detection
# Uses same features as QNN for fair comparison
# Author: Farheen Shaikh
# -------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


FEATURE_COLS = ["ipi_minutes", "Amount1", "ResponseTime", "MessageCount"]
LABEL_COL = "label"


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing engineered features safely."""
    # ipi_minutes (if missing, compute from timestamps)
    if "ipi_minutes" not in df.columns:
        if "scam_msg_time" in df.columns and "first_pay_time" in df.columns:
            df["scam_msg_time"] = pd.to_datetime(df["scam_msg_time"], errors="coerce")
            df["first_pay_time"] = pd.to_datetime(df["first_pay_time"], errors="coerce")
            df["ipi_minutes"] = (df["first_pay_time"] - df["scam_msg_time"]).dt.total_seconds() / 60.0
        else:
            raise ValueError("ipi_minutes missing and timestamps not found to compute it.")

    # Amount1 = log1p(amount)
    if "Amount1" not in df.columns:
        if "amount" in df.columns:
            df["Amount1"] = np.log1p(pd.to_numeric(df["amount"], errors="coerce"))
        else:
            df["Amount1"] = 0.0

    # ResponseTime (placeholder if missing)
    if "ResponseTime" not in df.columns:
        ipi = pd.to_numeric(df["ipi_minutes"], errors="coerce")
        df["ResponseTime"] = (0.7 * ipi + np.random.normal(0, 1.0, size=len(df))).clip(lower=0)

    # MessageCount (placeholder if missing)
    if "MessageCount" not in df.columns:
        ipi = pd.to_numeric(df["ipi_minutes"], errors="coerce").fillna(ipi.median())
        ipi_max = ipi.max()
        if pd.isna(ipi_max) or ipi_max <= 0:
            df["MessageCount"] = 5
        else:
            msg = 10 - (ipi / ipi_max) * 10  # shorter IPI -> higher pressure -> more messages
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


def _prepare_xy(csv_path: str, test_size=0.3, seed=42):
    df = pd.read_csv(csv_path)
    df = _ensure_features(df)

    if LABEL_COL not in df.columns:
        raise ValueError(f"'{LABEL_COL}' column not found. Found: {df.columns.tolist()}")

    # Convert label to numeric and clean
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    df_model = df[FEATURE_COLS + [LABEL_COL]].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(df_model) == 0:
        raise ValueError(
            "After cleaning, 0 rows remain. Check NaNs in ipi_minutes/ResponseTime/etc."
        )

    X = df_model[FEATURE_COLS].values.astype(float)
    y = df_model[LABEL_COL].values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(csv_path="transactions.csv"):
    X_train, X_test, y_train, y_test = _prepare_xy(csv_path)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Inference time: average per prediction
    t0 = time.perf_counter()
    _ = model.predict(X_test)
    t1 = time.perf_counter()
    avg_infer = (t1 - t0) / len(X_test)

    return model, acc, cm, avg_infer, (X_test, y_test, y_pred)


def train_svm(csv_path="transactions.csv"):
    X_train, X_test, y_train, y_test = _prepare_xy(csv_path)

    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    t0 = time.perf_counter()
    _ = model.predict(X_test)
    t1 = time.perf_counter()
    avg_infer = (t1 - t0) / len(X_test)

    return model, acc, cm, avg_infer, (X_test, y_test, y_pred)


if __name__ == "__main__":
    lr_model, lr_acc, lr_cm, lr_t, lr_pack = train_logistic_regression("transactions.csv")
    svm_model, svm_acc, svm_cm, svm_t, svm_pack = train_svm("transactions.csv")

    print("\n=== Logistic Regression ===")
    print("Accuracy:", lr_acc)
    print("Avg inference time (sec/pred):", lr_t)
    print("Confusion Matrix:\n", lr_cm)
    print(classification_report(lr_pack[1], lr_pack[2], digits=3))

    print("\n=== SVM (RBF) ===")
    print("Accuracy:", svm_acc)
    print("Avg inference time (sec/pred):", svm_t)
    print("Confusion Matrix:\n", svm_cm)
    print(classification_report(svm_pack[1], svm_pack[2], digits=3))

       # Save plots into result/
    _plot_cm(lr_cm, "LR Confusion Matrix", "result/cm_lr.png")
    _plot_cm(svm_cm, "SVM Confusion Matrix", "result/cm_svm.png")
    print("\nSaved confusion matrices to result/ folder.")
    