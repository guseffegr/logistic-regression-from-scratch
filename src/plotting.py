import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve

def roc_curve_plot(y_val, y_proba):
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def precision_recall_curve_plot(y_val, y_proba):
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall")
    plt.legend()
    plt.show()