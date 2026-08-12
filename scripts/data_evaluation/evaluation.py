import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preproc
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preproc

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

# =========================
# Global plotting config
# =========================
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titleweight": "bold"
})

# =========================
# Config
# =========================
TEST_DIR = "D:/Project/final_val"          # Change if needed
IMG_SIZE = (224, 224)
BATCH = 32
TAU = 0.58
CLASSES = ["Non-Poisonous", "Poisonous"]

# =========================
# Load models
# =========================
resnet_model = load_model("resnet50_best.keras")
densenet_model = load_model("densenet121_best.keras")

# =========================
# Data generators
# =========================
def make_generator(preproc):
    return ImageDataGenerator(
        preprocessing_function=preproc
    ).flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False
    )

gen_resnet = make_generator(resnet_preproc)
gen_densenet = make_generator(densenet_preproc)

y_true = gen_resnet.classes

# =========================
# TTA prediction
# =========================
def predict_tta(model, generator):
    preds = []
    for i in range(len(generator)):
        x, _ = generator[i]

        p1 = model.predict(x, verbose=0)
        p2 = model.predict(x[:, :, ::-1, :], verbose=0)
        p3 = model.predict(np.rot90(x, 1, (1, 2)), verbose=0)

        preds.append((p1 + p2 + p3) / 3.0)

    return np.vstack(preds)

# =========================
# Predictions
# =========================
p_r = predict_tta(resnet_model, gen_resnet)
p_d = predict_tta(densenet_model, gen_densenet)

pred_r = p_r.argmax(axis=1)
pred_d = p_d.argmax(axis=1)

p_ens = (p_r + p_d) / 2.0
pred_ens = p_ens.argmax(axis=1)
pred_ens_tau = (p_ens[:, 1] >= TAU).astype(int)

# =========================
# Confusion matrices
# =========================
cm_resnet = confusion_matrix(y_true, pred_r)
cm_densenet = confusion_matrix(y_true, pred_d)
cm_ens = confusion_matrix(y_true, pred_ens)
cm_ens_tau = confusion_matrix(y_true, pred_ens_tau)

# =========================
# Plot Confusion Matrices
# =========================
def plot_cm(cm, title, fname):
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

plot_cm(cm_resnet,   "ResNet50",            "cm_resnet.png")
plot_cm(cm_densenet, "DenseNet121",         "cm_densenet.png")
plot_cm(cm_ens,      "Ensemble (Baseline)", "cm_ensemble.png")
plot_cm(cm_ens_tau,  "Ensemble + τ",        "cm_ensemble_tau.png")

# =========================
# Metrics from CM
# =========================
def metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return acc * 100, sens * 100, spec * 100, f1

# =========================
# Parameters (LOCKED)
# =========================
PARAMS = {
    "ResNet50": resnet_model.count_params() / 1e6,
    "DenseNet121": densenet_model.count_params() / 1e6,
    "Ensemble": (resnet_model.count_params() + densenet_model.count_params()) / 1e6,
    "Ensemble + τ": (resnet_model.count_params() + densenet_model.count_params()) / 1e6
}

# =========================
# Inference time (averaged)
# =========================
def measure_inference(model, sample, runs=20):
    times = []
    for _ in range(runs):
        start = time.time()
        model.predict(sample, verbose=0)
        times.append((time.time() - start) * 1000)
    return np.mean(times)

sample_batch, _ = gen_resnet[0]

t_resnet = measure_inference(resnet_model, sample_batch)
t_densenet = measure_inference(densenet_model, sample_batch)
t_ens = t_resnet + t_densenet
t_ens_tau = t_ens

# =========================
# Performance table
# =========================
configs = {
    "ResNet50": (cm_resnet, t_resnet),
    "DenseNet121": (cm_densenet, t_densenet),
    "Ensemble": (cm_ens, t_ens),
    "Ensemble + τ": (cm_ens_tau, t_ens_tau)
}

rows = []
for name, (cm, t_inf) in configs.items():
    acc, sen, spec, f1 = metrics_from_cm(cm)
    rows.append([name, acc, sen, spec, f1, PARAMS[name], t_inf])

df = pd.DataFrame(
    rows,
    columns=[
        "Model / Configuration",
        "Accuracy (%)",
        "Sensitivity (%)",
        "Specificity (%)",
        "F1-score",
        "Parameters (M)",
        "Inference Time (ms)"
    ]
)

# =========================
# Plot performance table
# =========================
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")

table = ax.table(
    cellText=np.round(df.iloc[:, 1:].values, 2),
    rowLabels=df["Model / Configuration"],
    colLabels=df.columns[1:],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.15, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    if row == 0:
        cell.set_facecolor("#4F81BD")
        cell.set_text_props(color="white", weight="bold")
    if row == 3:
        cell.set_facecolor("#E8F1FA")
        cell.set_text_props(weight="bold")

plt.title("Performance Metrics: Individual Models vs Ensemble", fontsize=16, pad=18)
plt.savefig("performance_metrics_table.png", bbox_inches="tight", dpi=300)
plt.close()

# =========================
# Accuracy vs Inference Trade-off (IMPROVED)
# =========================
models = df["Model / Configuration"].values
times = df["Inference Time (ms)"].values
accs = df["Accuracy (%)"].values
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.figure(figsize=(12, 7))
plt.scatter(times, accs, s=260, c=colors, edgecolor="black", zorder=3)

# Smart offsets to avoid overlap
# Left-side points (Ensemble) get negative dx, right-side points get positive dx
for i, (name, x, y) in enumerate(zip(models, times, accs)):
    # Determine horizontal offset: if x > average, move left; else right
    avg_time = np.mean(times)
    if x > avg_time:
        dx = -35
        ha = "right"
    else:
        dx = 25
        ha = "left"

    # Vertical offset: if y > avg, move down; else up
    avg_acc = np.mean(accs)
    if y > avg_acc:
        dy = -0.08
        va = "top"
    else:
        dy = 0.08
        va = "bottom"

    # Special tweaks for clarity
    if "ResNet" in name:
        dy = 0.08
        va = "bottom"
    if "DenseNet" in name:
        dy = -0.08
        va = "top"
    if "Ensemble + τ" in name:
        dy = 0.12
        va = "bottom"

    plt.annotate(
        f"{name}\n({y:.2f}%, {x:.0f} ms)",
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=12,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=colors[i])
    )

plt.xlabel("Inference Time (ms)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.title("Accuracy vs Inference Time Trade-off", fontsize=18, weight="bold")

# Adjust limits to give breathing room
plt.xlim(min(times) - 300, max(times) + 300)
plt.ylim(min(accs) - 0.3, max(accs) + 0.3)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_inference_tradeoff_final.png", dpi=300, bbox_inches="tight", pad_inches=0.6)
plt.close()

# =========================
# Precision–Recall Curve
# =========================
plt.figure(figsize=(6, 5))

def plot_pr(y, p, label):
    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})")

plot_pr(y_true, p_r[:, 1], "ResNet50")
plot_pr(y_true, p_d[:, 1], "DenseNet121")
plot_pr(y_true, p_ens[:, 1], "Ensemble")
plot_pr(y_true, (p_ens[:, 1] >= TAU).astype(int), "Ensemble + τ")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.close()

print("✅ All figures generated:")
print("  - cm_resnet.png")
print("  - cm_densenet.png")
print("  - cm_ensemble.png")
print("  - cm_ensemble_tau.png")
print("  - performance_metrics_table.png")
print("  - accuracy_inference_tradeoff_final.png")
print("  - precision_recall_curve.png")
