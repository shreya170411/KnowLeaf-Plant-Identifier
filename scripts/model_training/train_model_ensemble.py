# one_shot_binary_resnet_densenet.py
import os, random, json, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preproc
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preproc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========= REPRODUCIBILITY =========
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ========= CONFIG =========
train_dir = "D:/project/final_train"
val_dir   = "D:/project/final_val"
img_size  = (224, 224)     # keep 224 on CPU
batch_size = 32
epochs_head = 20
epochs_fine = 8
patience_head = 5
patience_fine = 3
label_smoothing = 0.05
l2_reg = 1e-4
workers = 0                  # Windows + CPU safe
use_multiprocessing = False

# ========= HELPERS =========
def build_resnet(num_classes=2):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(base.input, out), base

def build_densenet(num_classes=2):
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(base.input, out), base

def freeze_batchnorm(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

def make_gens(preproc, shuffle_train=True):
    # Moderate aug for CPU
    train_aug = ImageDataGenerator(
        preprocessing_function=preproc,
        rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    val_aug = ImageDataGenerator(preprocessing_function=preproc)
    train_gen = train_aug.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=shuffle_train
    )
    val_gen = val_aug.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen

loss_fn = CategoricalCrossentropy(label_smoothing=label_smoothing)

def train_one(model, base, preproc, tag, patience):
    train_gen, val_gen = make_gens(preproc)
    # save class map once
    if tag == "resnet50":
        with open("class_indices.json","w") as f:
            json.dump(val_gen.class_indices, f, indent=2)

    callbacks = [
        ModelCheckpoint(f"{tag}_best.keras", monitor="val_accuracy", mode="max",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    # Head training (frozen base)
    model.compile(optimizer=Adam(learning_rate=3e-5), loss=loss_fn, metrics=["accuracy"])
    print(f"\n[{tag}] Training (frozen base)...")
    model.fit(
        train_gen, validation_data=val_gen, epochs=epochs_head, callbacks=callbacks,
        verbose=1
    )


    # Fine-tune: unfreeze base, keep BN frozen
    base.trainable = True
    freeze_batchnorm(model)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])
    print(f"[{tag}] Fine-tuning (unfrozen base, BN frozen)...")
    callbacks[0].filepath = f"{tag}_best.keras"  # ensure same filename
    callbacks[1].patience = patience_fine
    model.fit(
        train_gen, validation_data=val_gen, epochs=epochs_head, callbacks=callbacks,
        verbose=1
    )


    best = load_model(f"{tag}_best.keras")
    print(f"[{tag}] Best model reloaded from checkpoint.")
    return best, val_gen.class_indices

def get_probs_tta(model, generator):
    """TTA: normal + horizontal flip + 90° rotate."""
    probs_all = []
    for i in range(len(generator)):
        x, _ = generator[i]
        p1 = model.predict(x, verbose=0)                        # original
        p2 = model.predict(x[:, :, ::-1, :], verbose=0)         # horizontal flip
        x_rot = np.rot90(x, k=1, axes=(1,2))                    # 90° rotate
        p3 = model.predict(x_rot, verbose=0)
        probs_all.append((p1 + p2 + p3) / 3.0)
    return np.vstack(probs_all)

def eval_and_ensemble(resnet_model, densenet_model):
    # Build val gens with matching preprocessing
    _, resnet_val = make_gens(resnet_preproc, shuffle_train=False)
    _, densenet_val = make_gens(densenet_preproc, shuffle_train=False)
    y_true = resnet_val.classes  # same order for both

    print("\nCollecting TTA probabilities...")
    probs_resnet   = get_probs_tta(resnet_model, resnet_val)
    probs_densenet = get_probs_tta(densenet_model, densenet_val)

    # Individual accuracies
    acc_r = (probs_resnet.argmax(1) == y_true).mean()
    acc_d = (probs_densenet.argmax(1) == y_true).mean()
    print(f"ResNet50 (TTA) Accuracy:    {acc_r*100:.2f}%")
    print(f"DenseNet121 (TTA) Accuracy: {acc_d*100:.2f}%")

    # Weighted ensemble α-search
    best_acc, best_a = 0.0, 0.0
    best_probs = None
    for a in np.linspace(0, 1, 21):  # 0.00..1.00 step 0.05
        ens = a * probs_resnet + (1 - a) * probs_densenet
        acc = (ens.argmax(1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_a, best_probs = acc, a, ens
    print(f"\nBest Ensemble α={best_a:.2f} → {best_acc*100:.2f}% (argmax)")

    # Threshold sweep on ensemble (binary boost)
    # Assumes class_indices like {'Non_Poisonous':0, 'Poisonous':1}
    p_pos = best_probs[:, 1]
    thresholds = np.linspace(0.35, 0.65, 31)
    best_t, best_t_acc = 0.5, 0.0
    for t in thresholds:
        pred_t = (p_pos >= t).astype(int)
        acc_t = (pred_t == y_true).mean()
        if acc_t > best_t_acc:
            best_t_acc, best_t = acc_t, t

    # Final metrics with best threshold
    final_pred = (p_pos >= best_t).astype(int)
    print(f"Best threshold τ={best_t:.2f} → Final Acc: {best_t_acc*100:.2f}%")

    # Report
    idx_to_label = {v:k for k,v in resnet_val.class_indices.items()}
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    print("\nClassification Report (Ensemble @ best τ):")
    print(classification_report(y_true, final_pred, target_names=target_names))
    print("Confusion Matrix (Ensemble @ best τ):")
    print(confusion_matrix(y_true, final_pred))

# ========= TRAIN & EVAL =========
resnet_model, resnet_base = build_resnet(num_classes=2)
resnet_best, class_idx_r = train_one(resnet_model, resnet_base, resnet_preproc, tag="resnet50", patience=patience_head)

densenet_model, densenet_base = build_densenet(num_classes=2)
densenet_best, class_idx_d = train_one(densenet_model, densenet_base, densenet_preproc, tag="densenet121", patience=patience_head)

assert class_idx_r == class_idx_d, f"Class index mismatch: {class_idx_r} vs {class_idx_d}"

eval_and_ensemble(resnet_best, densenet_best)
print("\nDone.")
