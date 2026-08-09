# ==============================================================
# Dr. AI — Skin Disease Model Training
# Trains EfficientNetB0 on HAM10000 dataset
# Saves model as:  skin_disease_model4.h5
# Labels match Dr_AI_code.py skin_labels exactly
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# PATHS — update these to match your machine
# ==============================================================

DATA_DIR1     = "D:/Dr_AI/Dr_Yogi/HAM10000_images_part_1"
DATA_DIR2     = "D:/Dr_AI/Dr_Yogi/HAM10000_images_part_2"
METADATA_PATH = "D:/Dr_AI/Dr_Yogi/HAM10000_metadata_enriched.csv"
SAVE_PATH     = "skin_disease_model.h5"   # ← loaded in Dr_AI_code.py

# ==============================================================
# LABELS — must exactly match Dr_AI_code.py skin_labels order
# ==============================================================

skin_labels = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions"
]

# Internal mapping from HAM10000 dx codes → label strings above
disease_dict = {
    'akiec': 'Actinic keratoses',
    'bcc':   'Basal cell carcinoma',
    'bkl':   'Benign keratosis-like lesions',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic nevi',
    'vasc':  'Vascular lesions'
}

# ==============================================================
# STEP 1 — Load & prepare metadata
# ==============================================================

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH)
metadata["label"] = metadata["dx"].map(disease_dict)

def find_image_path(image_id):
    p1 = os.path.join(DATA_DIR1, image_id + ".jpg")
    p2 = os.path.join(DATA_DIR2, image_id + ".jpg")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

metadata["image_path"] = metadata["image_id"].apply(find_image_path)
metadata = metadata.dropna(subset=["image_path"])

print(f"Total images found : {len(metadata)}")
print(f"Classes            : {sorted(metadata['label'].unique())}")
print(metadata["label"].value_counts(), "\n")

# ==============================================================
# STEP 2 — Train / Validation split (stratified)
# ==============================================================

train_df, valid_df = train_test_split(
    metadata,
    stratify=metadata["label"],
    test_size=0.2,
    random_state=42
)

print(f"Train samples : {len(train_df)}")
print(f"Valid samples : {len(valid_df)}\n")

# ==============================================================
# STEP 3 — Image Generators
# ==============================================================

IMG_SIZE   = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_gen = val_datagen.flow_from_dataframe(
    valid_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Confirm label order matches skin_labels in Dr_AI_code.py
print("Generator class order:", train_gen.class_indices)

# ==============================================================
# STEP 4 — Build EfficientNetB0 Model
# ==============================================================

print("\nBuilding model...")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False   # Freeze base layers for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(len(skin_labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================================================
# STEP 5 — Phase 1: Train top layers (base frozen)
# ==============================================================

print("\n--- Phase 1: Training top layers ---\n")

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history1 = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
    callbacks=callbacks_phase1
)

# ==============================================================
# STEP 6 — Phase 2: Fine-tune last 30 layers of EfficientNetB0
# ==============================================================

print("\n--- Phase 2: Fine-tuning last 30 base layers ---\n")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00005),   # Lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history2 = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
    callbacks=callbacks_phase2
)

# ==============================================================
# STEP 7 — Save Final Model
# ==============================================================

model.save(SAVE_PATH)
print(f"\n✅ Model saved as: {SAVE_PATH}")
print(f"   Load in Dr_AI_code.py with:  skin_model = load_model('{SAVE_PATH}')")

# ==============================================================
# STEP 8 — Plot Training Curves
# ==============================================================

def plot_history(h1, h2):
    acc  = h1.history["accuracy"]       + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]   + h2.history["val_accuracy"]
    loss = h1.history["loss"]           + h2.history["loss"]
    vloss= h1.history["val_loss"]       + h2.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    phase2_start = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    axes[0].plot(epochs, acc,  color='#9b4da5', linewidth=2, label='Train Accuracy')
    axes[0].plot(epochs, val,  color='#4cd137', linewidth=2, label='Val Accuracy')
    axes[0].axvline(phase2_start, color='#e94560', linestyle='--', linewidth=1.5, label='Fine-tune start')
    axes[0].set_title('Accuracy')
    axes[0].legend(facecolor='#16213e', labelcolor='white')

    axes[1].plot(epochs, loss,  color='#9b4da5', linewidth=2, label='Train Loss')
    axes[1].plot(epochs, vloss, color='#4cd137', linewidth=2, label='Val Loss')
    axes[1].axvline(phase2_start, color='#e94560', linestyle='--', linewidth=1.5, label='Fine-tune start')
    axes[1].set_title('Loss')
    axes[1].legend(facecolor='#16213e', labelcolor='white')

    plt.suptitle('Dr. AI — Skin Model Training', color='#e94560', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("skin_model_training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Training curves saved as skin_model_training_curves.png")

plot_history(history1, history2)

# ==============================================================
# STEP 9 — Quick sanity check prediction
# ==============================================================

def quick_predict(img_path, model, labels):
    img       = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr       = image.img_to_array(img)
    arr       = preprocess_input(arr)
    arr       = np.expand_dims(arr, axis=0)
    preds     = model.predict(arr, verbose=0)[0]
    top_idx   = int(np.argmax(preds))
    print(f"\n🔬 Prediction: {labels[top_idx]}  ({round(float(preds[top_idx])*100, 2)}%)")

# Uncomment to test with a sample image:
# quick_predict("test_skin.jpg", model, skin_labels)

print("\n✅ Training complete. Model ready to use in Dr_AI_code.py")
print("""
────────────────────────────────────────────
To activate in Dr_AI_code.py, uncomment:

    skin_model  = load_model("skin_disease_model4.h5")
    skin_labels = [
        "Actinic keratoses",
        "Basal cell carcinoma",
        "Benign keratosis-like lesions",
        "Dermatofibroma",
        "Melanoma",
        "Melanocytic nevi",
        "Vascular lesions"
    ]
────────────────────────────────────────────
""")
