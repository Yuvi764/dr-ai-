"""
Dr. AI — Disease Symptom Deep Learning Model  (FIXED v2)
=========================================================
ROOT CAUSE of headache→Paralysis bug:
  The dataset has only 4 symptoms for Paralysis. Old augmentation
  randomly dropped symptoms, creating samples with ONLY 'headache'.
  The DNN learned: headache_alone → Paralysis (highest confidence).

THREE-PART FIX applied here:
  1. ANCHOR unique symptoms  — symptoms unique to a disease are NEVER
     dropped during augmentation (altered_sensorium, weakness_of_one_body_side
     are always present in every Paralysis sample).
  2. MINIMUM FLOOR          — every augmented sample keeps ≥50% of
     core symptoms (minimum 2), so no sample is ambiguously sparse.
  3. ZERO NOISE addition    — no random foreign symptoms added.
     They blur class boundaries and hurt accuracy on sparse queries.

Architecture: Residual MLP
  Input(137 severity-weighted) → Dense(512) → Dense(256)+Residual
  → Dense(128) → Dense(64) → Softmax(41)
"""

import os, json, random, csv
import numpy as np

# ─────────────────────────────────────────────
# PATHS — update to match your machine
# ─────────────────────────────────────────────
DATASET_CSV      = "D:/Dr_AI/Dr_Yogi/Disease_symtom/dataset.csv"
DESCRIPTION_CSV  = "D:/Dr_AI/Dr_Yogi/Disease_symtom/symptom_Description.csv"
PRESCRIPTION_CSV = "D:/Dr_AI/Dr_Yogi/Disease_symtom/Detailed_Disease_Prescriptions.csv"
PRECAUTION_CSV   = "D:/Dr_AI/Dr_Yogi/Disease_symtom/symptom_precaution.csv"
SEVERITY_CSV     = "D:/Dr_AI/Dr_Yogi/Disease_symtom/Symptom-severity.csv"
MODEL_OUT        = "disease_model.h5"
METADATA_OUT     = "model_metadata.json"

# ─────────────────────────────────────────────
# STEP 1 — Load all CSV data
# ─────────────────────────────────────────────

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

print("━━━ Step 1: Loading datasets ━━━")

raw_rows = []
with open(DATASET_CSV, encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        disease = row[0].strip()
        syms = [s.strip().replace(" ", "_") for s in row[1:] if s.strip() and s.strip() != "0"]
        raw_rows.append({"disease": disease, "symptoms": [s for s in syms if s]})

descriptions = {r["Disease"].strip(): r["Description"].strip() for r in load_csv(DESCRIPTION_CSV)}

prescriptions = {}
for r in load_csv(PRESCRIPTION_CSV):
    prescriptions[r["Disease"].strip()] = {
        "first_line":  r.get("First Line",  "").strip(),
        "second_line": r.get("Second Line", "").strip(),
        "last_line":   r.get("Last Line",   "").strip()
    }

precautions = {}
for r in load_csv(PRECAUTION_CSV):
    d = r["Disease"].strip()
    precautions[d] = [v.strip() for k, v in r.items() if k.startswith("Precaution") and v.strip()]

severity = {}
for r in load_csv(SEVERITY_CSV):
    severity[r["Symptom"].strip()] = int(r["weight"])

# Merge duplicate disease rows
disease_map = {}
for r in raw_rows:
    d = r["disease"]
    if d not in disease_map:
        disease_map[d] = list(r["symptoms"])
    else:
        disease_map[d] = list(set(disease_map[d] + r["symptoms"]))

ALL_SYMPTOMS = sorted(set(s for syms in disease_map.values() for s in syms))
DISEASES     = sorted(disease_map.keys())
N_SYMPTOMS   = len(ALL_SYMPTOMS)
N_DISEASES   = len(DISEASES)
sym2idx      = {s: i for i, s in enumerate(ALL_SYMPTOMS)}
dis2idx      = {d: i for i, d in enumerate(DISEASES)}

print(f"  Diseases  : {N_DISEASES}")
print(f"  Symptoms  : {N_SYMPTOMS}")

# ─────────────────────────────────────────────
# STEP 2 — Pre-compute UNIQUE (discriminative) symptoms
# A symptom is unique to a disease if NO other disease has it.
# These are the ANCHORS we never drop during augmentation.
# ─────────────────────────────────────────────

print("\n━━━ Step 2: Computing discriminative symptom anchors ━━━")

all_sets = {d: set(syms) for d, syms in disease_map.items()}

def get_unique_symptoms(disease):
    others = set()
    for d2, s2 in all_sets.items():
        if d2 != disease:
            others.update(s2)
    return all_sets[disease] - others

unique_per_disease = {d: get_unique_symptoms(d) for d in DISEASES}

for d, u in sorted(unique_per_disease.items(), key=lambda x: -len(x[1])):
    if u:
        print(f"  {d}: {sorted(u)}")

no_unique = [d for d, u in unique_per_disease.items() if not u]
print(f"\n  ⚠ Diseases with NO unique symptoms ({len(no_unique)}) — rely on symptom combinations:")
for d in no_unique:
    print(f"    {d}: {sorted(disease_map[d])}")

# ─────────────────────────────────────────────
# STEP 3 — FIXED Data Augmentation
# ─────────────────────────────────────────────

AUGMENT_PER_CLASS = 250   # more samples = more robust DNN

def augment_sample(disease, core_symptoms):
    """
    FIXED augmentation — three guarantees:
      G1. Unique/anchor symptoms are ALWAYS kept (never dropped)
      G2. At least max(2, 50% of core) symptoms kept per sample
      G3. No random noise symptoms from other diseases
    """
    unique = unique_per_disease.get(disease, set())
    vec    = np.zeros(N_SYMPTOMS, dtype=np.float32)
    kept   = []

    for sym in core_symptoms:
        if sym not in sym2idx:
            continue
        sev = severity.get(sym, 3)

        if sym in unique:
            keep = True                              # G1: anchors always kept
        else:
            drop_prob = 0.25 * max(0.0, 1.0 - sev / 7.0)
            keep = random.random() > drop_prob

        if keep:
            kept.append(sym)
            vec[sym2idx[sym]] = sev / 7.0

    # G2: enforce minimum floor
    min_keep = max(2, int(len(core_symptoms) * 0.5))
    if len(kept) < min_keep:
        missing = [s for s in core_symptoms if s not in kept and s in sym2idx]
        random.shuffle(missing)
        for s in missing[:min_keep - len(kept)]:
            kept.append(s)
            sev = severity.get(s, 3)
            vec[sym2idx[s]] = sev / 7.0

    return vec

print("\n━━━ Step 3: Augmenting dataset ━━━")

X_list, y_list = [], []
random.seed(42)
np.random.seed(42)

for disease, core_syms in disease_map.items():
    label = dis2idx[disease]
    for _ in range(AUGMENT_PER_CLASS):
        vec = augment_sample(disease, core_syms)
        X_list.append(vec)
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# 85/15 split
split = int(len(X) * 0.85)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"  Total samples : {len(X)}")
print(f"  Train         : {len(X_train)}")
print(f"  Validation    : {len(X_val)}")

# Verify fix: check Paralysis samples
print("\n  Verification — Paralysis sample stats:")
p_idx   = dis2idx["Paralysis (brain hemorrhage)"]
p_mask  = (y_train == p_idx)
p_vecs  = X_train[p_mask]
syms_per_sample = (p_vecs > 0).sum(axis=1)
print(f"    Min symptoms in any Paralysis sample : {syms_per_sample.min()}")
print(f"    Avg symptoms per Paralysis sample    : {syms_per_sample.mean():.1f}")

# Check anchors always present
anchor_idx_list = [sym2idx[s] for s in unique_per_disease.get("Paralysis (brain hemorrhage)", set()) if s in sym2idx]
for aidx in anchor_idx_list:
    sym_name = ALL_SYMPTOMS[aidx]
    always_present = (p_vecs[:, aidx] > 0).all()
    print(f"    '{sym_name}' always present: {always_present} ✅" if always_present else f"    '{sym_name}' MISSING in some samples ❌")

# ─────────────────────────────────────────────
# STEP 4 — Build Residual MLP
# ─────────────────────────────────────────────

print("\n━━━ Step 4: Building model ━━━")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                      Input, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU

y_train_cat = to_categorical(y_train, N_DISEASES)
y_val_cat   = to_categorical(y_val,   N_DISEASES)

def build_model(input_dim, output_dim):
    inp = Input(shape=(input_dim,), name="symptom_input")

    # Block 1
    x = Dense(512, kernel_regularizer=l2(1e-4))(inp)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Block 2 + residual connection
    res = Dense(256)(x)
    x   = Dense(256, kernel_regularizer=l2(1e-4))(x)
    x   = LeakyReLU(0.1)(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.35)(x)
    x   = Add()([x, res])

    # Block 3
    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 4
    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    out = Dense(output_dim, activation="softmax", name="output")(x)
    return Model(inputs=inp, outputs=out, name="DrAI_DiseaseModel_v2")

model = build_model(N_SYMPTOMS, N_DISEASES)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ─────────────────────────────────────────────
# STEP 5 — Train
# ─────────────────────────────────────────────

print("\n━━━ Step 5: Training ━━━")

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=20,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=300,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# STEP 6 — Evaluate + Sanity Tests
# ─────────────────────────────────────────────

print("\n━━━ Step 6: Evaluation ━━━")

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

val_probs  = model.predict(X_val, verbose=0)
val_preds  = np.argmax(val_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=DISEASES))

# ── Sanity tests ──
print("\n━━━ Sanity tests (the headache problem) ━━━")

def test_query(label, syms):
    vec   = np.zeros(N_SYMPTOMS, dtype=np.float32)
    found = []
    for s in syms:
        s_norm = s.lower().strip().replace(" ", "_")
        if s_norm in sym2idx:
            sev = severity.get(s_norm, 3)
            vec[sym2idx[s_norm]] = sev / 7.0
            found.append(s_norm)
    probs = model.predict(vec.reshape(1, -1), verbose=0)[0]
    top3  = np.argsort(probs)[::-1][:3]
    print(f"\n  Query: {label} → detected: {found}")
    for i, idx in enumerate(top3):
        print(f"    #{i+1}  {DISEASES[idx]:45s}  {probs[idx]*100:.1f}%")

test_query("headache only",                   ["headache"])
test_query("headache + fever",                ["headache", "high_fever"])
test_query("headache + weakness one side",    ["headache", "weakness_of_one_body_side"])
test_query("headache + vomiting + altered sensorium", ["headache", "vomiting", "altered_sensorium"])
test_query("itching + skin rash",             ["itching", "skin_rash"])
test_query("chest pain + sweating + vomiting",["chest_pain", "sweating", "vomiting"])
test_query("polyuria + fatigue + weight loss",["polyuria", "fatigue", "weight_loss"])

# ── Confusion matrix ──
try:
    cm  = confusion_matrix(y_val, val_preds)
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#13161d')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISEASES, yticklabels=DISEASES,
                ax=ax, linewidths=0.3, cbar=False)
    ax.set_xlabel("Predicted", color='white', fontsize=9)
    ax.set_ylabel("Actual",    color='white', fontsize=9)
    ax.set_title("Dr. AI v2 — Disease Model Confusion Matrix", color='#5b8dee', fontsize=14)
    ax.tick_params(colors='white', labelsize=6)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches='tight')
    print("\n📊 Confusion matrix saved → confusion_matrix.png")
except Exception as e:
    print(f"  (Confusion matrix skipped: {e})")

# ── Training curves ──
try:
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor('#0d0f14')
    for ax in axes:
        ax.set_facecolor('#13161d')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    axes[0].plot(history.history['accuracy'],     color='#5b8dee', lw=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#3ecf8e', lw=2, label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend(facecolor='#1a1e28', labelcolor='white')
    axes[1].plot(history.history['loss'],         color='#5b8dee', lw=2, label='Train')
    axes[1].plot(history.history['val_loss'],     color='#3ecf8e', lw=2, label='Val')
    axes[1].set_title('Loss')
    axes[1].legend(facecolor='#1a1e28', labelcolor='white')
    plt.suptitle('Dr. AI v2 — Training Curves', color='#5b8dee', fontsize=14)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120, bbox_inches='tight')
    print("📊 Training curves saved → training_curves.png")
except Exception as e:
    print(f"  (Training curves skipped: {e})")

# ─────────────────────────────────────────────
# STEP 7 — Save metadata JSON
# ─────────────────────────────────────────────

print("\n━━━ Step 7: Saving metadata ━━━")

disease_info = {}
for disease in DISEASES:
    disease_info[disease] = {
        "description":       descriptions.get(disease, ""),
        "prescription":      prescriptions.get(disease, {}),
        "precautions":       precautions.get(disease, []),
        "symptoms":          disease_map.get(disease, []),
        "unique_symptoms":   sorted(unique_per_disease.get(disease, set()))
    }

metadata = {
    "symptoms":     ALL_SYMPTOMS,
    "diseases":     DISEASES,
    "n_symptoms":   N_SYMPTOMS,
    "n_diseases":   N_DISEASES,
    "disease_info": disease_info,
    "severity":     severity
}

with open(METADATA_OUT, "w") as f:
    json.dump(metadata, f, indent=2)

model.save(MODEL_OUT)

print(f"  ✅ Model    → {MODEL_OUT}")
print(f"  ✅ Metadata → {METADATA_OUT}")
print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING COMPLETE

Fix summary:
  ✓ Anchor symptoms never dropped during augmentation
  ✓ Minimum 50% of core symptoms per sample enforced  
  ✓ No noise symptom injection
  ✓ 250 samples per class (10,250 total)

Expected results:
  • "headache only"     → Migraine/Hypertension  (NOT Paralysis)
  • "headache + weakness of one body side" → Paralysis (correct)
  • Overall val accuracy: 93-98%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
