"""
Dr. AI — Flask Backend  (v2 — Fixed)
======================================
Disease prediction uses trained DNN (disease_model.h5).
Fallback uses fixed Jaccard with small-disease penalty.
"""
import os, json
import numpy as np
from flask import Flask, request, jsonify
from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── globals ──
disease_model = None
skin_model    = None
meta          = None
ALL_SYMPTOMS  = []
DISEASES      = []
DISEASE_INFO  = {}
SEVERITY      = {}
SYM2IDX       = {}
UNIQUE_SYMS   = {}   # unique discriminative symptoms per disease


def load_models():
    global disease_model, skin_model
    global ALL_SYMPTOMS, DISEASES, DISEASE_INFO, SEVERITY, SYM2IDX, UNIQUE_SYMS

    print("━━━ Loading Dr. AI models ━━━")

    # ── Disease DNN ──
    try:
        from tensorflow.keras.models import load_model
        disease_model = load_model("disease_model.h5", compile=False)
        disease_model.compile(optimizer="adam",
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])
        print("  ✅ disease_model.h5 loaded  (Deep Neural Network v2)")
    except Exception as e:
        print(f"  ⚠️  disease_model.h5 not found — run train_disease_model.py first!")
        print(f"     Error: {e}")

    # ── Metadata ──
    try:
        with open("model_metadata.json") as f:
            meta = json.load(f)
        ALL_SYMPTOMS = meta["symptoms"]
        DISEASES     = meta["diseases"]
        DISEASE_INFO = meta["disease_info"]
        SEVERITY     = meta["severity"]
        SYM2IDX      = {s: i for i, s in enumerate(ALL_SYMPTOMS)}
        UNIQUE_SYMS  = {d: set(info.get("unique_symptoms", []))
                        for d, info in DISEASE_INFO.items()}
        print(f"  ✅ metadata loaded  ({len(DISEASES)} diseases, {len(ALL_SYMPTOMS)} symptoms)")
    except Exception as e:
        print(f"  ⚠️  model_metadata.json not found — run train_disease_model.py first!")
        print(f"     Error: {e}")

    # ── Skin CNN (optional) ──
    try:
        from tensorflow.keras.models import load_model as lm
        skin_model = lm("skin_disease_model.h5", compile=False)
        print("  ✅ skin_disease_model.h5 loaded  (EfficientNetB0)")
    except Exception:
        print("  ℹ️  Skin model not found — upload analysis disabled")


# ─────────────────────────────────────────────
# SYMPTOM PARSING
# ─────────────────────────────────────────────

def normalise(s):
    return s.lower().strip().replace(" ", "_").replace("-", "_")

def text_to_symptoms(text):
    """Extract recognised symptoms from free-text patient input"""
    text_lower = text.lower()
    text_norm  = normalise(text)
    found = []
    for sym in ALL_SYMPTOMS:
        sym_spaced = sym.replace("_", " ")
        if sym in text_norm or sym_spaced in text_lower:
            found.append(sym)
    return found

def build_input_vector(symptoms):
    """
    Severity-weighted input vector for DNN.
    Same preprocessing as used during training.
    """
    vec = np.zeros(len(ALL_SYMPTOMS), dtype=np.float32)
    for sym in symptoms:
        sym_norm = normalise(sym)
        if sym_norm in SYM2IDX:
            sev = SEVERITY.get(sym_norm, 3)
            vec[SYM2IDX[sym_norm]] = sev / 7.0
    return vec


# ─────────────────────────────────────────────
# DISEASE PREDICTION — DNN
# ─────────────────────────────────────────────

def predict_diseases(symptoms, top_n=3):
    detected = [normalise(s) for s in symptoms if normalise(s) in SYM2IDX]

    if not detected:
        return [], []

    if disease_model is None or not ALL_SYMPTOMS:
        return fallback_predict(detected, top_n), detected

    # ── DNN inference ──
    vec   = build_input_vector(detected).reshape(1, -1)
    probs = disease_model.predict(vec, verbose=0)[0]   # shape (41,)

    top_idx = np.argsort(probs)[::-1][:top_n]
    results = []
    for idx in top_idx:
        disease = DISEASES[idx]
        conf    = round(float(probs[idx]) * 100, 2)
        info    = DISEASE_INFO.get(disease, {})
        core    = set(info.get("symptoms", []))
        matched = [s for s in detected if s in core]
        results.append({
            "disease":          disease,
            "confidence":       conf,
            "matched_symptoms": matched,
            "description":      info.get("description", ""),
            "prescription":     info.get("prescription", {}),
            "precautions":      info.get("precautions", [])
        })

    return results, detected


def fallback_predict(symptoms, top_n=3):
    """
    Fixed Jaccard fallback — penalises diseases with few total symptoms
    so small-symptom diseases (Paralysis, 4 syms) don't dominate on
    single-symptom queries.
    """
    input_set = set(symptoms)
    scores = []

    for disease, info in DISEASE_INFO.items():
        core  = set(normalise(s) for s in info.get("symptoms", []))
        inter = input_set & core
        if not inter:
            continue

        union    = input_set | core
        jaccard  = len(inter) / len(union)
        coverage = len(inter) / len(input_set) if input_set else 0

        # ── Small-disease penalty ──
        # A disease with only 4 symptoms gets a size penalty so it doesn't
        # beat diseases with more symptoms on partial queries.
        # Penalty scales with (min_expected_symptoms / actual_core_size)
        min_expected = 8   # average disease has ~8-10 symptoms in this dataset
        size_penalty = min(1.0, len(core) / min_expected)

        score = round((0.5 * jaccard + 0.3 * coverage + 0.2 * size_penalty) * 100, 2)

        scores.append({
            "disease":          disease,
            "confidence":       score,
            "matched_symptoms": list(inter),
            "description":      info.get("description", ""),
            "prescription":     info.get("prescription", {}),
            "precautions":      info.get("precautions", [])
        })

    scores.sort(key=lambda x: x["confidence"], reverse=True)
    return scores[:top_n]


# ─────────────────────────────────────────────
# SKIN PREDICTION
# ─────────────────────────────────────────────

SKIN_INFO = {
    "akiec": {"name":"Actinic Keratoses","risk":"Pre-cancerous",
              "description":"Rough scaly patches on sun-damaged skin. Can progress to squamous cell carcinoma.",
              "prescription":"Cryotherapy, topical 5-fluorouracil, imiquimod, or photodynamic therapy.",
              "precaution":"Daily SPF 30+ sunscreen, protective clothing, avoid peak sun hours, annual skin checks."},
    "bcc":   {"name":"Basal Cell Carcinoma","risk":"Malignant",
              "description":"Most common skin cancer from basal cells. Rarely spreads but causes local damage.",
              "prescription":"Surgical excision (Mohs preferred), topical imiquimod or 5-FU, radiation.",
              "precaution":"Avoid UV exposure, broad-spectrum sunscreen, monitor for new or changing lesions."},
    "bkl":   {"name":"Benign Keratosis-like Lesions","risk":"Benign",
              "description":"Non-cancerous waxy or scaly growths including seborrheic keratoses.",
              "prescription":"No treatment required. Cryotherapy for cosmetic removal if desired.",
              "precaution":"Monitor for changes. Annual skin check-ups. Sun protection."},
    "df":    {"name":"Dermatofibroma","risk":"Benign",
              "description":"Common benign fibrous skin bump usually on the legs.",
              "prescription":"No treatment necessary. Surgical excision if bothersome.",
              "precaution":"Monitor for changes. Consult dermatologist if rapid growth occurs."},
    "mel":   {"name":"Melanoma","risk":"Malignant — URGENT",
              "description":"Most serious skin cancer. Can spread rapidly to other organs.",
              "prescription":"IMMEDIATE surgical excision. Immunotherapy or targeted therapy for advanced cases.",
              "precaution":"⚠️ URGENT: Consult oncologist immediately. Avoid all UV exposure."},
    "nv":    {"name":"Melanocytic Nevi (Moles)","risk":"Benign",
              "description":"Common benign pigmented lesions from clusters of melanocytes.",
              "prescription":"No treatment unless atypical features present (ABCDE criteria).",
              "precaution":"Monthly self-exams, annual dermatologist checks, sunscreen, track ABCDE changes."},
    "vasc":  {"name":"Vascular Lesions","risk":"Benign",
              "description":"Lesions from blood vessels including cherry angiomas and pyogenic granulomas.",
              "prescription":"Laser therapy, electrocautery, or surgical excision depending on type.",
              "precaution":"Avoid trauma. Consult dermatologist for growing or bleeding lesions."}
}
SKIN_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def predict_skin(image_file, top_n=3):
    if skin_model is None:
        return {"error": "Skin model not loaded. Run train_skin_model.py first.", "predictions": []}
    try:
        from PIL import Image
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img  = Image.open(image_file).convert("RGB").resize((224, 224))
        arr  = np.array(img, dtype=np.float32)
        arr  = preprocess_input(arr)
        arr  = np.expand_dims(arr, axis=0)
        pred = skin_model.predict(arr, verbose=0)[0]
        results = []
        for idx in np.argsort(pred)[::-1][:top_n]:
            dx_code = SKIN_LABELS[idx]
            conf    = round(float(pred[idx]) * 100, 2)
            if conf < 1.0:
                break
            info = SKIN_INFO[dx_code]
            results.append({"dx_code": dx_code, "label": info["name"],
                             "confidence": conf, "risk": info["risk"],
                             "description": info["description"],
                             "prescription": info["prescription"],
                             "precaution": info["precaution"]})
        return {"predictions": results}
    except Exception as e:
        return {"error": str(e), "predictions": []}


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data     = request.get_json(force=True)
    sym_list = data.get("symptoms", [])
    text     = data.get("text", "")

    if text:
        sym_list = list(set(sym_list + text_to_symptoms(text)))

    if not sym_list:
        return jsonify({"error": "No symptoms provided",
                        "predictions": [], "detected_symptoms": []})

    preds, detected = predict_diseases(sym_list, top_n=3)
    return jsonify({
        "detected_symptoms": detected,
        "predictions":       preds,
        "model": "Deep Neural Network v2" if disease_model else "Rule-based (train model first)"
    })


@app.route("/api/skin", methods=["POST"])
def analyze_skin():
    if "image" not in request.files:
        return jsonify({"error": "No image provided", "predictions": []})
    return jsonify(predict_skin(request.files["image"], top_n=3))


@app.route("/api/hybrid", methods=["POST"])
def hybrid():
    sym_list = [s.strip() for s in request.form.get("symptoms","").split(",") if s.strip()]
    text     = request.form.get("text", "")
    img_type = request.form.get("image_type", "")
    response = {}

    if text:
        sym_list = list(set(sym_list + text_to_symptoms(text)))

    if sym_list:
        preds, detected = predict_diseases(sym_list, top_n=3)
        response["symptom_analysis"] = {
            "detected_symptoms": detected,
            "predictions": preds,
            "model": "Deep Neural Network v2" if disease_model else "Rule-based"
        }

    if "image" in request.files and img_type == "skin":
        response["skin_analysis"] = predict_skin(request.files["image"], top_n=3)

    if not response:
        return jsonify({"error": "No input provided"})
    return jsonify(response)


@app.route("/api/symptoms", methods=["GET"])
def list_symptoms():
    return jsonify({"symptoms": ALL_SYMPTOMS, "count": len(ALL_SYMPTOMS)})


@app.route("/api/diseases", methods=["GET"])
def list_diseases():
    return jsonify({"diseases": DISEASES, "count": len(DISEASES)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "disease_model": disease_model is not None,
        "skin_model":    skin_model is not None,
        "n_diseases":    len(DISEASES),
        "n_symptoms":    len(ALL_SYMPTOMS),
        "model_type":    "Deep Neural Network v2 (MLP + Residual, Fixed Augmentation)"
                         if disease_model else "Fallback (model not trained)"
    })


# Home page
@app.route("/")
def home():
    return render_template("drfront.html")


@app.route("/dr_ai_chat")
@app.route("/dr_ai_chat.html")
def dr_ai_chat():
    return render_template("dr_ai_chat.html")

# Load models
load_models()


if __name__ == "__main__":
    print("\n━━━ Dr. AI server ready ━━━")
    print("  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)