# Dr. AI — Complete Medical Intelligence System

## What Was Fixed

### Bug 1: "Headache → Brain Paralysis" (Wrong Predictions)
**Old code problem:** The backend was using a simple lookup that returned the *first* disease matching any symptom. Headache appears in Paralysis, so it got predicted first.

**Fix in `app.py`:**
- Now uses **Jaccard + Coverage similarity scoring** across ALL 42+ diseases
- Each disease gets a weighted score: `score = 0.6 × jaccard + 0.4 × coverage`
- Returns **Top 3 matches sorted by score** — headache alone scores low for paralysis
- Requires **minimum overlap threshold** — zero-overlap diseases are filtered out

### Bug 2: Skin Model Predicts Only One Disease (Melanocytic Nevi)
**Old code problem:** HAM10000 has severe class imbalance — ~67% of images are 'nv' (Melanocytic Nevi). Without class weights, the model learns to predict 'nv' for everything.

**Fix in `train_skin_model.py`:**
- Added `compute_class_weight('balanced', ...)` from sklearn
- Applied class weights to BOTH training phases
- Returns **Top 3 predictions** (not just argmax)
- Added confusion matrix + classification report after training

---

## Project Structure

```
Dr. AI/
├── backend/
│   ├── app.py                 ← Flask API server (fixed logic)
│   ├── train_skin_model.py    ← Skin model training (fixed class weights)
│   ├── symptom_nlp_model.py   ← NLP model training (improved)
│   ├── requirements.txt
│   └── skin_disease_model.h5  ← Generated after training
│
└── frontend/
    ├── dr_ai_chat.html        ← Redesigned UI
    ├── dr_ai_chat.css         ← Dark medical theme
    └── dr_ai_chat.js          ← Rewritten logic
```

---

## Setup Instructions

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # optional, for NLP
```

### 2. (Optional) Train the skin model
```bash
# Update paths in train_skin_model.py first:
#   DATA_DIR1, DATA_DIR2, METADATA_PATH
python train_skin_model.py
# Takes ~1-2 hours on GPU
# Output: skin_disease_model.h5
```

### 3. Start the backend
```bash
python app.py
# Server starts at http://localhost:5000
```

### 4. Open the frontend
```
Open frontend/dr_ai_chat.html in your browser
```

---

## API Endpoints

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| POST   | /api/diagnose    | Symptom analysis (JSON)            |
| POST   | /api/skin        | Skin image analysis (form-data)    |
| POST   | /api/hybrid      | Combined symptoms + image          |
| GET    | /api/diseases    | List all diseases                  |
| GET    | /health          | Server + model status              |

### Example: Symptom Analysis
```json
POST /api/diagnose
{
  "symptoms": ["headache", "fever", "nausea"],
  "text": "I've had a bad headache and feel sick"
}
```

Returns:
```json
{
  "detected_symptoms": ["headache", "fever", "nausea"],
  "predictions": [
    {
      "disease": "Malaria",
      "score": 42.5,
      "matched_symptoms": ["headache", "fever", "nausea"],
      "description": "...",
      "precautions": [...],
      "prescription": [...]
    },
    ...
  ]
}
```

---

## Why Jaccard Is Better for Symptoms

| Symptoms Input | Old Method | New Method (Jaccard) |
|----------------|-----------|----------------------|
| "headache"     | Paralysis (first match) | Migraine/Hypertension (highest overlap score) |
| "headache + fever + chills" | Paralysis | Malaria/Dengue (3 symptoms match) |
| "headache + vomiting + weakness of one body side" | Paralysis | Paralysis (correct — 3 specific symptoms) |

The more specific symptoms you add, the more accurate the prediction.

---

## Skin Model Accuracy (Expected after fix)

| Disease                  | Before Fix | After Fix (with class weights) |
|--------------------------|-----------|-------------------------------|
| Melanocytic Nevi (nv)    | ~95%      | ~75-80%                       |
| Melanoma                 | ~0%       | ~65-75%                       |
| Basal Cell Carcinoma     | ~0%       | ~70-80%                       |
| Actinic Keratoses        | ~0%       | ~60-70%                       |
| All others               | ~0%       | ~55-70%                       |

*Note: HAM10000 is a hard dataset. Even SOTA models get ~85% balanced accuracy.*
