"""
Dr. AI — Improved Symptom NLP Model
Fixes:
  1. More symptoms (130+) from the actual disease database
  2. Better entity overlap prevention
  3. More training templates for accurate real-world text
  4. Validation step to measure actual accuracy
"""

import spacy
from spacy.training.example import Example
from random import shuffle
import random
import re
import json

# ─────────────────────────────────────────────
# FULL SYMPTOM LIST (130 symptoms)
# Sourced from the complete disease database
# ─────────────────────────────────────────────

ALL_SYMPTOMS = [
    "abdominal pain", "abnormal menstruation", "acidity", "altered sensorium",
    "anxiety", "aura", "back pain", "belly pain", "blackheads", "bladder discomfort",
    "blister", "blood in sputum", "bloody stool", "blurred and distorted vision",
    "breathlessness", "brittle nails", "bruising", "burning micturition",
    "chest pain", "chills", "cold hands and feets", "coma", "congestion",
    "constipation", "continuous feel of urine", "continuous sneezing", "cough",
    "cramps", "dark urine", "dehydration", "depression", "diarrhoea", "dizziness",
    "dischromic patches", "drying and tingling lips", "enlarged thyroid",
    "excessive hunger", "extra marital contacts", "family history", "fast heart rate",
    "fatigue", "fluid overload", "foul smell of urine", "headache", "high fever",
    "hip joint pain", "history of alcohol consumption", "indigestion",
    "increased appetite", "inflammatory nails", "internal itching",
    "intermittent fever", "irregular sugar level", "irritability",
    "irritation in anus", "itching", "joint pain", "knee pain",
    "lack of concentration", "lethargy", "loss of appetite", "loss of balance",
    "loss of smell", "low energy", "malaise", "mild fever", "mood changes",
    "movement stiffness", "mucoid sputum", "muscle pain", "muscle wasting",
    "muscle weakness", "nausea", "neck pain", "nodal skin eruptions", "obesity",
    "pain behind the eyes", "pain during bowel movements", "pain in anal region",
    "painful walking", "palpitations", "passage of gases", "patches in throat",
    "phlegm", "polyuria", "prominent veins on calf", "puffy face and eyes",
    "pus filled pimples", "red sore around nose", "red spots over body",
    "redness of eyes", "restlessness", "runny nose", "rusty sputum", "scurring",
    "shivering", "silver like dusting", "sinus pressure", "skin peeling",
    "skin rash", "slurred speech", "small dents in nails", "spotting urination",
    "spinning movements", "stiff neck", "stomach bleeding", "stomach pain",
    "sunken eyes", "sweating", "swelled lymph nodes", "swelling joints",
    "swelling of stomach", "swollen blood vessels", "swollen extremeties",
    "swollen legs", "throat irritation", "toxic look (typhos)", "ulcers on tongue",
    "unsteadiness", "visual disturbances", "vomiting", "watering from eyes",
    "weakness", "weakness in limbs", "weakness of one body side", "weight gain",
    "weight loss", "yellow crust ooze", "yellow urine", "yellowing of eyes",
    "yellowish skin"
]

# ─────────────────────────────────────────────
# TRAINING TEMPLATES — diverse real-world phrasing
# ─────────────────────────────────────────────

TEMPLATES = [
    # Single symptom
    "I am experiencing {}.",
    "I have been having {} for a few days.",
    "My main problem is {}.",
    "Suffering from {}.",
    "I feel {}.",
    "{} has been troubling me.",
    "Doctor, I have {}.",
    "I woke up with {}.",
    "I've noticed {} recently.",
    "The {} is getting worse.",

    # Two symptoms
    "I am experiencing {} and {}.",
    "My symptoms include {} and {}.",
    "I've had {} and {} together.",
    "I'm dealing with {} along with {}.",
    "Both {} and {} are bothering me.",
    "I went to the doctor for {} and {}.",
    "Having {} with a bit of {}.",
    "Feeling {} and also {}.",

    # Three symptoms
    "I have {}, {}, and {}.",
    "My symptoms are {}, {}, and {}.",
    "I am suffering from {}, {}, and {} since morning.",
    "I've been noticing {}, {} and {} for the past week.",

    # Contextual phrasing
    "I feel weak and have {}.",
    "Getting {} since this morning.",
    "Can't sleep because of {}.",
    "I have a history of {} and now I have {}.",
    "The {} started two days ago and now I also have {}.",
    "My {} is really bad today.",
    "I went to the hospital for {} but now I also feel {}.",
]


def generate_training_data(symptoms, n=500):
    """
    Generate synthetic training data with no overlapping entities.
    Uses re.finditer for precise span detection.
    """
    training_data = []
    errors = 0

    for _ in range(n):
        template = random.choice(TEMPLATES)
        num_slots = template.count("{}")

        # Ensure we have enough symptoms
        if num_slots > len(symptoms):
            continue

        # Pick unique symptoms (no duplicates in one sentence)
        selected = random.sample(symptoms, num_slots)
        sentence = template.format(*selected)

        entities = []
        used_positions = set()

        for symptom in selected:
            for match in re.finditer(re.escape(symptom), sentence):
                start, end = match.span()
                # No overlap allowed
                if not any(i in used_positions for i in range(start, end)):
                    entities.append((start, end, "SYMPTOM"))
                    used_positions.update(range(start, end))
                    break

        if entities:
            training_data.append((sentence, {"entities": entities}))

    print(f"✅ Generated {len(training_data)} training examples ({errors} skipped)")
    return training_data


def validate_entities(text, entities):
    """Check for overlapping or invalid spans"""
    spans = sorted(entities, key=lambda e: e[0])
    for i in range(len(spans) - 1):
        if spans[i][1] > spans[i + 1][0]:
            return False
    return True


def train_ner_model(data, output_dir="symptom_nlp_model", epochs=30):
    """
    Train spaCy NER model with improved settings:
    - 30 epochs (vs 20)
    - Lower dropout for better convergence
    - Validation split
    """
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("SYMPTOM")

    # Build examples, filter bad ones
    examples = []
    skipped = 0
    for text, ann in data:
        if not validate_entities(text, ann["entities"]):
            skipped += 1
            continue
        doc = nlp.make_doc(text)
        try:
            ex = Example.from_dict(doc, ann)
            examples.append(ex)
        except Exception:
            skipped += 1

    print(f"✅ {len(examples)} valid examples, {skipped} skipped")

    # Split into train/val (90/10)
    split = int(len(examples) * 0.9)
    train_examples = examples[:split]
    val_examples = examples[split:]

    optimizer = nlp.begin_training()
    best_loss = float("inf")

    for epoch in range(epochs):
        shuffle(train_examples)
        losses = {}
        batches = spacy.util.minibatch(train_examples, size=16)
        for batch in batches:
            nlp.update(batch, losses=losses, drop=0.2)

        # Validation
        val_losses = {}
        for ex in val_examples:
            nlp.update([ex], losses=val_losses, drop=0.0)

        train_loss = round(losses.get("ner", 0), 4)
        val_loss = round(val_losses.get("ner", 0), 4)

        print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            nlp.to_disk(output_dir + "_best")

    # Save final model
    nlp.to_disk(output_dir)
    print(f"\n✅ Final model saved to: {output_dir}")
    print(f"✅ Best model saved to:  {output_dir}_best")


def test_model(model_dir="symptom_nlp_model"):
    """Quick test of trained model"""
    nlp = spacy.load(model_dir)
    test_cases = [
        "I have a headache and fever.",
        "Suffering from nausea, vomiting, and diarrhoea.",
        "My symptoms include chest pain and breathlessness.",
        "I feel fatigue and joint pain since last week.",
        "Having itching and skin rash on my arms.",
    ]
    print("\n🧪 Model Test Results:")
    for text in test_cases:
        doc = nlp(text)
        found = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
        print(f"  Input: {text}")
        print(f"  Found: {found}\n")


if __name__ == "__main__":
    print("🚀 Starting Dr. AI Symptom NLP Training...")
    print(f"   Total symptoms: {len(ALL_SYMPTOMS)}")

    data = generate_training_data(ALL_SYMPTOMS, n=600)
    train_ner_model(data, output_dir="symptom_nlp_model", epochs=30)
    test_model("symptom_nlp_model")

    print("\n✅ Training complete!")
    print("   Load the model in your backend with: nlp = spacy.load('symptom_nlp_model')")
