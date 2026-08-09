/* ── Dr. AI — Frontend Logic ── */

const API_BASE = "http://localhost:5000";

// Symptoms are loaded dynamically from /api/symptoms (matches trained model exactly)
// Fallback list used if server is offline
const FALLBACK_SYMPTOMS = [
    "abdominal_pain","abnormal_menstruation","acidity","altered_sensorium",
    "anxiety","back_pain","belly_pain","blackheads","bladder_discomfort",
    "blister","blood_in_sputum","bloody_stool","blurred_and_distorted_vision",
    "breathlessness","brittle_nails","bruising","burning_micturition",
    "chest_pain","chills","cold_hands_and_feets","coma","congestion",
    "constipation","continuous_feel_of_urine","continuous_sneezing","cough",
    "cramps","dark_urine","dehydration","depression","diarrhoea","dizziness",
    "dischromic__patches","drying_and_tingling_lips","enlarged_thyroid",
    "excessive_hunger","extra_marital_contacts","family_history","fast_heart_rate",
    "fatigue","fluid_overload","foul_smell_of_urine","headache","high_fever",
    "hip_joint_pain","history_of_alcohol_consumption","indigestion",
    "increased_appetite","inflammatory_nails","internal_itching",
    "intermittent_fever","irregular_sugar_level","irritability",
    "irritation_in_anus","itching","joint_pain","knee_pain",
    "lack_of_concentration","lethargy","loss_of_appetite","loss_of_balance",
    "loss_of_smell","low_energy","malaise","mild_fever","mood_swings",
    "movement_stiffness","mucoid_sputum","muscle_pain","muscle_wasting",
    "muscle_weakness","nausea","neck_pain","nodal_skin_eruptions","obesity",
    "pain_behind_the_eyes","pain_during_bowel_movements","pain_in_anal_region",
    "painful_walking","palpitations","passage_of_gases","patches_in_throat",
    "phlegm","polyuria","prominent_veins_on_calf","puffy_face_and_eyes",
    "pus_filled_pimples","red_sore_around_nose","red_spots_over_body",
    "redness_of_eyes","restlessness","runny_nose","rusty_sputum","scurring",
    "shivering","silver_like_dusting","sinus_pressure","skin_peeling",
    "skin_rash","slurred_speech","small_dents_in_nails","spotting__urination",
    "spinning_movements","stiff_neck","stomach_bleeding","stomach_pain",
    "sunken_eyes","sweating","swelled_lymph_nodes","swelling_joints",
    "swelling_of_stomach","swollen_blood_vessels","swollen_extremeties",
    "swollen_legs","throat_irritation","toxic_look_(typhos)","ulcers_on_tongue",
    "unsteadiness","visual_disturbances","vomiting","watering_from_eyes",
    "weakness_in_limbs","weakness_of_one_body_side","weight_gain",
    "weight_loss","yellow_crust_ooze","yellow_urine","yellowing_of_eyes","yellowish_skin"
];

let SYMPTOMS = [...FALLBACK_SYMPTOMS];

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

let selectedSymptoms = new Set();
let skinFile = null;
let xrayFile = null;
let isListening = false;
let recognition = null;
let history = JSON.parse(localStorage.getItem("dr_ai_history") || "[]");

// ─────────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────────

const symptomGrid   = document.getElementById("symptomGrid");
const symptomSearch = document.getElementById("symptomSearch");
const selectedBadges= document.getElementById("selectedBadges");
const chatMessages  = document.getElementById("chatMessages");
const chatInput     = document.getElementById("chatInput");
const sendBtn       = document.getElementById("sendBtn");
const analyzeBtn    = document.getElementById("analyzeBtn");
const analyzeImageBtn = document.getElementById("analyzeImageBtn");
const micBtn        = document.getElementById("micBtn");
const historyList   = document.getElementById("historyList");

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
    setupTabs();
    setupUpload("skinFile", "skinPreview", "skinPreviewImg", "skinFileName", "skin");
    setupUpload("xrayFile", "xrayPreview", "xrayPreviewImg", "xrayFileName", "xray");
    setupVoice();
    renderHistory();
    showWelcome();

    // Load live symptom list + health info from backend
    try {
        const [symRes, healthRes] = await Promise.all([
            fetch(API_BASE + "/api/symptoms"),
            fetch(API_BASE + "/health")
        ]);
        if (symRes.ok) {
            const data = await symRes.json();
            if (data.symptoms && data.symptoms.length > 0) {
                SYMPTOMS = data.symptoms;
                console.log("Loaded " + SYMPTOMS.length + " symptoms from backend");
            }
        }
        if (healthRes.ok) {
            const h = await healthRes.json();
            const badge = document.querySelector(".accuracy-badge");
            if (badge) {
                if (h.disease_model) {
                    badge.innerHTML = "<i class='fa-solid fa-brain'></i> DNN Active · " + h.n_diseases + " diseases";
                    badge.style.background = "rgba(62,207,142,0.1)";
                    badge.style.borderColor = "rgba(62,207,142,0.25)";
                    badge.style.color = "var(--success)";
                } else {
                    badge.innerHTML = "<i class='fa-solid fa-triangle-exclamation'></i> Train model first";
                    badge.style.background = "rgba(240,160,64,0.1)";
                    badge.style.borderColor = "rgba(240,160,64,0.25)";
                    badge.style.color = "var(--warn)";
                }
            }
        }
    } catch(e) {
        console.warn("Backend not reachable — using offline symptoms");
    }

    renderSymptoms();
});

// ─────────────────────────────────────────────
// WELCOME MESSAGE
// ─────────────────────────────────────────────

function showWelcome() {
    const welcome = `Hello! I'm **Dr. AI**, your intelligent medical assistant.

I can help you with:
• **Symptom analysis** — select from the sidebar or describe in your own words
• **Skin condition analysis** — upload a photo for AI assessment
• **Disease information** — detailed descriptions, precautions & prescriptions

I use **Jaccard similarity scoring** to match your symptoms against 42+ diseases accurately — so adding more symptoms gives better results.

*Always consult a licensed physician for diagnosis and treatment.*`;

    appendBotMessage(welcome, { isWelcome: true });
}

// ─────────────────────────────────────────────
// TABS
// ─────────────────────────────────────────────

function setupTabs() {
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
        });
    });
}

// ─────────────────────────────────────────────
// SYMPTOM RENDERING
// ─────────────────────────────────────────────

function renderSymptoms(filter = "") {
    symptomGrid.innerHTML = "";
    const q = filter.toLowerCase().replace(/ /g, "_").trim();
    // Match against both the raw snake_case AND the display (spaced) form
    const filtered = SYMPTOMS.filter(s =>
        s.includes(q) || s.replace(/_/g, " ").includes(filter.toLowerCase().trim())
    );

    if (filtered.length === 0) {
        const empty = document.createElement("div");
        empty.style.cssText = "font-size:12px;color:var(--text-3);padding:12px 8px;text-align:center;";
        empty.textContent = "No symptoms matched";
        symptomGrid.appendChild(empty);
        return;
    }

    filtered.forEach(s => {
        const btn = document.createElement("button");
        btn.className = "sym-item" + (selectedSymptoms.has(s) ? " active" : "");
        btn.dataset.symptom = s;
        btn.textContent = capitalize(s);
        symptomGrid.appendChild(btn);
    });
}

symptomSearch.addEventListener("input", e => renderSymptoms(e.target.value));

symptomGrid.addEventListener("click", e => {
    const btn = e.target.closest(".sym-item");
    if (!btn) return;
    const sym = btn.dataset.symptom;

    if (selectedSymptoms.has(sym)) {
        selectedSymptoms.delete(sym);
        btn.classList.remove("active");
    } else {
        selectedSymptoms.add(sym);
        btn.classList.add("active");
    }
    renderBadges();
    analyzeBtn.disabled = selectedSymptoms.size === 0;
});

function renderBadges() {
    selectedBadges.innerHTML = "";
    selectedSymptoms.forEach(sym => {
        const badge = document.createElement("div");
        badge.className = "badge";
        badge.innerHTML = `${capitalize(sym)} <button onclick="removeSym('${sym}')"><i class='fa-solid fa-xmark'></i></button>`;
        selectedBadges.appendChild(badge);
    });
}

function removeSym(sym) {
    selectedSymptoms.delete(sym);
    renderSymptoms(symptomSearch.value);
    renderBadges();
    analyzeBtn.disabled = selectedSymptoms.size === 0;
}

// ─────────────────────────────────────────────
// ANALYZE SYMPTOMS BUTTON
// ─────────────────────────────────────────────

analyzeBtn.addEventListener("click", () => {
    if (selectedSymptoms.size === 0) return;
    const syms = Array.from(selectedSymptoms);
    appendUserMessage(`Selected symptoms: ${syms.map(capitalize).join(", ")}`);
    diagnose(syms, "");
});

// ─────────────────────────────────────────────
// TEXT INPUT (free text)
// ─────────────────────────────────────────────

sendBtn.addEventListener("click", sendText);
chatInput.addEventListener("keydown", e => { if (e.key === "Enter") sendText(); });

function sendText() {
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = "";
    appendUserMessage(text);
    const syms = Array.from(selectedSymptoms);
    diagnose(syms, text);
}

// ─────────────────────────────────────────────
// DIAGNOSE — SYMPTOM API CALL
// ─────────────────────────────────────────────

async function diagnose(symptoms, text) {
    const loadingId = showTyping();

    try {
        const res = await fetch(`${API_BASE}/api/diagnose`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symptoms, text })
        });

        removeTyping(loadingId);

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        if (data.error) {
            appendBotMessage(`⚠️ ${data.error}`);
            return;
        }

        renderDiagnosisResults(data);
        saveHistory(symptoms, text, data);

    } catch (err) {
        removeTyping(loadingId);
        appendBotMessage(`❌ **Connection error:** Could not reach the Dr. AI backend.\n\nMake sure the Flask server is running:\n\`\`\`\ncd backend\npython app.py\n\`\`\``);
    }
}

// ─────────────────────────────────────────────
// RENDER DIAGNOSIS RESULTS
// ─────────────────────────────────────────────

function renderDiagnosisResults(data) {
    const container = document.createElement("div");
    container.style.cssText = "display:flex;flex-direction:column;gap:8px;width:100%;max-width:700px;";

    const { detected_symptoms, predictions, model } = data;

    if (!predictions || predictions.length === 0) {
        appendBotMessage("I could not find a strong match for those symptoms. Please add more specific symptoms or describe your condition in more detail.");
        return;
    }

    // Model badge row
    const modelBadge = model
        ? `<span style="font-size:10px;padding:2px 8px;border-radius:10px;background:rgba(91,141,238,0.12);color:var(--accent);border:1px solid rgba(91,141,238,0.2);margin-left:8px;">${model}</span>`
        : "";

    // Detected symptoms section
    if (detected_symptoms && detected_symptoms.length > 0) {
        const detWrap = document.createElement("div");
        detWrap.innerHTML =
            '<div class="results-header"><i class="fa-solid fa-circle-nodes"></i> DETECTED SYMPTOMS' + modelBadge + '</div>' +
            '<div class="detected-symptoms">' +
            detected_symptoms.map(s => '<span class="det-sym">' + capitalize(s) + '</span>').join("") +
            '</div>';
        container.appendChild(detWrap);
    }

    // Results header
    const hdr = document.createElement("div");
    hdr.className = "results-header";
    hdr.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> TOP ' + predictions.length + ' POSSIBLE CONDITIONS';
    container.appendChild(hdr);

    predictions.forEach((pred, idx) => {
        container.appendChild(buildResultCard(pred, idx + 1, "symptom"));
    });

    const disclaimer = document.createElement("div");
    disclaimer.style.cssText = "font-size:11px;color:var(--text-3);padding:6px 10px;border-left:2px solid var(--border);margin-top:2px;";
    disclaimer.textContent = "⚕️ These predictions are for informational purposes only. Always consult a licensed physician for diagnosis.";
    container.appendChild(disclaimer);

    appendBotElement(container);
}

function formatPrescription(presc) {
    // Handles both old string format and new {first_line, second_line, last_line} object
    if (!presc) return "—";
    if (typeof presc === "string") return presc;
    if (Array.isArray(presc)) return presc.join(" · ");
    const parts = [];
    if (presc.first_line)  parts.push("<strong>1st Line:</strong> "  + presc.first_line);
    if (presc.second_line) parts.push("<strong>2nd Line:</strong> " + presc.second_line);
    if (presc.last_line)   parts.push("<strong>Escalation:</strong> "   + presc.last_line);
    return parts.join("<br>");
}

function buildResultCard(pred, rank, type) {
    const card = document.createElement("div");
    card.className = "result-card rank-" + rank;

    const score = pred.score || pred.confidence || 0;
    const fillColor = rank === 1 ? "var(--success)" : rank === 2 ? "var(--accent)" : "var(--accent-2)";
    // Cap bar width at 100% visually
    const barWidth = Math.min(score, 100);

    let headerHTML = "";
    let bodyHTML   = "";

    if (type === "symptom") {
        headerHTML =
            '<div class="result-card-title">' +
            '<span style="color:var(--text-3);font-size:12px;">#' + rank + '</span>' +
            pred.disease +
            '</div>' +
            '<div class="confidence-bar-wrap">' +
            '<div class="confidence-bar"><div class="confidence-fill" style="width:' + barWidth + '%;background:' + fillColor + ';"></div></div>' +
            '<span class="confidence-label" style="color:' + fillColor + '">' + score + '%</span>' +
            '</div>';

        const matchedHTML = pred.matched_symptoms && pred.matched_symptoms.length > 0
            ? '<div class="result-section"><div class="result-section-label">MATCHED SYMPTOMS</div>' +
              '<div class="symptom-match-list">' +
              pred.matched_symptoms.map(s => '<span class="match-tag">' + capitalize(s) + '</span>').join("") +
              '</div></div>'
            : "";

        bodyHTML =
            matchedHTML +
            '<div class="result-section"><div class="result-section-label">ABOUT</div>' + (pred.description || "—") + '</div>' +
            '<div class="result-section"><div class="result-section-label">PRECAUTIONS</div>' +
            (Array.isArray(pred.precautions) ? pred.precautions.map((p,i) => (i+1) + '. ' + p).join("  ") : pred.precautions || "—") +
            '</div>' +
            '<div class="result-section"><div class="result-section-label">PRESCRIPTION</div>' +
            formatPrescription(pred.prescription) + '</div>';

    } else if (type === "skin") {
        const riskClass = (pred.risk || "").toLowerCase().includes("malignant") ? "risk-malignant"
                       : (pred.risk || "").toLowerCase().includes("pre")        ? "risk-precan"
                       : "risk-benign";

        headerHTML =
            '<div class="result-card-title">' +
            '<span style="color:var(--text-3);font-size:12px;">#' + rank + '</span>' +
            pred.label +
            '<span class="risk-badge ' + riskClass + '">' + (pred.risk || "Benign") + '</span>' +
            '</div>' +
            '<div class="confidence-bar-wrap">' +
            '<div class="confidence-bar"><div class="confidence-fill" style="width:' + barWidth + '%;background:' + fillColor + ';"></div></div>' +
            '<span class="confidence-label" style="color:' + fillColor + '">' + score + '%</span>' +
            '</div>';

        bodyHTML =
            '<div class="result-section"><div class="result-section-label">ABOUT</div>' + (pred.description || "—") + '</div>' +
            '<div class="result-section"><div class="result-section-label">TREATMENT</div>' + (pred.prescription || "—") + '</div>' +
            '<div class="result-section"><div class="result-section-label">PRECAUTION</div>' + (pred.precaution || "—") + '</div>';
    }

    card.innerHTML =
        '<div class="result-card-header">' + headerHTML + '</div>' +
        '<div class="result-card-body">'   + bodyHTML   + '</div>';

    return card;
}

// ─────────────────────────────────────────────
// IMAGE UPLOAD SETUP
// ─────────────────────────────────────────────

function setupUpload(inputId, previewId, imgId, nameId, type) {
    const input   = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const imgEl   = document.getElementById(imgId);
    const nameEl  = document.getElementById(nameId);
    const inner   = document.getElementById(inputId.replace("File", "UploadInner"));

    input.addEventListener("change", e => {
        const file = e.target.files[0];
        if (!file) return;

        if (type === "skin") skinFile = file;
        else                 xrayFile = file;

        const reader = new FileReader();
        reader.onload = ev => {
            imgEl.src = ev.target.result;
            nameEl.textContent = file.name.length > 20 ? file.name.substring(0, 17) + "..." : file.name;
            inner.classList.add("hidden");
            preview.classList.remove("hidden");
            analyzeImageBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    });
}

function removeUpload(type) {
    if (type === "skin") {
        skinFile = null;
        document.getElementById("skinFile").value = "";
        document.getElementById("skinPreview").classList.add("hidden");
        document.getElementById("skinUploadInner").classList.remove("hidden");
    } else {
        xrayFile = null;
        document.getElementById("xrayFile").value = "";
        document.getElementById("xrayPreview").classList.add("hidden");
        document.getElementById("xrayUploadInner").classList.remove("hidden");
    }
    analyzeImageBtn.disabled = !skinFile && !xrayFile;
}

// ─────────────────────────────────────────────
// ANALYZE IMAGE
// ─────────────────────────────────────────────

analyzeImageBtn.addEventListener("click", async () => {
    if (!skinFile && !xrayFile) return;

    if (skinFile) {
        appendUserMessage("Uploaded skin image for analysis.");
        await analyzeSkin(skinFile);
    }

    if (xrayFile) {
        appendUserMessage("Uploaded X-Ray for analysis.");
        appendBotMessage("X-Ray deep learning analysis requires a trained chest X-Ray model. Please run the training script for the radiology model to enable this feature.");
    }
});

async function analyzeSkin(file) {
    const loadingId = showTyping();

    try {
        const formData = new FormData();
        formData.append("image", file);

        const res = await fetch(`${API_BASE}/api/skin`, {
            method: "POST",
            body: formData
        });

        removeTyping(loadingId);

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        if (data.error) {
            appendBotMessage(`⚠️ ${data.error}\n\nRun \`train_skin_model.py\` to train the skin model first.`);
            return;
        }

        const { predictions } = data;
        if (!predictions || predictions.length === 0) {
            appendBotMessage("The skin analysis did not return confident predictions. Please use a clear, well-lit close-up photo of the affected area.");
            return;
        }

        const container = document.createElement("div");
        container.style.cssText = "display:flex;flex-direction:column;gap:8px;width:100%;max-width:680px;";

        const hdr = document.createElement("div");
        hdr.className = "results-header";
        hdr.innerHTML = `<i class="fa-solid fa-microscope"></i> SKIN CONDITION ANALYSIS — TOP ${predictions.length} RESULTS`;
        container.appendChild(hdr);

        predictions.forEach((pred, idx) => {
            container.appendChild(buildResultCard(pred, idx + 1, "skin"));
        });

        const disclaimer = document.createElement("div");
        disclaimer.style.cssText = "font-size:11px;color:var(--text-3);padding:6px 10px;border-left:2px solid var(--border);";
        disclaimer.textContent = "⚕️ Skin AI analysis is not a diagnosis. Consult a licensed dermatologist for proper evaluation.";
        container.appendChild(disclaimer);

        appendBotElement(container);

    } catch (err) {
        removeTyping(loadingId);
        appendBotMessage(`❌ Skin analysis failed: ${err.message}\n\nEnsure the backend is running at ${API_BASE}`);
    }
}

// ─────────────────────────────────────────────
// VOICE INPUT
// ─────────────────────────────────────────────

function setupVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        micBtn.style.display = "none";
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = e => {
        const transcript = Array.from(e.results).map(r => r[0].transcript).join("");
        chatInput.value = transcript;
    };

    recognition.onend = () => {
        isListening = false;
        micBtn.classList.remove("listening");
        micBtn.innerHTML = `<i class="fa-solid fa-microphone"></i>`;
    };

    recognition.onerror = () => {
        isListening = false;
        micBtn.classList.remove("listening");
        micBtn.innerHTML = `<i class="fa-solid fa-microphone"></i>`;
    };

    micBtn.addEventListener("click", () => {
        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
            isListening = true;
            micBtn.classList.add("listening");
            micBtn.innerHTML = `<i class="fa-solid fa-stop"></i>`;
        }
    });
}

// ─────────────────────────────────────────────
// HISTORY
// ─────────────────────────────────────────────

function saveHistory(symptoms, text, data) {
    const entry = {
        id: Date.now(),
        time: new Date().toLocaleString(),
        symptoms: symptoms,
        text: text,
        top: data.predictions?.[0]?.disease || "Unknown"
    };
    history.unshift(entry);
    if (history.length > 20) history = history.slice(0, 20);
    localStorage.setItem("dr_ai_history", JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    if (!historyList) return;
    historyList.innerHTML = "";

    if (history.length === 0) {
        historyList.innerHTML = `<div class="history-empty"><i class="fa-regular fa-folder-open"></i><p>No previous sessions</p></div>`;
        return;
    }

    history.forEach(entry => {
        const item = document.createElement("div");
        item.className = "history-item";
        item.innerHTML = `
            <div class="history-item-title">${entry.top}</div>
            <div class="history-item-time">${entry.time} · ${entry.symptoms.length} symptoms</div>
        `;
        historyList.appendChild(item);
    });
}

document.getElementById("clearHistoryBtn")?.addEventListener("click", () => {
    history = [];
    localStorage.setItem("dr_ai_history", "[]");
    renderHistory();
});

// ─────────────────────────────────────────────
// CHAT HELPERS
// ─────────────────────────────────────────────

function appendUserMessage(text) {
    const time = now();
    const div = document.createElement("div");
    div.className = "msg user";
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-user"></i></div>
        <div class="msg-body">
            <div class="msg-bubble">${escapeHtml(text)}</div>
            <span class="msg-time">${time}</span>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function appendBotMessage(text) {
    const time = now();
    const div = document.createElement("div");
    div.className = "msg bot";
    const formatted = formatMarkdown(text);
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-user-doctor"></i></div>
        <div class="msg-body">
            <div class="msg-bubble">${formatted}</div>
            <span class="msg-time">${time}</span>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function appendBotElement(element) {
    const time = now();
    const wrapper = document.createElement("div");
    wrapper.className = "msg bot";
    wrapper.style.maxWidth = "100%";

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = `<i class="fa-solid fa-user-doctor"></i>`;

    const body = document.createElement("div");
    body.className = "msg-body";
    body.style.maxWidth = "100%";
    body.appendChild(element);

    const timeEl = document.createElement("span");
    timeEl.className = "msg-time";
    timeEl.textContent = time;
    body.appendChild(timeEl);

    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

let typingCounter = 0;

function showTyping() {
    const id = "typing-" + (++typingCounter);
    const div = document.createElement("div");
    div.className = "msg bot";
    div.id = id;
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-user-doctor"></i></div>
        <div class="msg-body">
            <div class="msg-bubble">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    document.getElementById(id)?.remove();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function now() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function capitalize(s) {
    // Handles snake_case symptom names from the trained model
    return s.replace(/_/g, ' ').replace(/\w/g, c => c.toUpperCase());
}

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatMarkdown(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
        .replace(/•/g, "•")
        .replace(/\n/g, "<br>");
}

function clearChat() {
    chatMessages.innerHTML = "";
    showWelcome();
}
