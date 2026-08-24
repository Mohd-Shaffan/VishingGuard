"""
=============================================================================
 SCAM GUARD - ENHANCED HYBRID AI ENGINE v3.0
 Author: Mohd Shaffan
 Manipal University Jaipur

 FIXES in this revision:
  * classify_intent() now accepts (text, nlp_model, model_name) -- 3 params
    matching every call-site in main.py and the experiment suite.
  * classify_intent() now returns (is_scam, confidence, details) -- 3-tuple.
  * load_nlp_model() uses the correct SentenceTransformer model name
    ('distilbert-base-nli-mean-tokens' -- 768-dim, same as training).
  * Model dict {"encoder", "classifier"} is consumed correctly inside
    classify_intent instead of being called like a HuggingFace pipeline.
  * REAL_TEST_SET is loaded lazily (only when experiments are requested)
    so that importing the module from main.py does not crash or block.
=============================================================================
"""

import time
import math
import random
import warnings
import sys
import re
import os
import numpy as np

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

# ??? DEPENDENCY CHECK ????????????????????????????????????????????????????????
import joblib
from sentence_transformers import SentenceTransformer

try:
    import whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False  # ASR disabled -- text-only mode

try:
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  precision_recall_fscore_support)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")           # non-GUI backend for headless runs
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# =============================================================================
#  SECTION 1 -- TIERED KEYWORD DICTIONARY  (India-centric + Global)
# =============================================================================
KEYWORD_TIERS = {
    "CRITICAL": {
        "weight": 3,
        "words": [
            "otp", "cvv", "pin", "aadhaar", "pan", "password", "passphrase",
            "wire transfer", "cryptocurrency", "gift card", "remote access",
            "anydesk", "teamviewer", "verify your identity", "confirm your otp",
            "share your otp", "give me your otp"
        ]
    },
    "HIGH": {
        "weight": 2,
        "words": [
            "blocked", "suspended", "arrest", "legal action", "fir", "police",
            "court", "emi", "refund", "lottery", "prize", "won", "reward",
            "kyc", "update kyc", "bank account", "sbi", "hdfc", "icici",
            "axis bank", "paytm", "gpay", "phonepay", "upi", "neft", "rtgs",
            "debit card", "credit card", "atm card", "income tax",
            "customs", "parcel", "package", "delivery charge", "clearance fee"
        ]
    },
    "MEDIUM": {
        "weight": 1,
        "words": [
            "urgent", "immediately", "right now", "last chance", "expire",
            "limited time", "act fast", "do not tell", "keep secret",
            "do not hang up", "stay on the line", "important notice",
            "government", "rbi", "trai", "insurance", "policy", "claim",
            "emi waiver", "loan approval", "interest rate", "outstanding"
        ]
    }
}

# Flatten for quick lookup
KEYWORD_WEIGHT_MAP: dict[str, int] = {}
for _tier, _data in KEYWORD_TIERS.items():
    for _word in _data["words"]:
        KEYWORD_WEIGHT_MAP[_word] = _data["weight"]


# =============================================================================
#  SECTION 2 -- NLP MODEL LOADING
# =============================================================================

def load_nlp_model() -> tuple:
    """
    Load the SentenceTransformer encoder and the trained Logistic Regression
    classifier from disk.

    Returns:
        (nlp_model_dict, model_name_str)
        nlp_model_dict = {"encoder": SentenceTransformer, "classifier": LogisticRegression}
        Falls back to (None, "keyword-only") on any error.

    IMPORTANT: The SentenceTransformer model name MUST match the model used
    during training (train_vishing_model.py).  Both use 768-dim embeddings.
    'distilbert-base-nli-mean-tokens' is the correct sentence-level model;
    'distilbert-base-uncased' is a raw tokeniser and will NOT work here.
    """
    print("  -> Loading SentenceTransformer + Logistic Regression...")
    try:
        # ?? FIX: use the same model as training ?????????????????????????????
        # train_scam_model.py used 'distilbert-base-nli-mean-tokens' (768-dim).
        # If your vishing model was trained with a different encoder, swap here.
        encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')

        # Locate the .pkl relative to this file so it works on any CWD
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'logistic_vishing_model.pkl')
        clf = joblib.load(pkl_path)
        print("  [OK] Vishing-specific model loaded.")
        return {"encoder": encoder, "classifier": clf}, "DistilBERT+LR(Vishing)"
    except FileNotFoundError:
        print("  [ERR] logistic_vishing_model.pkl not found -- falling back to keyword-only.")
        return None, "keyword-only"
    except Exception as exc:
        print(f"  [ERR] Model load error: {exc}")
        return None, "keyword-only"


# =============================================================================
#  SECTION 3 -- KEYWORD SCORER
# =============================================================================

def compute_keyword_score(text: str) -> tuple[float, int, list]:
    """
    Returns (raw_weight_sum, hit_count, matched_keywords).
    matched_keywords is a list of (keyword_str, weight_int) tuples.
    """
    text_lower = text.lower()
    matched = []
    total_weight = 0
    for keyword, weight in KEYWORD_WEIGHT_MAP.items():
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, text_lower):
            matched.append((keyword, weight))
            total_weight += weight
    return total_weight, len(matched), matched


# =============================================================================
#  SECTION 4 -- HYBRID INTENT CLASSIFIER  (Algorithm 1 of paper)
# =============================================================================

def classify_intent(text: str,
                    nlp_model=None,
                    model_name: str = "keyword-only") -> tuple:
    """
    Hybrid classifier: DistilBERT sentence embedding + Logistic Regression
    fused with a tiered keyword heuristic (paper Algorithm 1).

    Parameters
    ----------
    text       : str  -- transcribed conversational chunk
    nlp_model  : dict | None
                 {"encoder": SentenceTransformer, "classifier": LogisticRegression}
                 as returned by load_nlp_model().  None -> keyword-only mode.
    model_name : str  -- informational label used in logs / response payloads

    Returns
    -------
    (is_scam: bool, confidence: float, details: dict)
        confidence  -- scam probability in [0, 1]
        details     -- keyword_weight, keyword_hits, matched_keywords,
                      nlp_label, nlp_raw_score
    """
    # ?? Step 1: Keyword analysis ?????????????????????????????????????????????
    kw_weight, kw_hits, matched_keywords = compute_keyword_score(text)

    # ?? Step 2: Negation detection ??????????????????????????????????????????
    text_lower = text.lower()
    negation_words = [
        "don't", "do not", "never", "shouldn't", "should not",
        "safe", "protect", "warning", "beware", "don't share", "do not share"
    ]
    is_negated = any(neg in text_lower for neg in negation_words)

    # ?? Step 3: Critical keyword flag ???????????????????????????????????????
    has_critical = any(weight == 3 for _, weight in matched_keywords)

    # ?? Step 4: NLP deep semantic score ?????????????????????????????????????
    nlp_score = 0.30        # safe-leaning default when model unavailable
    nlp_label = "N/A"

    if (nlp_model is not None
            and isinstance(nlp_model, dict)
            and "encoder" in nlp_model
            and "classifier" in nlp_model):
        try:
            encoder = nlp_model["encoder"]
            clf     = nlp_model["classifier"]
            # encode() returns a 2-D array; we need the single row
            embedding = encoder.encode([text])          # shape (1, 768)
            proba     = clf.predict_proba(embedding)[0] # shape (n_classes,)
            # class 1 = scam; class 0 = safe
            nlp_score = float(proba[1])
            nlp_label = "SCAM" if nlp_score >= 0.50 else "SAFE"
        except Exception as exc:
            nlp_label = f"ERR:{exc}"
            # nlp_score stays 0.5 -- keyword layer will carry the decision

    # ?? Step 5: Fusion logic (Equation 4 of paper) ??????????????????????????
    if has_critical and not is_negated:
        # Critical keyword gives a moderate boost; NLP must also be suspicious
        combined_score = nlp_score + 0.25   # reduced boost (was 0.35)
        combined_score = min(combined_score, 1.0)
        is_scam = combined_score >= 0.60
        final_score = combined_score

    elif has_critical and is_negated:
        # e.g. "Never share your OTP" -- protective statement
        final_score = max(0.05, nlp_score - 0.40)
        is_scam = False

    elif is_negated and kw_hits > 0:
        # Non-critical keywords in a negation context -> likely a warning/advice
        kw_normalised = min(1.0, kw_weight / 15.0)
        final_score   = 0.6 * nlp_score + 0.4 * kw_normalised * 0.3  # heavily dampen keywords
        is_scam       = final_score >= 0.60

    else:
        # Standard hybrid blend: 60% NLP + 40% normalised keyword weight
        kw_normalised = min(1.0, kw_weight / 15.0)
        final_score   = 0.6 * nlp_score + 0.4 * kw_normalised

        # When no keywords are matched, NLP must be very confident to flag as scam
        # This prevents false positives on normal imperative/request sentences
        if kw_hits == 0:
            is_scam = final_score >= 0.65
        else:
            is_scam = final_score >= 0.60

    details = {
        "keyword_weight":    kw_weight,
        "keyword_hits":      kw_hits,
        "matched_keywords":  matched_keywords,   # list of (word, weight)
        "nlp_label":         nlp_label,
        "nlp_raw_score":     nlp_score,
    }

    return is_scam, final_score, details


# =============================================================================
#  SECTION 5 -- TEMPORAL THREAT DECAY ALGORITHM  (Equation 5 of paper)
# =============================================================================

class TemporalThreatScorer:
    """
    CT = ??S_T + (1-?) ? ?_{i=1}^{k} S_{T-i} ? e^{-?i}

    ?      = weight of most-recent chunk (default 0.85)
    ?      = exponential decay rate       (default 0.3)
    k      = history window size          (default 10)
    ?_alert = amber-warning threshold    (default 0.55)
    ?_drop  = auto-drop threshold        (default 0.80)
    """

    def __init__(self, alpha=0.99, lambda_decay=0.3, history_k=10,
                 tau_alert=0.50, tau_drop=0.80):
        self.alpha     = alpha
        self.lam       = lambda_decay
        self.k         = history_k
        self.tau_alert = tau_alert
        self.tau_drop  = tau_drop
        self.history: list[float] = []   # newest first
        self.CT = 0.0

    def update(self, s_t: float) -> float:
        """Push a new chunk score and return the updated cumulative CT."""
        self.history.insert(0, s_t)
        if len(self.history) > self.k:
            self.history = self.history[:self.k]

        ct = self.alpha * s_t
        hist_sum = sum(
            score * math.exp(-self.lam * i)
            for i, score in enumerate(self.history[1:], start=1)
        )
        ct += (1.0 - self.alpha) * hist_sum
        self.CT = ct
        return ct

    def status(self) -> str:
        if self.CT >= self.tau_drop:
            return "DROP"
        elif self.CT >= self.tau_alert:
            return "ALERT"
        return "SAFE"

    def reset(self):
        self.history.clear()
        self.CT = 0.0


# =============================================================================
#  SECTION 6 -- WHISPER ASR  (Module 2 of paper)
# =============================================================================

def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper (base model)."""
    if not WHISPER_OK:
        print("[ASR] Whisper not installed. pip install openai-whisper")
        return ""
    print(f"[ASR] Transcribing: {audio_path}")
    model  = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


# =============================================================================
#  SECTION 7 -- INTERACTIVE CLI
# =============================================================================

def run_interactive_mode(nlp_model, model_name):
    print("\n" + "="*65)
    print("  [SHIELD]  SCAM GUARD v2.0  --  HYBRID AI + TEMPORAL SCORING  [SHIELD]")
    print("="*65)
    print(f"  NLP Model : {model_name}")
    print(f"  Commands  : 'exit' to quit | 'reset' to start new call")
    print("="*65)

    scorer    = TemporalThreatScorer()
    chunk_num = 0

    while True:
        try:
            user_input = input(f"\n[Chunk {chunk_num+1}] Enter sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            scorer.reset(); chunk_num = 0
            print("  [RESET]  New call session started.")
            continue
        if not user_input:
            continue

        chunk_num += 1
        t0 = time.perf_counter()
        is_scam, confidence, details = classify_intent(user_input, nlp_model, model_name)
        latency_ms = (time.perf_counter() - t0) * 1000

        chunk_score = confidence
        CT     = scorer.update(chunk_score)
        status = scorer.status()

        print(f"\n  {'?'*55}")
        print(f"  Chunk score  : {chunk_score:.4f}  |  Cumulative CT : {CT:.4f}")
        print(f"  Keywords     : {details['keyword_weight']} pts  ->  "
              f"{[k for k,_ in details['matched_keywords']]}")
        print(f"  NLP label    : {details['nlp_label']} ({details['nlp_raw_score']:.2%})")
        print(f"  Latency      : {latency_ms:.1f} ms")
        print(f"  {'?'*55}")

        if status == "DROP":
            print(f"  [ALERT]  AUTO-DROP: MALICIOUS CALL (CT={CT:.2f})")
        elif status == "ALERT":
            print(f"  [RED]  ALERT: SCAM DETECTED  (Confidence: {confidence:.2%})")
        else:
            print(f"  [OK]  SAFE  (Confidence: {confidence:.2%})")

    print("\n  Session ended. Stay safe! [SHIELD]")


# =============================================================================
#  SECTION 8 -- EXPERIMENT SUITE
# =============================================================================

def load_vishing_test_data(filepath='vishing_data.csv', sample_size=200):
    """Load a balanced test sample from the vishing dataset."""
    if not PANDAS_OK:
        print("  [ERROR] pandas not installed.")
        return []
    try:
        df   = pd.read_csv(filepath)
        scam = df[df['label'] == 1].sample(min(sample_size // 2, len(df[df['label'] == 1])))
        safe = df[df['label'] == 0].sample(sample_size - len(scam))
        test_df = pd.concat([scam, safe]).sample(frac=1)
        return [(row['text'], row['label']) for _, row in test_df.iterrows()]
    except Exception as exc:
        print(f"  [ERROR] Could not load {filepath}: {exc}")
        return []


# REAL_TEST_SET is loaded lazily in run_all_experiments() so that simply
# importing this module (from main.py) never causes a crash or slow start-up.
REAL_TEST_SET: list = []


def _ensure_test_data():
    global REAL_TEST_SET
    if not REAL_TEST_SET:
        REAL_TEST_SET = load_vishing_test_data(sample_size=200)


def run_classification_experiment(nlp_model, model_name):
    _ensure_test_data()
    print("\n" + "="*65)
    print("  EXPERIMENT 1 -- Classification Metrics (Real Data)")
    print("="*65)

    y_true, y_pred, latencies = [], [], []
    skipped = 0
    for text, label in REAL_TEST_SET:
        try:
            t0 = time.perf_counter()
            is_scam, _, _ = classify_intent(text, nlp_model, model_name)
            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(label)
            y_pred.append(1 if is_scam else 0)
        except Exception as exc:
            print(f"  [WARN] Skipping sample: {exc}")
            skipped += 1

    if skipped:
        print(f"  [INFO] Skipped {skipped} samples.")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = (tp + tn) / len(y_true) if y_true else 0

    print(f"\n  Confusion Matrix:")
    print(f"  {'':10s}  Predicted SCAM  Predicted SAFE")
    print(f"  {'Actual SCAM':10s}  TP={tp:3d}            FN={fn:3d}")
    print(f"  {'Actual SAFE':10s}  FP={fp:3d}            TN={tn:3d}")
    print(f"\n  Precision : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    return latencies, y_true, y_pred


def run_latency_experiment(nlp_model, model_name, n_runs=50):
    _ensure_test_data()
    print("\n" + "="*65)
    print("  EXPERIMENT 2 -- Latency Profiling")
    print("="*65)

    test_texts = [t for t, _ in REAL_TEST_SET]
    if not test_texts:
        return []

    latencies = []
    for _ in range(n_runs):
        text = random.choice(test_texts)
        try:
            t0 = time.perf_counter()
            classify_intent(text, nlp_model, model_name)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            print(f"  [WARN] Latency run failed: {exc}")

    print(f"\n  Runs : {n_runs}")
    print(f"  Mean : {np.mean(latencies):.2f} ms")
    print(f"  Max  : {np.max(latencies):.2f} ms")
    print(f"  P99  : {np.percentile(latencies, 99):.2f} ms")
    return latencies


def run_noise_robustness_experiment(nlp_model, model_name):
    _ensure_test_data()
    print("\n" + "="*65)
    print("  EXPERIMENT 3 -- Noise Robustness (Real Data)")
    print("="*65)

    scam_samples = [t for t, l in REAL_TEST_SET if l == 1][:30]
    if not scam_samples:
        return

    def corrupt(text, level=0.10):
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < level:
                chars[i] = ("" if random.choice(["drop", "swap"]) == "drop"
                            else random.choice("abcdefghijklmnopqrstuvwxyz "))
        return "".join(chars)

    print(f"\n  {'Noise Level':15s}  {'Recall':10s}")
    print(f"  {'-'*30}")
    for noise in [0.0, 0.05, 0.10, 0.15, 0.20]:
        correct = sum(
            1 for text in scam_samples
            if classify_intent(corrupt(text, noise) if noise > 0 else text,
                               nlp_model, model_name)[0]
        )
        print(f"  {noise*100:>5.0f}%          {correct/len(scam_samples)*100:>6.1f}%")


def run_threshold_sensitivity(nlp_model, model_name):
    _ensure_test_data()
    print("\n" + "="*65)
    print("  EXPERIMENT 4 -- Threshold Sensitivity")
    print("="*65)

    safe_msgs = [text for text, label in REAL_TEST_SET if label == 0][:2]
    scam_msgs = [text for text, label in REAL_TEST_SET if label == 1][:8]
    if not safe_msgs or len(scam_msgs) < 8:
        print("  [ERROR] Not enough data.")
        return

    call_chunks = [(msg, 0) for msg in safe_msgs] + [(msg, 1) for msg in scam_msgs]
    tau_pairs   = [(0.4, 0.7), (0.5, 0.75), (0.55, 0.80), (0.6, 0.85)]

    print(f"\n  {'?_alert':10s} {'?_drop':10s} {'Alert @chunk':15s} {'Drop @chunk':12s}")
    print(f"  {'-'*50}")

    for tau_a, tau_d in tau_pairs:
        scorer = TemporalThreatScorer(tau_alert=tau_a, tau_drop=tau_d)
        alert_at = drop_at = None
        for i, (text, _) in enumerate(call_chunks, 1):
            is_scam, conf, _ = classify_intent(text, nlp_model, model_name)
            chunk_score = conf
            CT = scorer.update(chunk_score)
            if drop_at  is None and CT >= tau_d: drop_at  = i
            if alert_at is None and CT >= tau_a: alert_at = i

        print(f"  {tau_a:<10.2f} {tau_d:<10.2f} "
              f"{'chunk ' + str(alert_at) if alert_at else 'never':<15s} "
              f"{'chunk ' + str(drop_at)  if drop_at  else 'never':<12s}")


def run_all_experiments(nlp_model, model_name):
    _ensure_test_data()
    if not REAL_TEST_SET:
        print("\n  [ERROR] Cannot run experiments without data.")
        return

    latencies, y_true, y_pred = run_classification_experiment(nlp_model, model_name)
    lat_bench = run_latency_experiment(nlp_model, model_name)
    run_noise_robustness_experiment(nlp_model, model_name)
    run_threshold_sensitivity(nlp_model, model_name)

    if MATPLOTLIB_OK:
        # Confusion Matrix
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        cm = np.array([[tp, fn], [fp, tn]])
        ax1.imshow(cm, cmap='gray', vmin=0, vmax=np.max(cm))
        ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Pred Scam', 'Pred Safe'])
        ax1.set_yticklabels(['Act Scam', 'Act Safe'])
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=14, fontweight='bold', color=color)
        ax1.set_title('Confusion Matrix')
        plt.tight_layout()
        fig1.savefig('fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("  [CHART] Confusion matrix -> fig_confusion_matrix.png")

        # Latency Histogram
        if lat_bench:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(lat_bench, bins=15, color='gray', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(lat_bench), color='black', linestyle='--',
                        linewidth=2, label=f'Mean={np.mean(lat_bench):.1f} ms')
            ax2.set_xlabel('Latency (ms)'); ax2.set_ylabel('Frequency')
            ax2.set_title('NLP Inference Latency Distribution')
            ax2.legend(); ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            fig2.savefig('fig_latency_histogram.png', dpi=300, bbox_inches='tight')
            print("  [CHART] Latency histogram -> fig_latency_histogram.png")

        # Temporal Scoring Case Study
        scorer       = TemporalThreatScorer()
        chunk_scores = [0.15, 0.35, 0.82, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65]
        ct_values    = [scorer.update(s) for s in chunk_scores]
        fig3, ax3    = plt.subplots(figsize=(6, 4))
        ax3.plot(range(1, len(ct_values)+1), ct_values, marker='o',
                 color='black', linewidth=2, markersize=6, label='$C_T$')
        ax3.axhline(y=0.55, color='gray',     linestyle='--', linewidth=1.5, label='$\\tau_{alert}$=0.55')
        ax3.axhline(y=0.80, color='darkgray', linestyle='--', linewidth=1.5, label='$\\tau_{drop}$=0.80')
        ax3.set_xlabel('Chunk Number')
        ax3.set_ylabel('Cumulative Threat Score ($C_T$)')
        ax3.set_title('Temporal Threat Score Progression')
        ax3.legend(); ax3.grid(alpha=0.3); ax3.set_ylim(0, 1.05)
        plt.tight_layout()
        fig3.savefig('fig_temporal_scoring.png', dpi=300, bbox_inches='tight')
        print("  [CHART] Temporal scoring -> fig_temporal_scoring.png")
        plt.close('all')
    else:
        print("\n  [INFO] Install matplotlib for plots: pip install matplotlib")


# =============================================================================
#  SECTION 9 -- AUDIO FILE MODE
# =============================================================================

def run_audio_mode(audio_path: str, nlp_model, model_name):
    print(f"\n[AUDIO MODE] Processing: {audio_path}")
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return
    print(f"[TRANSCRIPT] {transcript}\n")

    sentences = re.split(r"[.!?,]\s*", transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    scorer = TemporalThreatScorer()
    for i, chunk in enumerate(sentences, 1):
        is_scam, conf, details = classify_intent(chunk, nlp_model, model_name)
        chunk_score = conf
        CT     = scorer.update(chunk_score)
        status = scorer.status()
        print(f"  Chunk {i:02d}: [{status}] CT={CT:.3f} | \"{chunk}\"")
        if status == "DROP":
            print("  [ALERT]  AUTO-DROP TRIGGERED.")
            break


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  [SHIELD]  SCAM GUARD -- Enhanced Hybrid AI Engine v2.0  [SHIELD]")
    print("="*65)
    print("\n  Loading NLP model...")
    nlp_model, model_name = load_nlp_model()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "experiments":
            run_all_experiments(nlp_model, model_name)
            return
        elif cmd == "audio" and len(sys.argv) > 2:
            run_audio_mode(sys.argv[2], nlp_model, model_name)
            return
        elif cmd == "help":
            print("\n  Usage:")
            print("    python scamguard_enhanced.py              # Interactive mode")
            print("    python scamguard_enhanced.py experiments  # Full paper experiments")
            print("    python scamguard_enhanced.py audio <path> # Audio transcription")
            return

    run_interactive_mode(nlp_model, model_name)


if __name__ == "__main__":
    main()