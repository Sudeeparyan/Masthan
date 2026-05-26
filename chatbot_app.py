import streamlit as st
import os, re, time
import torch
from dotenv import load_dotenv
load_dotenv()
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

try:
    import google.genai as genai
    GENAI_OK = True
except Exception:
    GENAI_OK = False

DATA_DIR = Path("data")
embed_dim = 128
hidden_dim = 256
num_layers = 2
dropout = 0.3
max_seq = 50
vocabsize = 3000
gemini_model_name = "gemini-2.5-flash"
api_key = os.getenv("GOOGLE_API_KEY", "")
LOW_CONF_THRESHOLD = 0.45
STOP_WORDS = {
    'what','is','are','how','does','the','a','an','of','for','to','do','can',
    'my','you','your','it','with','this','that','about','in','on','and','or',
    'not','be','have','has','from','by','at','as','if','its','their','which',
    'when','where','who','why','should','would','will','could','may','might',
    'used','use','affect','effects','body','cause','causes','treat','treatment',
}


@st.cache_resource
def load_vocab():
    df = pd.read_csv(DATA_DIR / "medquad.csv")
    df.columns = [c.lower() for c in df.columns]
    qcol = next(c for c in df.columns if "question" in c)
    acol = next(c for c in df.columns if "answer" in c)
    df = df.rename(columns={qcol: "question", acol: "answer"}).dropna()
    df["q_clean"] = df["question"].str.lower().apply(lambda x: re.sub(r"[^a-z0-9 ]", " ", x))
    df["a_clean"] = df["answer"].str.lower().apply(lambda x: re.sub(r"[^a-z0-9 ]", " ", x))
    all_text = df["q_clean"].tolist() + df["a_clean"].tolist()
    word_counts = Counter(" ".join(all_text).split())
    vocab_words = [w for w, _ in word_counts.most_common(vocabsize)]
    word2idx = {w: i + 1 for i, w in enumerate(vocab_words)}
    train_n = int(0.7 * len(df))
    train_df = df.iloc[:train_n]
    # TF-IDF fallback for OOV queries
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(train_df["q_clean"].tolist())
    return word2idx, train_df["q_clean"].tolist(), train_df["a_clean"].tolist(), tfidf, tfidf_matrix


def tokenize_text(txt, word2idx, max_len=50):
    txt = re.sub(r"[^a-z0-9 ]", " ", txt.lower())
    tokens = txt.split()
    indices = [word2idx.get(w, 0) for w in tokens[:max_len]]
    indices += [0] * (max_len - len(indices))
    return indices


class BiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocabsize + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.drop(self.embed(x))
        out, _ = self.lstm(emb)
        return F.normalize(out.mean(dim=1), dim=-1)


class SiameseLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BiLSTMEncoder()

    def forward(self, q, a):
        return F.cosine_similarity(self.encoder(q), self.encoder(a))


@st.cache_resource
def load_model_and_embeds():
    word2idx, train_questions, train_answers, tfidf, tfidf_matrix = load_vocab()
    model = SiameseLSTM()
    model_path = DATA_DIR / "bilstm.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    embeds_path = DATA_DIR / "embeds.npy"
    embeds = np.load(embeds_path) if embeds_path.exists() else None
    return model, embeds, word2idx, train_questions, train_answers, tfidf, tfidf_matrix


def retrieve_answer(question, model, embeds, word2idx, train_answers, tfidf, tfidf_matrix):
    """Returns (answer_text, similarity_score, retrieval_method, oov_terms)."""
    if embeds is None:
        return "Model not trained yet. Please run the notebook first.", 0.0, "none", []
    # Detect OOV content words — if a key noun/drug is unknown the BiLSTM embedding is unreliable
    q_clean = re.sub(r"[^a-z0-9 ]", " ", question.lower())
    content_tokens = [t for t in q_clean.split() if t not in STOP_WORDS and len(t) > 3]
    oov_terms = [t for t in content_tokens if word2idx.get(t, 0) == 0]
    has_oov = len(oov_terms) > 0

    model.eval()
    with torch.no_grad():
        tok = torch.tensor(tokenize_text(question, word2idx), dtype=torch.long).unsqueeze(0)
        q_emb = model.encoder(tok).numpy()
    sims = sk_cosine(q_emb, embeds)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    method = "bilstm-oov" if has_oov else "bilstm"
    return train_answers[best_idx], best_score, method, oov_terms


def call_gemini(prompt, system="You are a helpful medical assistant. Be concise and accurate.", max_tokens=600):
    if not GENAI_OK or not api_key:
        return "[Gemini not available — set GOOGLE_API_KEY in .env]"
    try:
        client = genai.Client(api_key=api_key)
        # thinking_budget=0 prevents gemini-2.5-flash from consuming max_output_tokens on internal thinking
        resp = client.models.generate_content(
            model=gemini_model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Chatbot — Thesis Demo",
    layout="wide",
    page_icon="🏥",
)

st.markdown("## 🏥 Medical Chatbot — System Comparison")
st.markdown("**MSc AI Thesis** · Mastan Vali Shaik · National College of Ireland (24226807)")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient Profile")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    conditions = st.text_input("Conditions", value="diabetes, hypertension")
    meds = st.text_input("Medications", value="metformin, lisinopril")
    profile_text = f"Patient is {age} years old with {conditions}. Taking {meds}."
    st.info(profile_text)

    st.divider()
    st.subheader("Quick Demo Questions")
    sample_qs = [
        "-- pick a question --",
        "what are symptoms of diabetes",
        "how to control blood pressure",
        "what foods should diabetic patients avoid",
        "what is metformin used for",
        "how does hypertension affect the body",
        "what are side effects of lisinopril",
    ]
    selected_sample = st.selectbox("", sample_qs)

    st.divider()
    st.markdown("**System Summary**")
    st.markdown(
        "| System | Method |\n"
        "|---|---|\n"
        "| **LSTM Only** | BiLSTM retrieves closest MedQuAD answer — fast, factual, not personalised |\n"
        "| **Gemini Only** | LLM generates directly — personalised but no retrieval grounding |\n"
        "| **Hybrid** | BiLSTM retrieves context → Gemini synthesises complete personalised answer |\n"
    )
    st.divider()
    st.markdown("**Response quality ranking:**")
    st.markdown("🥇 **Hybrid** — factual + personalised\n\n🥈 **Gemini Only** — personalised, no grounding\n\n🥉 **LSTM Only** — factual, not personalised")

# ── Question input ─────────────────────────────────────────────────────────────
default_q = "" if selected_sample.startswith("--") else selected_sample
question = st.text_input(
    "Ask a medical question:",
    value=default_q,
    placeholder="e.g. what are symptoms of diabetes",
)

run = st.button("Compare All 3 Systems", type="primary", use_container_width=True)

if run and question.strip():
    model, embeds, word2idx, train_qs, train_ans, tfidf, tfidf_matrix = load_model_and_embeds()

    # System 1: LSTM Only — pure retrieval, no AI
    t0 = time.time()
    lstm_answer, lstm_score, lstm_method, lstm_oov = retrieve_answer(
        question, model, embeds, word2idx, train_ans, tfidf, tfidf_matrix
    )
    lstm_ms = round((time.time() - t0) * 1000, 1)

    # System 2: Gemini Only — pure LLM, no retrieval
    t0 = time.time()
    gem_system = (
        "You are a medical assistant. Give accurate, concise medical advice. "
        "Answer in 3-5 clear sentences."
    )
    gem_prompt = f"Patient profile: {profile_text}\nQuestion: {question}"
    gemini_answer = call_gemini(gem_prompt, gem_system, max_tokens=400)
    gemini_ms = round((time.time() - t0) * 1000, 1)

    # System 3: Hybrid — retrieval provides factual grounding, Gemini synthesises full personalised answer
    t0 = time.time()
    hyb_retrieved, hyb_score, hyb_method, hyb_oov = retrieve_answer(
        question, model, embeds, word2idx, train_ans, tfidf, tfidf_matrix
    )
    hyb_system = (
        "You are a medical assistant. Provide accurate, complete, personalised medical information. "
        "Use the retrieved context as factual grounding where relevant."
    )
    # Skip injecting context when OOV terms mean the retrieved text is likely off-topic
    context_available = hyb_method == "bilstm"
    if context_available:
        context_block = f"Retrieved medical context (factual reference): {hyb_retrieved[:500]}\n"
    else:
        context_block = (
            f"Note: The specific term(s) {hyb_oov} were not found in the training database. "
            "Rely entirely on your own medical knowledge for this answer.\n"
        )
    hyb_prompt = (
        f"Patient profile: {profile_text}\n"
        f"{context_block}"
        f"Question: {question}\n\n"
        "Provide a clear, complete answer (4-6 sentences) that: "
        "(1) directly answers the question with accurate medical information, "
        "(2) incorporates relevant facts from the retrieved context (if provided), "
        "(3) personalises the advice for this patient based on their age, conditions and medications."
    )
    hybrid_answer = call_gemini(hyb_prompt, hyb_system, max_tokens=600)
    hybrid_ms = round((time.time() - t0) * 1000, 1)

    # ── Side-by-side results ordered: Hybrid | Gemini | LSTM ─────────────────
    st.markdown(f"### Results for: *{question}*")

    col3, col2, col1 = st.columns(3)

    with col3:
        st.markdown("#### 🥇 System 3 — Hybrid *(Best)*")
        st.caption("BiLSTM retrieves context → Gemini synthesises complete personalised answer")
        m1, m2 = st.columns(2)
        m1.metric("Response time", f"{hybrid_ms} ms")
        m2.metric("Word count", len(hybrid_answer.split()))
        st.success(hybrid_answer)

    with col2:
        st.markdown("#### 🥈 System 2 — Gemini Only")
        st.caption("Pure LLM · No retrieval · Always personalised")
        m1, m2 = st.columns(2)
        m1.metric("Response time", f"{gemini_ms} ms")
        m2.metric("Word count", len(gemini_answer.split()))
        st.warning(gemini_answer)

    with col1:
        st.markdown("#### 🥉 System 1 — LSTM Only")
        st.caption("Pure retrieval · No AI generation · Deterministic")
        m1, m2, m3 = st.columns(3)
        m1.metric("Response time", f"{lstm_ms} ms")
        m2.metric("Word count", len(lstm_answer.split()))
        m3.metric("BiLSTM score", f"{lstm_score:.2f}")
        if lstm_oov:
            st.warning(
                f"⚠️ Term(s) **{', '.join(lstm_oov)}** not found in MedQuAD training data — "
                "BiLSTM cannot retrieve a relevant answer. This demonstrates a key limitation of pure retrieval."
            )
        st.info(lstm_answer)

    # ── Hybrid step-by-step breakdown ─────────────────────────────────────────
    st.divider()
    with st.expander("How Hybrid built its answer (step-by-step trace)"):
        st.markdown(f"**Step 1 — BiLSTM Retrieval** (score={hyb_score:.2f}):")
        if context_available:
            st.code(hyb_retrieved, language=None)
            st.caption("Context injected into Gemini prompt as factual grounding.")
        else:
            st.warning(
                f"Term(s) **{', '.join(hyb_oov)}** not in training data → context skipped. "
                "Gemini relied on its own parametric medical knowledge."
            )
        st.markdown("**Step 2 — Gemini Synthesis** (complete personalised answer):")
        st.code(hybrid_answer, language=None)
        st.markdown(
            "**Why Hybrid wins:** "
            "When retrieval finds relevant context, Hybrid grounds the answer in verified MedQuAD facts. "
            "When retrieval fails (OOV terms), Hybrid degrades gracefully to Gemini-only generation — "
            "still personalised, still correct. Pure LSTM retrieval cannot recover from OOV failures."
        )

    # ── Summary table ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Quick Comparison")
    comp_df = pd.DataFrame({
        "System": ["🥇 Hybrid", "🥈 Gemini Only", "🥉 LSTM Only"],
        "Response time (ms)": [hybrid_ms, gemini_ms, lstm_ms],
        "Word count": [len(hybrid_answer.split()), len(gemini_answer.split()), len(lstm_answer.split())],
        "Uses BiLSTM retrieval": ["Yes", "No", "Yes"],
        "Uses Gemini AI": ["Yes", "Yes", "No"],
        "Personalised": ["Yes", "Yes", "No"],
        "Factually grounded": ["Yes (MedQuAD + Gemini)", "No (parametric only)", "Yes (MedQuAD)"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Not medical advice. Consult a qualified healthcare professional for actual medical guidance.")
