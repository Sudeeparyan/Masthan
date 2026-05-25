# Code Explanation: Medical Chatbot Thesis
### `mathan_thesis_code/medical_chatbot.py`

**MSc Artificial Intelligence — National College of Ireland**  
**Student: Mastan Vali Shaik | ID: 24226807**

---

## Overview

This single Python file is your complete thesis experiment. It builds and compares **three AI systems** on the same medical question-answering task, then automatically generates publication-quality charts and a written report.

**Research hypothesis:** `Hybrid (LSTM + Gemini) > Gemini Only > LSTM Only`

The full pipeline runs end-to-end in one command: `python medical_chatbot.py`

---

## Architecture Diagram

```
                    ┌────────────────────────────────────────┐
                    │         MedQuAD Dataset (3,000 Q&A)    │
                    └────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                    │
               Train (70%)        Val (15%)           Test (15%)
                    │                   │
        ┌───────────▼───────────────────▼────────────┐
        │     Vocabulary + SiameseDataset             │
        │     (positive pairs + negative pairs)       │
        └───────────────────┬─────────────────────────┘
                            │
                ┌───────────▼─────────────┐
                │  BiLSTM Siamese Network  │
                │  Embedding(5000,128)     │
                │  → BiLSTM(128→256×2)    │
                │  → mean pool → 512-d    │
                │  → cosine similarity    │
                └───────────┬─────────────┘
                            │ trained model
              ┌─────────────┼─────────────┐
              │             │             │
    ┌─────────▼───┐  ┌──────▼──────┐  ┌──▼──────────────────┐
    │ LSTM Only   │  │ Gemini Only │  │ Hybrid               │
    │ (retrieval) │  │ (LLM only)  │  │ (retrieval + LLM)    │
    └─────────────┘  └─────────────┘  └──────────────────────┘
              │             │             │
              └─────────────┴─────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Evaluation Engine          │
              │  ROUGE, BLEU, Person.,      │
              │  Completeness, Composite    │
              │  + LLM-as-Judge             │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  7 Figures + 3 Report files │
              └────────────────────────────┘
```

---

## Section-by-Section Breakdown

---

### Section 1: Imports & Configuration (Lines 29–131)

This section sets up all constants and global settings.

**Key hyperparameters:**

| Constant | Value | Purpose |
|---|---|---|
| `VOCAB_MAX` | 5,000 | Maximum vocabulary size for tokeniser |
| `EMBED_DIM` | 128 | Word embedding dimension |
| `HIDDEN_DIM` | 256 | LSTM hidden units per direction |
| `NUM_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.3 | Dropout rate for regularisation |
| `MAX_SEQ_LEN` | 50 | Tokens kept per question/answer |
| `EPOCHS` | 15 | Training epochs |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `LR` | 0.001 | Adam learning rate |
| `DATASET_SIZE` | 3,000 | Q&A pairs used from MedQuAD |
| `N_EVAL` | 50 | Test questions evaluated per system |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini API model used |

**File paths produced:**

```
mathan_thesis_code/
├── medical_data/
│   ├── medquad.csv          ← downloaded dataset cache
│   ├── lstm_model.pt        ← trained BiLSTM weights
│   ├── vocab.pkl            ← serialised vocabulary
│   ├── train_embeddings.npy ← pre-computed question embeddings
│   └── loss_history.json    ← per-epoch losses
└── output/
    ├── evaluation_results.json
    ├── evaluation_report.txt
    ├── sample_predictions.csv
    └── figures/
        ├── 1_training_loss.png
        ├── 2_rouge_comparison.png
        ├── 3_bleu_comparison.png
        ├── 4_response_time.png
        ├── 5_combined_metrics.png
        ├── 6_sample_responses.png
        └── 7_clinical_quality.png
```

**Why `matplotlib.use("Agg")`?**  
This forces the headless backend so the script runs on a server or in a terminal without a display — figures are saved to files rather than shown interactively.

**Reproducibility:** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` — ensures the same train/val/test split and negative pair sampling every run.

---

### Section 2: Dataset Download & Load (Lines 136–291)

#### `download_dataset()`
Downloads the **MedQuAD** dataset from HuggingFace on the first run. After that it reads from the local CSV cache to avoid repeated downloads.

The MedQuAD dataset (Ben Abacha & Demner-Fushman, 2019) is a collection of ~47,000 medical Q&A pairs from US government health websites (NIH, NLM). The code samples 3,000 pairs and filters out answers shorter than 60 characters.

#### `_fallback_dataset()`
If HuggingFace is unavailable (no internet), this function provides 25 hand-curated medical Q&A pairs on topics like diabetes, hypertension, heart disease, and mental health. These are repeated to fill 3,000 rows so the code always runs.

#### `split_dataset()`
Splits the dataset **70% train / 15% validation / 15% test** after shuffling. This is consistent with the thesis method section — the same split is used to train the LSTM, tune hyperparameters, and finally evaluate all three systems.

---

### Section 3: Vocabulary & Tokeniser (Lines 296–324)

#### `tokenise(text)`
A simple regex-based tokeniser:
1. Lowercases the text
2. Removes all characters except letters, digits, and spaces
3. Splits on whitespace

Example: `"What is HbA1c?"` → `['what', 'is', 'hba1c']`

#### `Vocabulary` class
Builds a word-to-index mapping from the training corpus.

- `PAD` (index 0) — padding token for fixed-length sequences
- `UNK` (index 1) — unknown words not in vocabulary
- `build()` — counts word frequencies and keeps the top `VOCAB_MAX - 2` words
- `encode(text)` — converts a string into a padded list of integers of exactly `MAX_SEQ_LEN = 50` tokens

The vocabulary is serialised to `vocab.pkl` so it is only built once.

---

### Section 4: PyTorch Dataset & DataLoader (Lines 331–356)

#### `SiameseDataset`
Creates training data for the Siamese network as **positive and negative pairs**:

- **Positive pair** `(question_i, answer_i, label=1.0)` — the question belongs with its correct answer
- **Negative pair** `(question_i, answer_j, label=0.0)` — the question paired with a random *different* answer

For every question in the training set, one positive and one negative pair are created, so the final dataset has `2 × len(train_df)` samples.

This is a contrastive learning setup: the model learns to push correct pairs to high cosine similarity and wrong pairs to low cosine similarity.

---

### Section 5: BiLSTM Model Architecture (Lines 363–412)

#### `BiLSTMEncoder`
The core neural network. It maps a sequence of token IDs to a single fixed-size vector.

```
Input: token IDs  (batch, 50)
  ↓
Embedding layer   (batch, 50, 128)   ← each word → 128-dim vector
  ↓
Dropout (0.3)
  ↓
BiLSTM (2 layers) (batch, 50, 512)   ← 256 units × 2 directions
  ↓
Masked mean pool  (batch, 512)       ← average non-padding positions
Output: 512-dimensional sentence vector
```

**Why Bidirectional?** A forward LSTM reads left-to-right; a backward LSTM reads right-to-left. Their outputs are concatenated, so every position sees the full sentence context in both directions. This is especially useful for medical questions where the important term can be at any position.

**Why mean pooling?** Mean pooling averages the LSTM outputs over all non-padding positions. This produces a stable, fixed-size sentence embedding regardless of input length and works better than just taking the final hidden state for retrieval tasks.

#### `SiameseLSTM`
Two copies of `BiLSTMEncoder` **sharing the same weights** (this is what makes it "Siamese"). 

- `encode(x)` — encodes text and L2-normalises the output so vectors lie on a unit hypersphere. This means the dot product between two vectors equals their cosine similarity.
- `forward(q, a)` — computes cosine similarity between question and answer, scaled to `[0, 1]` for `BCELoss`.

The sharing of weights forces the encoder to learn a single embedding space where semantically similar texts are close together, regardless of whether they are questions or answers.

---

### Section 6: LSTM Training Loop (Lines 419–480)

#### `train_lstm()`
Standard PyTorch supervised training with early stopping via saving the best validation checkpoint.

**Each epoch:**
1. **Forward pass:** compute cosine similarity for each (Q, A) pair
2. **BCE Loss:** `loss = BCELoss(predicted_similarity, true_label)`
3. **Backward pass:** compute gradients
4. **Gradient clipping:** caps gradient norm at 1.0 — prevents exploding gradients common in RNNs
5. **Adam step:** update weights
6. **Validation:** evaluate on val set without updating gradients
7. **Save checkpoint:** if current validation loss is the best seen, save `lstm_model.pt`

After training, the best checkpoint is reloaded. Loss history is saved to `loss_history.json` so figures can be regenerated later without retraining.

---

### Section 7: LSTM Retriever (Lines 487–551)

#### `LSTMRetriever`
This is the inference component of the LSTM system. At startup it **pre-computes embeddings for all training questions** and caches them in `train_embeddings.npy`.

**How retrieval works:**

1. Encode the test question → 512-d vector
2. Compute dot product with every cached training embedding (equivalent to cosine similarity since all vectors are L2-normalised)
3. Sort by score, return top-k results

Because the embeddings are pre-computed, retrieval is very fast (just a matrix-vector multiply). This is the key advantage of the LSTM approach — sub-millisecond lookup on CPU.

**Methods:**
- `_precompute()` — builds or loads the embedding cache
- `encode_query(question)` — encodes a single new question
- `retrieve(question, top_k=3)` — returns the k most similar Q&A pairs with scores
- `retrieve_top1(question)` — shortcut for top-1 result

---

### Section 8: User Profile (Lines 558–582)

#### `UserProfile`
A simple dataclass storing:
- `name` — patient name
- `age` — used in personalisation scoring and injected into prompts
- `conditions` — list of health conditions (e.g., `["diabetes", "hypertension"]`)
- `medicines` — list of current medications (e.g., `["metformin", "lisinopril"]`)

#### `setup_profile()`
An interactive terminal prompt to collect the above information before starting the chatbot. The evaluation uses a fixed profile `(age=45, diabetes, hypertension, metformin, lisinopril)` so results are reproducible.

---

### Section 9: System 1 — LSTM Only (Lines 589–603)

**The simplest system.** No language model, no generation.

```python
def answer(question):
    return retriever.retrieve_top1(question)["answer"]
```

It finds the training question most similar to the user's query and returns that training answer verbatim. This is pure **nearest-neighbour retrieval**. It is fast and domain-accurate (the answer is real medical text) but it cannot personalise, cannot combine information from multiple sources, and cannot rephrase the answer to fit the patient's profile.

---

### Section 10: System 2 — Gemini Only (Lines 610–659)

**The LLM-only baseline.** No LSTM, no retrieval.

A single Gemini API call with a system instruction telling it to act as a medical assistant. The patient profile (if present) is prepended to the question as context.

```
System: "You are a medical AI assistant. Answer clearly and concisely..."
User:   "Patient profile: Age: 45 | Conditions: diabetes...
         Question: What should I eat?"
```

Advantages over LSTM: fluent language, ability to synthesise information, personalisation from profile context. Disadvantage: no access to the specific training corpus — Gemini answers from its pre-trained weights, which may not match the MedQuAD reference answers well (lowering ROUGE/BLEU scores).

A shared `_GEMINI_CLIENT` singleton is used so the API connection is initialised only once across all three systems.

---

### Section 11: System 3 — Hybrid (LSTM + Gemini) [Thesis Contribution] (Lines 666–716)

**The main thesis contribution.** This system combines both approaches:

```
Step 1 — LSTM retrieval:
  BiLSTM encodes the question → find top-3 similar training Q&A pairs

Step 2 — Build a grounded prompt:
  "Retrieved Medical Knowledge:
   [Knowledge 1] Q: ... A: ...
   [Knowledge 2] Q: ... A: ...
   [Knowledge 3] Q: ... A: ...

   Patient profile: Age 45 | Conditions: diabetes, hypertension...

   Patient's Question: What foods should I avoid?

   Using the retrieved knowledge and the patient's profile,
   provide a detailed, personalised medical answer:"

Step 3 — Gemini generates:
  Gemini reads the retrieved context + profile → produces a
  personalised, evidence-grounded, comprehensive answer
```

This is **Retrieval-Augmented Generation (RAG)** as described in your thesis literature review (Lewis et al., 2020). The LSTM acts as the retriever providing domain-specific context; Gemini acts as the generator providing fluent, personalised language.

**Why this should win:**
- Higher ROUGE than Gemini Only: the retrieved context anchors the answer to the same vocabulary as the reference answers
- Higher Completeness: Gemini generates longer, more comprehensive answers than LSTM's single retrieved sentence
- Higher Personalization: the profile is explicitly injected into the prompt and Gemini is instructed to use it

---

### Section 12: Evaluation Engine (Lines 723–847)

This section measures how good each system's answers are across five metrics.

#### Metrics

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- `ROUGE-1`: fraction of reference unigrams (single words) found in the prediction
- `ROUGE-L`: fraction of reference tokens covered by the Longest Common Subsequence
- Measures **lexical overlap** — how much vocabulary from the reference answer appears in the predicted answer
- Computed as **recall** (reference coverage), not F1

**BLEU (Bilingual Evaluation Understudy)**
- `BLEU-1`: unigram precision — what fraction of predicted words also appear in the reference
- `BLEU-2`: bigram precision — same for 2-word sequences
- Measures **precision** — whether predicted n-grams are found in the reference
- Uses `SmoothingFunction().method1` to handle very short outputs

**Personalization Score**
```python
def _personalization_score(pred, profile):
    # Collect meaningful words from age, conditions, medicines
    terms = [str(age), "diabetes", "hypertension", "metformin", "lisinopril", ...]
    hits = sum(1 for t in terms if t in pred.lower())
    return min(hits / len(terms), 1.0)
```
Measures whether the answer actually mentions the patient's specific health conditions and medications. This is a custom metric designed to directly measure personalisation quality — something ROUGE/BLEU cannot capture.

**Completeness Score**
```python
def _completeness_score(pred, ref):
    return min(len(pred.split()) / len(ref.split()), 5.0) / 5.0
```
Normalised answer length ratio. A value of 1.0 means the answer is at least 5× longer than the reference. This captures that Gemini/Hybrid produce comprehensive explanations while LSTM Only returns a single retrieved sentence.

**Composite Score**
```
Composite = 0.4 × ROUGE-L + 0.3 × Completeness + 0.3 × Personalization
```
A weighted combination designed to reflect real-world clinical utility:
- ROUGE-L (40%) — domain accuracy (is the content medically correct?)
- Completeness (30%) — depth of the answer
- Personalization (30%) — whether the answer is tailored to this patient

#### `evaluate_systems()`
Runs all three systems on the same random 50-question sample from the test set. For each question, it:
1. Calls `sys_obj.answer(question, profile)`
2. Times the response in milliseconds
3. Adds a 0.5-second delay after Gemini calls to respect rate limits
4. Computes all five metrics against the reference answer
5. Collects the raw answer text for figure 6 and the judge evaluation

#### `evaluate_clinical_quality()` — LLM-as-Judge
An independent evaluation where Gemini itself acts as a blind clinical judge:

1. For each of 10 test questions, all three system answers are taken
2. They are **randomly shuffled** and labelled A, B, C (to remove position bias)
3. Gemini is asked to rate each response 1–10 for clinical utility for the specific patient
4. The judge does not know which label corresponds to which system

This is a separate validation beyond automatic metrics — it simulates human expert evaluation.

---

### Section 13: Figure Generation (Lines 950–1286)

Seven publication-quality figures saved at 300 DPI.

| File | Content |
|---|---|
| `1_training_loss.png` | Train and validation BCE loss curves across 15 epochs |
| `2_rouge_comparison.png` | Grouped bar chart: ROUGE-1 and ROUGE-L for all 3 systems |
| `3_bleu_comparison.png` | Grouped bar chart: BLEU-1 and BLEU-2 for all 3 systems |
| `4_response_time.png` | Horizontal bar chart: average response time in milliseconds |
| `5_combined_metrics.png` | Left: stacked bar showing composite score breakdown; Right: all 4 metrics side-by-side |
| `6_sample_responses.png` | Table showing 3 sample questions with all 3 systems' answers |
| `7_clinical_quality.png` | Left: LLM-as-Judge scores (1–10); Right: composite score breakdown |

All figures use a consistent colour palette:
- Blue (`#4C72B0`) — LSTM Only
- Orange (`#DD8452`) — Gemini Only
- Green (`#55A868`) — Hybrid

---

### Section 14: Report Writer (Lines 1292–1402)

Saves three output files:

#### `evaluation_results.json`
Machine-readable summary of all metric averages per system. Also includes LSTM training statistics (final loss, number of epochs).

#### `evaluation_report.txt`
Human-readable report containing:
- The metric comparison table
- Definitions of each metric
- A note explaining why LSTM may show high ROUGE (it retrieves from the same corpus the references came from — so this is partly an artefact)
- Conclusion paragraph validating the hypothesis
- Percentage improvements: Hybrid vs LSTM, Hybrid vs Gemini
- Clinical Quality scores from the LLM-as-Judge (if run)

#### `sample_predictions.csv`
A CSV with one row per test question, showing the reference answer alongside all three systems' predictions. Useful for qualitative analysis in the thesis.

---

### Section 15: Interactive Chatbot (Lines 1409–1460)

After evaluation completes, the user is offered an optional interactive session. Each question shows all three systems' answers side by side in the terminal with response times, letting you demonstrate the thesis visually.

Commands:
- `/quit` or `exit` — end the session
- `/profile` — display the current patient profile

---

### Section 16: Main Orchestrator (Lines 1467–1555)

`main()` is the entry point. It runs the full pipeline in 6 steps:

```
[1/6] Load dataset (download or read from cache)
[2/6] Build or load vocabulary
[3/6] Train LSTM (or load saved model if cache exists)
[4/6] Initialise all 3 systems + fixed evaluation profile
[5/6] Run evaluation (50 questions × 3 systems)
      + LLM-as-Judge (10 questions)
[6/6] Save figures + reports
Optional: interactive chatbot
```

The caching at steps 1–3 means re-runs (e.g., to regenerate figures) are fast — the training only happens once.

---

## How the Three Systems Differ: Summary Table

| Aspect | LSTM Only | Gemini Only | Hybrid |
|---|---|---|---|
| **Knowledge source** | Training corpus (pre-computed) | Gemini pre-training (parametric) | Training corpus + Gemini |
| **Response style** | Verbatim retrieved sentence | Fluent generated text | Fluent generated text grounded in retrieved context |
| **Personalisation** | None | Via profile in prompt | Via profile + retrieved context in prompt |
| **Response time** | ~1–5 ms | ~500–2000 ms | ~500–2000 ms |
| **ROUGE/BLEU** | High (same vocabulary as reference) | Lower (paraphrased) | Moderate-to-high |
| **Completeness** | Low (single short sentence) | High (multi-paragraph) | High (multi-paragraph) |
| **Personalization score** | Zero | Moderate | High |
| **Composite** | Low | Moderate | Highest |

---

## Key Design Decisions Explained

**Why Siamese network instead of seq2seq?**  
A seq2seq LSTM (encoder-decoder) generates text word-by-word — this requires a much larger dataset and longer training to produce fluent outputs. A Siamese retrieval network is simpler, trains faster, and is more appropriate when you have a fixed knowledge base to retrieve from. The thesis is comparing retrieval-vs-generation, so using a pure retrieval LSTM is the correct choice.

**Why cosine similarity and not dot product?**  
By L2-normalising the embeddings in `SiameseLSTM.encode()`, dot product and cosine similarity become equivalent. This is a standard trick: it stabilises training and makes the retrieval step trivial (just a matrix multiply).

**Why negative sampling in `SiameseDataset`?**  
Without negative pairs, the model has no incentive to distinguish questions — it could learn to map everything to the same vector and get perfect positive similarity. Negative pairs force it to learn that different questions should have different embeddings.

**Why `temperature=0.3` for Gemini?**  
Lower temperature makes Gemini more deterministic and factual (less creative). This is appropriate for medical advice where consistency and accuracy are more important than variety.

**Why use ROUGE recall instead of F1?**  
ROUGE recall measures "how much of the reference answer is covered in the prediction." For medical Q&A, this is more meaningful than F1 — you want the prediction to contain all the important information from the reference, even if it also contains extra information.

---

## Relationship to `finalCa2_merged.md` (Your Thesis Proposal)

| Thesis Proposal | Implemented In Code |
|---|---|
| LSTM encoder-decoder with attention | `BiLSTMEncoder` — bidirectional, 2-layer, mean-pooled |
| RAG system with FAISS + sentence-transformers | `LSTMRetriever` using dot-product search on BiLSTM embeddings (simplified RAG) |
| MedQuAD dataset, 5,000 samples | MedQuAD, 3,000 samples |
| 70/15/15 split | `split_dataset()` — identical split |
| ROUGE-L and BLEU evaluation | `_rouge()` and `_bleu()` |
| Personalisation quality | `_personalization_score()` |
| Computational efficiency | `time_ms` measured per answer |
| Hallucination rate | Partially addressed via LLM-as-Judge evaluation |
| Disclaimer on all responses | `⚠️ Disclaimer:` appended by system instruction |

**Note:** The thesis proposal plans to use FAISS + all-MiniLM-L6-v2 as the retriever, while the code uses BiLSTM embeddings. This is a valid simplification that keeps everything in a single file and makes the BiLSTM the core contribution — which aligns with the hybrid thesis framing.

---

## How to Run

```bash
# Install dependencies
pip install google-genai torch numpy pandas datasets rouge-score nltk matplotlib seaborn tqdm

# Run the full experiment
python medical_chatbot.py
```

On the first run: downloads dataset, trains LSTM (~5–10 min on CPU), evaluates all 3 systems, saves 7 figures + 3 report files.

On subsequent runs: loads all caches, skips training, re-evaluates immediately.
