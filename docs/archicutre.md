flowchart TD
    classDef data   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef model  fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef api    fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef sys    fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef hybrid fill:#d1fae5,stroke:#059669,stroke-width:3px,color:#064e3b
    classDef eval   fill:#ede9fe,stroke:#7c3aed,color:#2e1065

    %% ── 1. Data Pipeline ──────────────────────────────────────────
    subgraph DP["① Data Pipeline"]
        direction LR
        HF[("HuggingFace\nMedQuAD")]
        CSV[("medquad.csv\n3,000 Q&A pairs")]
        SP(["70% Train  ·  15% Val  ·  15% Test"])
        HF -->|"first-run download"| CSV --> SP
    end

    %% ── 2. BiLSTM Training ────────────────────────────────────────
    subgraph LT["② BiLSTM Training"]
        direction TB
        VOC["Vocabulary\n5,000 tokens\nPAD + UNK special tokens"]
        SDS["SiameseDataset\nPositive pairs  label = 1\nNegative pairs  label = 0"]
        ENC["BiLSTM Encoder\nEmbedding 128-d\nBiLSTM hidden 256 × 2 layers\nDropout 0.3  →  512-d mean-pool"]
        SL["SiameseLSTM\nCosine similarity + BCELoss\nAdam lr=0.001  ·  15 epochs\nGrad-clip 1.0"]
        MDL[("lstm_model.pt\nbest-val checkpoint")]
        EMB[("train_embeddings.npy\nL2-normalised 512-d vectors")]
        VOC --> SDS --> ENC --> SL -->|"save best val loss"| MDL
        MDL -->|"pre-compute all train questions"| EMB
    end

    %% ── Shared Inference Components ───────────────────────────────
    RET["LSTMRetriever\nCosine similarity search\nover pre-computed embeddings"]
    UP[/"UserProfile\nAge  ·  Conditions  ·  Medicines"/]
    GAPI[/"Google Gemini 2.5 Flash API\ntemp=0.3  ·  max_tokens=512"/]
    Q[/"Patient Question"/]

    %% ── 3. Three Inference Systems ────────────────────────────────
    subgraph SYS["③ Three Inference Systems"]
        direction LR
        S1["LSTM Only\n──────────────\nEncode query\nTop-1 cosine match\nReturn stored answer\nPure retrieval · No LLM"]
        S2["Gemini Only\n──────────────\nPatient profile in prompt\nDirect LLM generation\nNo retrieval context"]
        S3["Hybrid  LSTM + Gemini\n──────────────\nTop-3 BiLSTM retrieval\nRAG prompt with contexts\nPatient profile injected\nGemini grounded generation"]
    end

    %% ── 4. Evaluation Engine ──────────────────────────────────────
    subgraph EV["④ Evaluation  (50 test questions per system)"]
        direction LR
        AM["Automated Metrics\nROUGE-1  ·  ROUGE-L\nBLEU-1  ·  BLEU-2\nPersonalization Score\nCompleteness Score\nComposite = 0.4xROUGE-L\n+ 0.3xCompleteness\n+ 0.3xPersonalization"]
        JG["LLM-as-Judge\nGemini blind clinical rating\n10 questions  ·  score 1-10\nRandomised A/B/C order\nto remove position bias"]
    end

    %% ── 5. Output Artefacts ───────────────────────────────────────
    subgraph OUT["⑤ Output Artefacts"]
        direction LR
        FG["7 Figures at 300 DPI\nTraining loss curves\nROUGE  ·  BLEU  ·  Speed\nComposite  ·  Clinical Quality"]
        RP["Reports\nevaluation_results.json\nevaluation_report.txt\nsample_predictions.csv"]
    end

    %% ── Data flow ─────────────────────────────────────────────────
    DP -->|"Train + Val sets"| LT
    DP -->|"Train QA corpus"| RET
    EMB --> RET
    MDL --> RET
