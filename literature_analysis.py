"""
Literature Review Analysis Tool
MSc AI Thesis — Mastan Vali Shaik (24226807), National College of Ireland
Topic: LSTM Fine-Tuning vs RAG for Personalised Medical Chatbot
Generates publication-quality figures (DPI=300) saved to output/figures/
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
import textwrap

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Paper Database ────────────────────────────────────────────────────────────
PAPERS = [
    # id, author_short, year, title_short, venue, theme, cites_approx
    {"id":  1, "author": "Hochreiter & Schmidhuber", "year": 1997,
     "title": "Long Short-Term Memory",
     "venue": "Neural Computation", "venue_type": "Journal",
     "theme": "LSTM / Sequence Models", "core_rank": "A*", "cites": 75000},

    {"id":  2, "author": "Wirth & Hipe",             "year": 2000,
     "title": "CRISP-DM",
     "venue": "KDD Workshop", "venue_type": "Conference",
     "theme": "Methodology", "core_rank": "N/A", "cites": 4500},

    {"id":  3, "author": "Papineni et al.",           "year": 2002,
     "title": "BLEU: Automatic MT Evaluation",
     "venue": "ACL", "venue_type": "Conference",
     "theme": "Evaluation Metrics", "core_rank": "A*", "cites": 22000},

    {"id":  4, "author": "Lin",                       "year": 2004,
     "title": "ROUGE: Automatic Summarisation Evaluation",
     "venue": "ACL Workshop", "venue_type": "Workshop",
     "theme": "Evaluation Metrics", "core_rank": "A*", "cites": 18000},

    {"id":  5, "author": "Mikolov et al.",            "year": 2013,
     "title": "Word2Vec: Distributed Word Representations",
     "venue": "NeurIPS", "venue_type": "Conference",
     "theme": "LSTM / Sequence Models", "core_rank": "A*", "cites": 55000},

    {"id":  6, "author": "Vaswani et al.",            "year": 2017,
     "title": "Attention Is All You Need",
     "venue": "NeurIPS", "venue_type": "Conference",
     "theme": "Transformer & LLMs", "core_rank": "A*", "cites": 90000},

    {"id":  7, "author": "Johnson et al.",            "year": 2017,
     "title": "FAISS: Billion-Scale Similarity Search",
     "venue": "IEEE Trans. Big Data", "venue_type": "Journal",
     "theme": "RAG Infrastructure", "core_rank": "A", "cites": 8000},

    {"id":  8, "author": "Rajkomar et al.",           "year": 2018,
     "title": "Scalable Deep Learning with EHR",
     "venue": "NPJ Digital Medicine", "venue_type": "Journal",
     "theme": "Medical AI / Healthcare", "core_rank": "N/A", "cites": 3500},

    {"id":  9, "author": "Pampari et al.",            "year": 2018,
     "title": "emrQA: Medical QA from EHR",
     "venue": "EMNLP", "venue_type": "Conference",
     "theme": "Medical QA & Dialogue", "core_rank": "A*", "cites": 520},

    {"id": 10, "author": "Devlin et al.",             "year": 2019,
     "title": "BERT: Bidirectional Transformer Pre-Training",
     "venue": "NAACL", "venue_type": "Conference",
     "theme": "Transformer & LLMs", "core_rank": "A*", "cites": 85000},

    {"id": 11, "author": "Huang et al.",              "year": 2019,
     "title": "ClinicalBERT: Modelling Clinical Notes",
     "venue": "arXiv / CHIL Workshop", "venue_type": "Preprint/Workshop",
     "theme": "Domain-Adapted NLP", "core_rank": "N/A", "cites": 3200},

    {"id": 12, "author": "Radford et al.",            "year": 2019,
     "title": "GPT-2: Language Models as Multitask Learners",
     "venue": "OpenAI Blog", "venue_type": "Technical Report",
     "theme": "Transformer & LLMs", "core_rank": "N/A", "cites": 12000},

    {"id": 13, "author": "Reimers & Gurevych",        "year": 2019,
     "title": "Sentence-BERT: Semantic Sentence Embeddings",
     "venue": "EMNLP", "venue_type": "Conference",
     "theme": "RAG Infrastructure", "core_rank": "A*", "cites": 10000},

    {"id": 14, "author": "Ben Abacha & Demner-Fushman", "year": 2019,
     "title": "MedQuAD: Medical Question Answering",
     "venue": "BMC Bioinformatics", "venue_type": "Journal",
     "theme": "Medical QA & Dialogue", "core_rank": "A", "cites": 450},

    {"id": 15, "author": "Brown et al.",              "year": 2020,
     "title": "GPT-3: Few-Shot Language Learners",
     "venue": "NeurIPS", "venue_type": "Conference",
     "theme": "Transformer & LLMs", "core_rank": "A*", "cites": 42000},

    {"id": 16, "author": "Lewis et al.",              "year": 2020,
     "title": "RAG for Knowledge-Intensive NLP",
     "venue": "NeurIPS", "venue_type": "Conference",
     "theme": "RAG Infrastructure", "core_rank": "A*", "cites": 7500},

    {"id": 17, "author": "Guu et al.",                "year": 2020,
     "title": "REALM: Retrieval-Augmented LM Pre-Training",
     "venue": "ICML", "venue_type": "Conference",
     "theme": "RAG Infrastructure", "core_rank": "A*", "cites": 2800},

    {"id": 18, "author": "Lee et al.",                "year": 2020,
     "title": "BioBERT: Biomedical Language Representation",
     "venue": "Bioinformatics (Oxford)", "venue_type": "Journal",
     "theme": "Domain-Adapted NLP", "core_rank": "A", "cites": 9000},

    {"id": 19, "author": "Sherstinsky",               "year": 2020,
     "title": "Fundamentals of RNN and LSTM Networks",
     "venue": "Physica D", "venue_type": "Journal",
     "theme": "LSTM / Sequence Models", "core_rank": "N/A", "cites": 1200},

    {"id": 20, "author": "Zhang et al.",              "year": 2020,
     "title": "BERTScore: Text Generation Evaluation",
     "venue": "ICLR", "venue_type": "Conference",
     "theme": "Evaluation Metrics", "core_rank": "A*", "cites": 6500},

    {"id": 21, "author": "Johnson et al.",            "year": 2016,
     "title": "MIMIC-III Clinical Database",
     "venue": "Scientific Data (Nature)", "venue_type": "Journal",
     "theme": "Medical AI / Healthcare", "core_rank": "N/A", "cites": 12000},

    {"id": 22, "author": "Roller et al.",             "year": 2021,
     "title": "Recipes for an Open-Domain Chatbot (BlenderBot)",
     "venue": "EACL", "venue_type": "Conference",
     "theme": "Medical QA & Dialogue", "core_rank": "A*", "cites": 2100},

    {"id": 23, "author": "Thoppilan et al.",          "year": 2022,
     "title": "LaMDA: Language Models for Dialogue",
     "venue": "arXiv / Google", "venue_type": "Technical Report",
     "theme": "Medical QA & Dialogue", "core_rank": "N/A", "cites": 1800},

    {"id": 24, "author": "Ouyang et al.",             "year": 2022,
     "title": "InstructGPT: RLHF for Instruction Following",
     "venue": "NeurIPS", "venue_type": "Conference",
     "theme": "Transformer & LLMs", "core_rank": "A*", "cites": 7200},

    {"id": 25, "author": "Huang et al.",              "year": 2021,
     "title": "Knowledge Graph QA for Medical Domain",
     "venue": "PeerJ Computer Science", "venue_type": "Journal",
     "theme": "Medical QA & Dialogue", "core_rank": "N/A", "cites": 180},

    {"id": 26, "author": "Li et al.",                 "year": 2023,
     "title": "ChatDoctor: Medical LLM Fine-Tuning",
     "venue": "arXiv", "venue_type": "Preprint/Workshop",
     "theme": "Medical AI / Healthcare", "core_rank": "N/A", "cites": 950},

    {"id": 27, "author": "Singhal et al.",            "year": 2023,
     "title": "Med-PaLM: LLMs Encode Clinical Knowledge",
     "venue": "Nature", "venue_type": "Journal",
     "theme": "Medical AI / Healthcare", "core_rank": "N/A", "cites": 2400},

    {"id": 28, "author": "Touvron et al.",            "year": 2023,
     "title": "LLaMA: Open Efficient Foundation LMs",
     "venue": "arXiv / Meta AI", "venue_type": "Technical Report",
     "theme": "Transformer & LLMs", "core_rank": "N/A", "cites": 13000},

    {"id": 29, "author": "OpenAI",                    "year": 2023,
     "title": "GPT-4 Technical Report",
     "venue": "arXiv / OpenAI", "venue_type": "Technical Report",
     "theme": "Transformer & LLMs", "core_rank": "N/A", "cites": 11000},

    {"id": 30, "author": "Zakka et al.",              "year": 2024,
     "title": "Almanac: RAG for Clinical Medicine",
     "venue": "NEJM AI", "venue_type": "Journal",
     "theme": "Medical AI / Healthcare", "core_rank": "N/A", "cites": 420},
]

THEME_COLORS = {
    "LSTM / Sequence Models":  "#2E86AB",
    "Transformer & LLMs":      "#A23B72",
    "Domain-Adapted NLP":      "#F18F01",
    "RAG Infrastructure":      "#C73E1D",
    "Medical QA & Dialogue":   "#3B1F2B",
    "Medical AI / Healthcare": "#44BBA4",
    "Evaluation Metrics":      "#E94F37",
    "Methodology":             "#6B4226",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, name, dpi=300):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)


def theme_counter():
    return Counter(p["theme"] for p in PAPERS)


def year_theme_matrix():
    years = sorted(set(p["year"] for p in PAPERS))
    themes = list(THEME_COLORS.keys())
    mat = defaultdict(lambda: defaultdict(int))
    for p in PAPERS:
        mat[p["year"]][p["theme"]] += 1
    return years, themes, mat

# ── Figure 1: Publication Timeline ────────────────────────────────────────────

def fig_timeline():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')

    years = sorted(set(p["year"] for p in PAPERS))
    themes = list(THEME_COLORS.keys())
    yr_theme = defaultdict(lambda: defaultdict(int))
    for p in PAPERS:
        yr_theme[p["year"]][p["theme"]] += 1

    bottoms = {y: 0 for y in years}
    x = list(range(len(years)))
    yr_to_x = {y: i for i, y in enumerate(years)}

    for theme in themes:
        heights = [yr_theme[y][theme] for y in years]
        bars = ax.bar(x, heights, bottom=[bottoms[y] for y in years],
                      color=THEME_COLORS[theme], edgecolor='white',
                      linewidth=0.5, width=0.6, label=theme)
        for y, h in zip(years, heights):
            bottoms[y] += h

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Papers', fontsize=11)
    ax.set_title('Figure LR-1  Publication Timeline of 30 Reviewed Papers by Theme\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8, ncol=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'LR1_publication_timeline.png')

# ── Figure 2: Theme Distribution Pie ──────────────────────────────────────────

def fig_theme_pie():
    counts = theme_counter()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#F7F9FC')

    # Pie
    ax = axes[0]
    ax.set_facecolor('#F7F9FC')
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [THEME_COLORS[l] for l in labels]
    explode = [0.04] * len(labels)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.0f%%',
        colors=colors, explode=explode, startangle=140,
        pctdistance=0.78, wedgeprops=dict(edgecolor='white', linewidth=1.2)
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, [f"{l} ({c})" for l, c in zip(labels, sizes)],
              loc='lower center', bbox_to_anchor=(0.5, -0.18),
              fontsize=8, ncol=2, framealpha=0.8)
    ax.set_title('(a) Papers by Research Theme', fontsize=11, fontweight='bold')

    # Bar
    ax2 = axes[1]
    ax2.set_facecolor('#F7F9FC')
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names = ['\n'.join(textwrap.wrap(k, 18)) for k, _ in sorted_items]
    vals  = [v for _, v in sorted_items]
    bar_colors = [THEME_COLORS[k] for k, _ in sorted_items]
    bars = ax2.barh(names, vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', ha='left', fontsize=9)
    ax2.set_xlabel('Number of Papers', fontsize=10)
    ax2.set_title('(b) Count by Theme', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, max(vals) + 1.5)
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Figure LR-2  Research Theme Distribution of the 30 Reviewed Papers\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'LR2_theme_distribution.png')

# ── Figure 3: Venue Type ───────────────────────────────────────────────────────

def fig_venue():
    venue_counts = Counter(p["venue_type"] for p in PAPERS)
    venue_theme  = defaultdict(lambda: defaultdict(int))
    for p in PAPERS:
        venue_theme[p["venue_type"]][p["theme"]] += 1

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')

    venues = sorted(venue_counts.keys(), key=lambda v: venue_counts[v], reverse=True)
    themes = list(THEME_COLORS.keys())
    x = np.arange(len(venues))
    width = 0.12
    for i, theme in enumerate(themes):
        heights = [venue_theme[v][theme] for v in venues]
        ax.bar(x + i * width - width * len(themes) / 2 + width / 2,
               heights, width, label=theme,
               color=THEME_COLORS[theme], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(venues, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Number of Papers', fontsize=10)
    ax.set_title('Figure LR-3  Publication Venue Type by Research Theme\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=7.5, framealpha=0.8, ncol=2, loc='upper right')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'LR3_venue_distribution.png')

# ── Figure 4: Citation Impact Scatter ─────────────────────────────────────────

def fig_citations():
    fig, ax = plt.subplots(figsize=(13, 6), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')

    for p in PAPERS:
        c = THEME_COLORS[p["theme"]]
        size = max(30, min(800, p["cites"] / 100))
        ax.scatter(p["year"], p["cites"], s=size, color=c,
                   alpha=0.75, edgecolors='white', linewidths=0.8, zorder=3)
        if p["cites"] > 5000 or p["id"] in {1, 14, 16, 27, 30}:
            ax.annotate(p["author"].split(' &')[0].split(' et')[0],
                        (p["year"], p["cites"]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7.5, color='#333333')

    legend_handles = [
        mpatches.Patch(color=THEME_COLORS[t], label=t)
        for t in THEME_COLORS
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.8,
              loc='upper left', ncol=2)
    ax.set_yscale('log')
    ax.set_xlabel('Publication Year', fontsize=11)
    ax.set_ylabel('Approximate Citations (log scale)', fontsize=11)
    ax.set_title('Figure LR-4  Citation Impact vs Publication Year (bubble size ∝ citations)\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'LR4_citation_impact.png')

# ── Figure 5: Decade-level Heatmap ────────────────────────────────────────────

def fig_heatmap():
    themes = list(THEME_COLORS.keys())
    years  = sorted(set(p["year"] for p in PAPERS))
    matrix = np.zeros((len(themes), len(years)), dtype=int)
    for p in PAPERS:
        ti = themes.index(p["theme"])
        yi = years.index(p["year"])
        matrix[ti, yi] += 1

    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor='#F7F9FC')
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels(themes, fontsize=9)
    for i in range(len(themes)):
        for j in range(len(years)):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]),
                        ha='center', va='center', fontsize=9,
                        color='white' if matrix[i, j] >= 2 else '#333333')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Papers per cell')
    ax.set_title('Figure LR-5  Research Theme × Publication Year Heatmap (30 Papers)\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    save(fig, 'LR5_theme_year_heatmap.png')

# ── Figure 6: LSTM vs RAG Paradigm Comparison Table ───────────────────────────

def fig_paradigm_table():
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')
    ax.axis('off')

    columns = ['Dimension', 'LSTM Fine-Tuning', 'RAG System', 'Hybrid (LSTM + LLM)']
    rows = [
        ['Architecture',       'BiLSTM encoder-decoder',   'Retriever + LLM generator', 'LSTM retriever + LLM generator'],
        ['Knowledge Storage',  'Parametric (model weights)','Non-parametric (vector index)', 'Both'],
        ['Update Cost',        'Full retraining required',  'Index rebuild only',        'Partial (index only)'],
        ['Inference Latency',  'Very low (ms)',             'Medium (retrieval + gen)',  'Medium'],
        ['Hallucination Risk', 'High (no grounding)',       'Low (evidence-grounded)',   'Low'],
        ['Personalisation',    'User vector concat',        'Profile-filtered retrieval','Both mechanisms'],
        ['Key Papers',         'Hochreiter 1997\nSherstinsky 2020', 'Lewis 2020\nSinghal 2023', 'This Thesis'],
        ['Dataset',            'MedQuAD (3,000 samples)',   'MedQuAD (indexed)',         'MedQuAD (unified)'],
        ['Primary Metric',     'ROUGE-L, BLEU-4',           'BERTScore, ROUGE-L',        'All metrics'],
    ]

    col_colors = ['#2c3e50'] + ['#2E86AB', '#C73E1D', '#44BBA4']
    table = ax.table(
        cellText=rows, colLabels=columns,
        loc='center', cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.0)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(col_colors[c])
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EAF2FF' if c == 1 else '#FDECEA' if c == 2 else '#E8FAF4' if c == 3 else '#F5F5F5')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#CCCCCC')

    ax.set_title('Figure LR-6  Comparative Overview of Three AI Chatbot Paradigms\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=8, y=0.98)
    fig.tight_layout()
    save(fig, 'LR6_paradigm_comparison_table.png')

# ── Figure 7: CORE Rank Distribution ──────────────────────────────────────────

def fig_core_rank():
    rank_counts = Counter(p["core_rank"] for p in PAPERS)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')

    ranks = ['A*', 'A', 'N/A']
    colors_rank = ['#2E86AB', '#F18F01', '#AAAAAA']
    vals = [rank_counts.get(r, 0) for r in ranks]
    bars = ax.bar(ranks, vals, color=colors_rank, edgecolor='white',
                  linewidth=1.0, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Papers', fontsize=11)
    ax.set_title('Figure LR-7  CORE Venue Ranking of 30 Reviewed Papers\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylim(0, max(vals) + 2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    notes = ['Top-tier conferences\n& journals',
             'High-quality venues',
             'Technical reports,\narXiv, workshops']
    for i, (bar, note) in enumerate(zip(bars, notes)):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.3,
                note, ha='center', va='bottom', fontsize=8,
                color='#555555', style='italic')
    fig.tight_layout()
    save(fig, 'LR7_core_rank_distribution.png')

# ── Figure 8: Research Gap Roadmap ────────────────────────────────────────────

def fig_gap_roadmap():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    ax.set_title('Figure LR-8  Research Gap & Contribution Roadmap\n'
                 'MSc AI Thesis — Mastan Vali Shaik (24226807), NCI',
                 fontsize=11, fontweight='bold', pad=10)

    sections = [
        (1.0, 5.8, "#2E86AB", "LSTM\nFoundations",
         "Hochreiter 1997\nSherstinsky 2020\nMikolov 2013\nRajkomar 2018\nPampari 2018"),
        (3.5, 5.8, "#A23B72", "Transformer\n& LLM Era",
         "Vaswani 2017\nDevlin 2019\nBrown 2020\nRadford 2019\nOpenAI 2023"),
        (6.0, 5.8, "#C73E1D", "RAG\nFrameworks",
         "Lewis 2020\nGuu 2020\nJohnson 2017\nReimers 2019\nOuyang 2022"),
        (8.5, 5.8, "#44BBA4", "Medical\nAI",
         "Singhal 2023\nZakka 2024\nLi 2023\nTouvron 2023\nHuang 2021"),
    ]

    for x, y, color, title, papers_text in sections:
        rect = mpatches.FancyBboxPatch((x - 1.1, y - 2.0), 2.2, 2.4,
                                        boxstyle="round,pad=0.15",
                                        linewidth=1.5, edgecolor=color,
                                        facecolor=color + '22')
        ax.add_patch(rect)
        ax.text(x, y + 0.15, title, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=color)
        ax.text(x, y - 1.0, papers_text, ha='center', va='center',
                fontsize=7.5, color='#333333')

    for x in [2.4, 4.9, 7.4]:
        ax.annotate('', xy=(x + 0.2, 5.6), xytext=(x - 0.2, 5.6),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    gap_box = mpatches.FancyBboxPatch((2.5, 2.0), 5.0, 1.4,
                                       boxstyle="round,pad=0.2",
                                       linewidth=2, edgecolor='#E94F37',
                                       facecolor='#FDECEA')
    ax.add_patch(gap_box)
    ax.text(5.0, 2.7, 'Research Gap', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#E94F37')
    ax.text(5.0, 2.25,
            'No controlled, metric-consistent comparison of LSTM vs RAG\n'
            'with identical personalisation on the same medical QA dataset',
            ha='center', va='center', fontsize=9, color='#333333')

    contrib_box = mpatches.FancyBboxPatch((2.5, 0.3), 5.0, 1.4,
                                           boxstyle="round,pad=0.2",
                                           linewidth=2, edgecolor='#44BBA4',
                                           facecolor='#E8FAF4')
    ax.add_patch(contrib_box)
    ax.text(5.0, 1.0, 'Thesis Contribution', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#44BBA4')
    ax.text(5.0, 0.58,
            'Hybrid BiLSTM + Gemini system vs LSTM Only vs Gemini RAG Only\n'
            'MedQuAD dataset · ROUGE-L · BERTScore · BLEU · Personalisation Score',
            ha='center', va='center', fontsize=9, color='#333333')

    ax.annotate('', xy=(5.0, 2.0), xytext=(5.0, 3.4),
                arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.5))
    ax.annotate('', xy=(5.0, 1.7), xytext=(5.0, 2.0),
                arrowprops=dict(arrowstyle='->', color='#44BBA4', lw=1.5))

    fig.tight_layout()
    save(fig, 'LR8_research_gap_roadmap.png')

# ── Text Report ───────────────────────────────────────────────────────────────

def generate_report():
    lines = []
    lines.append("=" * 72)
    lines.append("LITERATURE REVIEW ANALYSIS REPORT")
    lines.append("MSc AI Thesis — Mastan Vali Shaik (24226807)")
    lines.append("National College of Ireland")
    lines.append("=" * 72)
    lines.append(f"\nTotal papers reviewed: {len(PAPERS)}")
    lines.append(f"Year range: {min(p['year'] for p in PAPERS)} – {max(p['year'] for p in PAPERS)}")
    lines.append(f"Median publication year: {sorted(p['year'] for p in PAPERS)[len(PAPERS)//2]}")

    lines.append("\n--- Theme Distribution ---")
    tc = theme_counter()
    for theme, count in sorted(tc.items(), key=lambda x: -x[1]):
        lines.append(f"  {theme:<35} {count:>2} papers")

    lines.append("\n--- Venue Type Distribution ---")
    vc = Counter(p["venue_type"] for p in PAPERS)
    for vtype, count in sorted(vc.items(), key=lambda x: -x[1]):
        lines.append(f"  {vtype:<30} {count:>2} papers")

    lines.append("\n--- CORE Venue Ranking ---")
    rc = Counter(p["core_rank"] for p in PAPERS)
    for rank in ['A*', 'A', 'N/A']:
        lines.append(f"  CORE {rank:<5} {rc.get(rank, 0):>2} papers")

    lines.append("\n--- Top 10 Papers by Estimated Citations ---")
    top = sorted(PAPERS, key=lambda x: x["cites"], reverse=True)[:10]
    for i, p in enumerate(top, 1):
        lines.append(f"  {i:>2}. {p['author']:<30} ({p['year']})  ~{p['cites']:,} cites")

    lines.append("\n--- Papers Added Beyond Original Thesis References ---")
    original_ids = {1, 4, 5, 9, 12, 20, 22, 24, 27, 29, 30}
    new = [p for p in PAPERS if p["id"] not in original_ids]
    for p in sorted(new, key=lambda x: x["year"]):
        lines.append(f"  [{p['id']:>2}] {p['author']:<30} ({p['year']})  {p['title'][:50]}")

    lines.append("\n--- Figures Generated ---")
    figures = [
        "LR1_publication_timeline.png   — Stacked bar timeline 1997–2024",
        "LR2_theme_distribution.png     — Pie + bar chart by theme",
        "LR3_venue_distribution.png     — Grouped bar by venue type & theme",
        "LR4_citation_impact.png        — Scatter: year vs citations (log)",
        "LR5_theme_year_heatmap.png     — Heatmap theme × year",
        "LR6_paradigm_comparison_table.png — LSTM vs RAG vs Hybrid table",
        "LR7_core_rank_distribution.png — CORE ranking bar chart",
        "LR8_research_gap_roadmap.png   — Research gap & contribution roadmap",
    ]
    for f in figures:
        lines.append(f"  {f}")

    lines.append("\n" + "=" * 72)
    report_text = "\n".join(lines)
    report_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'literature_review_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: {report_path}")
    return report_text


def save_paper_database():
    db_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'literature_papers.json')
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(PAPERS, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {db_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Literature Review Analysis — Generating Figures")
    print("=" * 60 + "\n")

    print("[1/8] Publication timeline...")
    fig_timeline()

    print("[2/8] Theme distribution...")
    fig_theme_pie()

    print("[3/8] Venue distribution...")
    fig_venue()

    print("[4/8] Citation impact scatter...")
    fig_citations()

    print("[5/8] Theme × year heatmap...")
    fig_heatmap()

    print("[6/8] Paradigm comparison table...")
    fig_paradigm_table()

    print("[7/8] CORE rank distribution...")
    fig_core_rank()

    print("[8/8] Research gap roadmap...")
    fig_gap_roadmap()

    print("\n[Report] Generating text report...")
    report = generate_report()

    print("[JSON]  Saving paper database...")
    save_paper_database()

    print("\n" + "=" * 60)
    print("  All outputs saved to output/")
    print("=" * 60)
    print(report)


if __name__ == '__main__':
    main()
