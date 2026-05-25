# Literature Review: Dynamic Healthcare Personalisation in AI Assistants — LSTM Fine-Tuning versus Retrieval-Augmented Generation

**Programme:** MSc in Artificial Intelligence
**Author:** Mastan Vali Shaik
**Student ID:** 24226807
**Institution:** National College of Ireland
**Module:** Research Methods / CA2

---

## 1. Introduction

Advances in artificial intelligence have transformed the development of healthcare dialogue systems, enabling conversational agents to provide personalised, evidence-based medical information. Two principal paradigms dominate the current landscape for building knowledge-intensive dialogue systems: (1) fine-tuning deep learning models — particularly Long Short-Term Memory (LSTM) recurrent networks — directly on domain-specific data, and (2) Retrieval-Augmented Generation (RAG), which combines a retrieval component with a generative language model to produce contextually grounded responses. The research question driving this thesis is: *which paradigm produces superior personalised medical responses in terms of accuracy, semantic quality, and efficiency?*

This literature review surveys thirty peer-reviewed and widely cited works that collectively define the theoretical foundations, technological evolution, and empirical benchmarks relevant to this question. The review is organised into eight thematic sections: (A) Sequence Modelling Foundations; (B) Transformer Architecture and Pre-trained Language Models; (C) Domain-Adapted Models for Biomedical and Clinical NLP; (D) Retrieval-Augmented Generation Frameworks; (E) Large Language Models in Healthcare; (F) Medical Dialogue and Question-Answering Systems; (G) Evaluation Metrics for Natural Language Generation; and (H) Data Infrastructure, Methodology, and Research Gaps. Where possible, papers within each section are critically compared, and contradictions or limitations in the existing literature are identified to justify the empirical contribution of this thesis.

---

## 2. Section A: Sequence Modelling Foundations — LSTM and Word Embeddings

### 2.1 Long Short-Term Memory Networks

The conceptual backbone of the fine-tuning approach examined in this thesis is the Long Short-Term Memory network, first proposed by Hochreiter and Schmidhuber (1997). The authors introduced gating mechanisms — an input gate, forget gate, and output gate — that allow networks to selectively retain or discard information over arbitrary time intervals, thereby resolving the vanishing gradient problem that had plagued earlier recurrent architectures. In a medical context, this is critical: a patient's symptom history, prior diagnoses, and medication timeline can span months or years of contextual data, and a model that cannot carry long-range dependencies will fail to personalise responses accurately.

Sherstinsky (2020) provides a rigorous theoretical treatment of LSTM networks aimed at practitioners, deriving the gating equations from first principles and analysing the sensitivity of LSTM performance to hyperparameters such as hidden layer dimensionality and dropout rate. Crucially, Sherstinsky highlights that practical LSTM configurations require careful regularisation and that the gap between LSTM expressivity and transformer-based models narrows considerably when datasets are small — a finding directly relevant to the low-resource medical domain. Taken together, Hochreiter and Schmidhuber (1997) establish the theoretical legitimacy of LSTM as a sequence modelling tool, while Sherstinsky (2020) calibrates practitioner expectations and informs the BiLSTM hyperparameter search strategy employed in this thesis.

### 2.2 Word Embeddings for Semantic Representation

Effective LSTM training for medical text requires meaningful numerical representations of medical vocabulary. Mikolov et al. (2013) introduced Word2Vec, demonstrating that neural word embeddings trained via continuous skip-gram or CBOW objectives capture syntactic and semantic regularities with remarkable fidelity (e.g., *king − man + woman ≈ queen*). In the medical domain, analogous relationships — such as *metformin − diabetes + hypertension ≈ lisinopril* — are precisely the kind of structured knowledge a fine-tuned LSTM must internalise to generate plausible therapeutic recommendations. Mikolov et al.'s method is computationally efficient, scales to large corpora, and produces 100–300 dimensional vectors that serve as effective initialisations for downstream NLP tasks. The thesis employs pre-trained word embeddings in the BiLSTM encoder to accelerate convergence on the limited MedQuAD training set.

### 2.3 Deep Learning on Electronic Health Records

Rajkomar et al. (2018) provided empirical evidence that deep learning models, including recurrent networks applied to raw electronic health record (EHR) data, match or exceed established clinical scoring systems on prediction tasks such as in-hospital mortality, 30-day readmission, and length of stay. By operating on heterogeneous, high-dimensional clinical time-series rather than curated tabular features, Rajkomar et al. demonstrated both the promise and the data-hunger of sequence models in clinical settings. Pampari et al. (2018) complemented this work by releasing emrQA, a large-scale question-answering corpus constructed from clinical notes in the i2b2 datasets, generating over one million question–logical-form pairs. The emrQA work is significant because it proved that clinical QA at scale is feasible when existing expert annotations are repurposed — a methodology that directly inspired MedQuAD and, by extension, the dataset used in this thesis.

---

## 3. Section B: Transformer Architecture and Pre-trained Language Models

### 3.1 The Attention Mechanism and the Transformer

Vaswani et al. (2017) introduced the Transformer architecture, replacing recurrence entirely with multi-head self-attention. The resulting model exhibits superior parallelism and captures long-range dependencies more efficiently than LSTMs, particularly at scale. While LSTMs remain competitive in low-resource and embedded deployment settings — as Sherstinsky (2020) notes — the Transformer has become the de facto backbone of virtually every high-performing NLP system. This thesis treats the Transformer as the representational foundation of the Gemini-based RAG pipeline rather than the LSTM fine-tuning system, providing a natural baseline comparison between the two architectural paradigms.

### 3.2 BERT and Bidirectional Contextual Representations

Devlin et al. (2019) presented BERT (Bidirectional Encoder Representations from Transformers), demonstrating that jointly conditioning on both left and right context during pre-training on large text corpora produces representations that transfer effectively to eleven downstream NLP tasks via fine-tuning with a single additional output layer. BERT achieved state-of-the-art results on SQuAD question answering, GLUE, and MultiNLI, establishing the template for pre-train-then-fine-tune that now dominates NLP research. For medical chatbots, BERT's ability to understand sentence-pair relationships — question and context — is directly relevant to the retrieval and answer-scoring components of a RAG pipeline. The BioBERT and ClinicalBERT models reviewed in Section C are direct adaptations of this architecture.

### 3.3 Generative Pre-trained Models: GPT-2 and GPT-3

While BERT targets understanding tasks, Radford et al. (2019) demonstrated with GPT-2 that autoregressive language models trained at scale on diverse web text acquire surprisingly general language generation capabilities without task-specific fine-tuning, achieving competitive results on reading comprehension and summarisation in zero-shot settings. Brown et al. (2020) scaled this insight dramatically with GPT-3, a 175-billion parameter model that exhibits strong few-shot in-context learning. GPT-3's performance across tasks including question answering and code generation showed that scale alone can produce emergent capabilities, motivating the use of large generative models as the response-generation backbone in RAG systems. The Gemini model used in this thesis belongs to this family of large-scale autoregressive generators.

### 3.4 GPT-4 and Multimodal Capabilities

OpenAI (2023) published the GPT-4 Technical Report, describing a large multimodal Transformer that accepts image and text inputs and achieves near-human performance on professional benchmarks including the medical licensing examination. GPT-4's reliability and factual grounding remain superior to GPT-3.5 in clinical benchmarks, reinforcing the value of scale and alignment in reducing hallucination — the tendency of language models to generate plausible but factually incorrect statements. This is a major concern for medical assistants, as Zakka et al. (2024) discuss explicitly in the context of retrieval augmentation.

---

## 4. Section C: Domain-Adapted Models for Biomedical and Clinical NLP

### 4.1 BioBERT

Lee et al. (2020) adapted BERT by continued pre-training on 4.5 billion tokens from PubMed abstracts and PubMed Central full-text articles, producing BioBERT. Fine-tuned BioBERT significantly outperforms general BERT on biomedical named entity recognition (+0.62% F1), relation extraction (+2.80% F1), and question answering (+12.24% MRR), demonstrating that domain-specific pre-training yields representations that capture the specialised vocabulary and entity relationships of biomedical text — drug names, dosages, disease ontologies — more effectively than general corpora. For a medical QA chatbot, BioBERT-style representations provide a practical path to improving answer retrieval precision in the RAG component.

### 4.2 ClinicalBERT

Huang et al. (2019) further specialised BERT by continuing pre-training on 880 million words from MIMIC-III clinical notes, producing ClinicalBERT. Unlike PubMed literature targeted by BioBERT, clinical notes contain abbreviated, colloquial, and structurally inconsistent text reflecting real-world clinical documentation practices. ClinicalBERT outperforms BioBERT on 30-day readmission prediction from discharge summaries, demonstrating that the specific corpus — literature versus clinical practice — materially affects model performance on patient-facing tasks. The implication for this thesis is that a production medical chatbot should employ models pre-trained on clinical dialogue data rather than biomedical literature.

### 4.3 Clinical Database: MIMIC-III

Johnson et al. (2016) describe the MIMIC-III Clinical Database, a freely available resource comprising de-identified health data from over 40,000 ICU patients at Beth Israel Deaconess Medical Center. MIMIC-III includes clinical notes, vital signs, laboratory results, and procedure codes spanning 2001–2012. It is the primary training corpus for ClinicalBERT and a widely used benchmark for clinical NLP research. Its availability under open access has been instrumental in democratising clinical AI research, enabling the development of the domain-specific models reviewed in this section. While this thesis uses MedQuAD rather than MIMIC-III for training, MIMIC-III contextualises the type of clinical language that future extensions of this work would need to handle.

---

## 5. Section D: Retrieval-Augmented Generation Frameworks

### 5.1 Retrieval-Augmented Generation

Lewis et al. (2020) introduced RAG as a general framework for knowledge-intensive NLP tasks, combining a dense passage retriever (DPR) with a seq2seq generator (BART). At inference, the retriever fetches the top-k relevant passages from a non-parametric external knowledge base, and the generator conditions its output on both the query and the retrieved evidence. In open-domain question answering, RAG outperforms purely parametric models on NaturalQuestions and TriviaQA, showing that externalising knowledge into a searchable index reduces the memorisation burden on the generator and enables up-to-date answers. This flexibility is of paramount importance in medicine, where guidelines change and patient-specific records must be incorporated dynamically.

### 5.2 REALM: Retrieval-Augmented Pre-training

Guu et al. (2020) presented REALM, which integrates retrieval into the pre-training stage itself. The REALM model learns what documents to retrieve by back-propagating through the retrieval step, resulting in a jointly-trained retriever and language model. Compared to Lewis et al.'s (2020) inference-time retrieval, REALM achieves stronger performance on open-domain QA but requires substantially more training compute. For the medical domain, where retrieval quality directly impacts answer safety, the argument for jointly-trained retrieval is compelling; however, the practical resource constraints of most research settings — including this thesis — favour the simpler inference-time RAG approach.

### 5.3 Efficient Similarity Search with FAISS

Johnson et al. (2017) introduced FAISS (Facebook AI Similarity Search), an open-source library for efficient similarity search over billion-scale dense vector collections using GPU acceleration. FAISS implements approximate nearest-neighbour algorithms — including product quantisation and inverted file indices — that achieve sub-linear query times without unacceptable accuracy degradation. In the context of RAG, FAISS serves as the retrieval index over pre-encoded medical passages; its efficiency determines the latency of each chatbot response. This thesis uses FAISS as the vector index for the RAG component, with the MedQuAD answer corpus encoded into a flat L2 index.

### 5.4 Sentence-BERT: Semantic Sentence Embeddings

Reimers and Gurevych (2019) proposed Sentence-BERT (SBERT), a modification of BERT employing a siamese network architecture and pooling operations to produce fixed-size sentence embeddings suitable for semantic similarity comparison via cosine distance. SBERT reduces the computational complexity of comparing sentence pairs from O(n²) BERT inference calls to a single O(n) encoding phase plus fast vector search. The all-MiniLM-L6-v2 model, a distilled SBERT variant, is used in this thesis as the retrieval encoder for the RAG pipeline, balancing embedding quality with inference speed — a practically important trade-off in a real-time medical chatbot.

---

## 6. Section E: Large Language Models in Healthcare

### 6.1 Med-PaLM and Clinical Knowledge

Singhal et al. (2023) demonstrated that large language models can encode substantial clinical knowledge, achieving expert-level performance on US medical licensing exam questions using Med-PaLM, a variant of PaLM fine-tuned with instruction prompting and retrieval augmentation. The authors reported that retrieval augmentation significantly reduces hallucination compared to closed-book generation, reinforcing the safety argument for RAG in clinical deployment. Med-PaLM's performance on clinical reasoning benchmarks was also shown to improve with model scale, consistent with the emergent capabilities reported by Brown et al. (2020) for general language models.

### 6.2 Almanac: Evidence-Grounded Clinical Responses

Zakka et al. (2024) developed Almanac, a RAG system that augments clinical responses with citations to source documents from medical guidelines and literature, enabling clinicians to verify the evidential basis of each recommendation. Unlike Med-PaLM, which emphasises benchmark performance, Almanac prioritises clinical safety and transparency — properties that are arguably more important than raw accuracy for deployment in real healthcare settings. This distinction between optimising for benchmark scores versus clinical utility is a recurring tension in the literature and motivates the hallucination rate metric included in this thesis's evaluation framework.

### 6.3 ChatDoctor: Fine-Tuning for Medical Dialogue

Li et al. (2023) adapted the LLaMA foundation model (Touvron et al., 2023) using a dataset of 100,000 real patient–doctor conversations from HealthCareMagic, plus 10,000 cases from iCliniq, producing ChatDoctor. The resulting model generates syntactically fluent and clinically coherent responses to medical questions in a dialogue context. ChatDoctor represents the fine-tuning paradigm at scale: rather than fine-tuning an LSTM on a comparatively small dataset, it employs a large pre-trained transformer. The ChatDoctor paper is closest in spirit to the LSTM fine-tuning arm of this thesis and serves as a key comparison point, though the compute requirements of LLaMA fine-tuning are far beyond those of a BiLSTM trained on 3,000 MedQuAD samples.

### 6.4 LLaMA: Efficient Open-Source Foundation Models

Touvron et al. (2023) introduced LLaMA, a family of open-source foundation language models ranging from 7B to 65B parameters trained on over one trillion tokens of publicly available text. LLaMA-13B outperforms GPT-3 (175B) on many benchmarks, demonstrating that smaller models trained on carefully curated data can outperform much larger closed models. LLaMA's open-source availability was critical for downstream medical fine-tuning work such as ChatDoctor, enabling researchers to adapt state-of-the-art generative capabilities without proprietary API dependencies.

### 6.5 InstructGPT: Alignment via Human Feedback

Ouyang et al. (2022) introduced InstructGPT, demonstrating that GPT-3 models fine-tuned using reinforcement learning from human feedback (RLHF) produce outputs that human evaluators significantly prefer over those of a 100× larger vanilla GPT-3 model. In the medical domain, alignment with human preferences is directly relevant to response quality: a clinically accurate answer must also be appropriately structured, empathetic, and free from harmful or alarming language. The RLHF framework has since been applied in Med-PaLM and underlies the instruction-following capabilities of the Gemini model used in this thesis's RAG pipeline.

---

## 7. Section F: Medical Dialogue and Question-Answering Systems

### 7.1 MedQuAD: The Medical Question-Answer Dataset

Ben Abacha and Demner-Fushman (2019) proposed the MedQuAD dataset, comprising 47,457 question-answer pairs curated from 12 National Institutes of Health websites covering topics including symptoms, treatments, and risk factors. The authors applied a question-entailment approach to answer selection, using semantic entailment to match consumer health questions to curated NIH answers without the need for end-to-end generation fine-tuning. MedQuAD's breadth of coverage across 37 question types — from genetic conditions to drug interactions — makes it an ideal benchmark for evaluating both retrieval-based and generation-based medical QA systems. It is the primary dataset used in this thesis.

### 7.2 LaMDA: Language Models for Dialogue Applications

Thoppilan et al. (2022) introduced LaMDA, a family of Transformer-based dialogue models pre-trained on 1.56 trillion words of public dialogue data and web text. LaMDA is trained with safety and factual grounding as explicit objectives, using annotated quality, safety, and groundedness labels alongside supervised fine-tuning. The LaMDA architecture demonstrates that dialogue-specific pre-training — as opposed to general language modelling — produces conversational agents with substantially improved coherence, informativeness, and safety scores in human evaluations. For medical chatbots, the grounding objective — ensuring that generated claims can be traced to external sources — is directly analogous to the citation mechanism of Almanac (Zakka et al., 2024).

### 7.3 BlenderBot: Blended Conversational Skills

Roller et al. (2021) described the design of BlenderBot, an open-domain chatbot trained to blend personality, empathy, knowledge, and engagement in multi-turn dialogue. The study demonstrated that scaling model parameters and providing training data targeting specific conversational skills produces chatbots that human evaluators prefer significantly over prior systems including DialoGPT. A key finding for this thesis is that long-term conversational coherence — simulating personalisation by retaining information across turns — requires explicit memory mechanisms, not merely scaling. This motivates the user profile personalisation layer implemented in both the LSTM and RAG systems.

### 7.4 Knowledge Graph-Based Medical QA

Huang et al. (2021) developed a knowledge-graph-based question-answering method for the medical domain, achieving an 81% matching accuracy on a clinical QA benchmark. The system links entities in natural language questions to nodes in a medical knowledge graph and traverses the graph to retrieve evidence-based answers. This approach is architecturally distinct from both the pure LSTM and RAG paradigms — it does not rely on vector similarity but on symbolic graph traversal — and thus represents a third paradigm not evaluated in this thesis but relevant to the research gap discussion. Knowledge-graph-enhanced RAG is an active research direction that could improve factual precision beyond what dense retrieval alone achieves.

---

## 8. Section G: Evaluation Metrics for Natural Language Generation

### 8.1 BLEU

Papineni et al. (2002) proposed BLEU (Bilingual Evaluation Understudy), the first widely adopted automatic metric for machine translation evaluation. BLEU computes a modified n-gram precision between a hypothesis and one or more reference translations, penalising overly short hypotheses with a brevity penalty. While originally designed for translation, BLEU has been widely applied to dialogue and QA evaluation. Its key limitation — that it measures surface-form n-gram overlap rather than semantic equivalence — is well-documented; a medical response can be clinically correct while receiving a low BLEU score if the phrasing differs from the reference. This thesis uses BLEU-4 as a standard baseline metric in combination with semantically-aware metrics.

### 8.2 ROUGE

Lin (2004) introduced ROUGE (Recall-Oriented Understudy for Gisting Evaluation), a family of metrics designed primarily for summarisation evaluation. ROUGE-L, the longest-common-subsequence variant, has become particularly popular for dialogue and QA evaluation because it captures recall and structural similarity simultaneously. Unlike BLEU, ROUGE-L penalises missing content, making it sensitive to completeness — an important property for medical QA where omitting critical information (e.g., drug contraindications) could have safety implications. This thesis uses ROUGE-L as its primary overlap-based metric.

### 8.3 BERTScore

Zhang et al. (2020) proposed BERTScore, which computes token-level semantic similarity between hypothesis and reference by matching contextual BERT embeddings using a greedy token alignment. BERTScore consistently correlates more strongly with human judgements of quality than BLEU and ROUGE across translation, summarisation, and adversarial paraphrase tasks. For medical QA specifically, where paraphrasing is common — "myocardial infarction" versus "heart attack" — BERTScore's semantic sensitivity is crucial. This thesis reports BERTScore F1 (using the `microsoft/deberta-xlarge-mnli` backbone) as its primary quality metric alongside ROUGE-L.

---

## 9. Section H: Data Infrastructure, Methodology, and Research Positioning

### 9.1 MIMIC-III Clinical Database

As noted in Section 4.3, Johnson et al. (2016) created the MIMIC-III Clinical Database. Its 40,000+ de-identified patient records represent the largest open clinical corpus available and have enabled the development of ClinicalBERT (Huang et al., 2019) and numerous clinical NLP benchmarks. The existence of MIMIC-III demonstrates the feasibility of building open, reproducible research infrastructure for clinical AI — a principle this thesis observes by using the publicly available MedQuAD dataset and releasing all code and results.

### 9.2 CRISP-DM Research Methodology

Wirth and Hipe (2000) formalised CRISP-DM (Cross-Industry Standard Process for Data Mining), a six-phase iterative methodology comprising business understanding, data understanding, data preparation, modelling, evaluation, and deployment. CRISP-DM has been adopted in over 40% of industry data science projects according to surveys, and its structured iterative approach is well-suited to the comparative evaluation task in this thesis, where multiple modelling pipelines must be built, evaluated, and refined in parallel. The research design in Section 3 of this thesis follows the CRISP-DM lifecycle.

---

## 10. Synthesis and Research Gap

The reviewed literature reveals three important patterns. First, the LSTM-based fine-tuning paradigm excels in low-latency, offline deployment: once trained, BiLSTM models require no external retrieval and return responses in milliseconds. However, they are brittle to out-of-distribution queries and cannot incorporate new knowledge without retraining, as Rajkomar et al. (2018) and Li et al. (2023) both acknowledge. Second, RAG systems address knowledge currency and hallucination, with Singhal et al. (2023) and Zakka et al. (2024) providing compelling medical evidence for retrieval-augmented safety. However, retrieval latency and dependence on retrieval quality limit RAG's applicability in time-critical or connectivity-constrained clinical environments. Third, hybrid approaches — combining parametric sequence encoders with non-parametric retrieval — remain underexplored; the closest existing work (Lewis et al., 2020; Guu et al., 2020) does not evaluate these approaches in a personalised medical dialogue setting.

The central research gap this thesis addresses is the absence of a controlled, metric-consistent comparison between LSTM fine-tuning and RAG on the same medical QA dataset with identical personalisation mechanisms. Prior studies either develop a single system without a rigorous baseline (ChatDoctor, Almanac) or compare retrieval augmentation to non-retrieval in a general-domain setting (Lewis et al., 2020). The Hybrid (BiLSTM + Gemini) system proposed in this thesis is novel in using the LSTM's retrieval output as a grounding signal for the Gemini generator, combining the domain specificity of fine-tuned sequence models with the generative fluency of large language models.

---

## 11. Conclusion

This literature review has examined thirty foundational and recent works spanning LSTM sequence modelling, Transformer pre-training, domain-adapted biomedical language models, retrieval-augmented generation, medical dialogue systems, and NLG evaluation metrics. Taken together, these works establish the theoretical basis for each component of this thesis's three-system comparison and confirm the existence of an empirical gap in the literature: no prior study directly compares LSTM fine-tuning, RAG, and a hybrid integration of the two for personalised medical question answering under controlled evaluation conditions. The experimental work in this thesis aims to fill this gap, providing a quantitative and reproducible answer to the question of which approach best serves the needs of a personalised AI health assistant.

---

## References

1. Ben Abacha, A. and Demner-Fushman, D. (2019) 'A question-entailment approach to question answering', *BMC Bioinformatics*, 20(511). doi:10.1186/s12859-019-3119-4.

2. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. and Agarwal, S. (2020) 'Language Models are Few-Shot Learners', *Advances in Neural Information Processing Systems (NeurIPS)*, 33, pp. 1877–1901.

3. Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding', *Proceedings of the 2019 Conference of the NAACL: Human Language Technologies*, Volume 1, pp. 4171–4186. Association for Computational Linguistics.

4. Guu, K., Lee, K., Tung, Z., Pasupat, P. and Chang, M.W. (2020) 'REALM: Retrieval-Augmented Language Model Pre-Training', *Proceedings of the International Conference on Machine Learning (ICML)*. PMLR.

5. Hochreiter, S. and Schmidhuber, J. (1997) 'Long Short-Term Memory', *Neural Computation*, 9(8), pp. 1735–1780. doi:10.1162/neco.1997.9.8.1735.

6. Huang, K., Altosaar, J. and Ranganath, R. (2019) 'ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission', *arXiv preprint* arXiv:1904.05342.

7. Huang, X., Zhang, J., Xu, Z., Ou, L. and Tong, J. (2021) 'A knowledge graph based question answering method for medical domain', *PeerJ Computer Science*, 7, e667. doi:10.7717/peerj-cs.667.

8. Johnson, A.E.W., Pollard, T.J., Shen, L., Li-Wei, H.L., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L.A. and Mark, R.G. (2016) 'MIMIC-III, a freely accessible critical care database', *Scientific Data*, 3, p. 160035. doi:10.1038/sdata.2016.35.

9. Johnson, J., Douze, M. and Jégou, H. (2017) 'Billion-scale similarity search with GPUs', *arXiv preprint* arXiv:1702.08734. (Published in *IEEE Transactions on Big Data*, 2019.)

10. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C.H. and Kang, J. (2020) 'BioBERT: a pre-trained biomedical language representation model for biomedical text mining', *Bioinformatics*, 36(4), pp. 1234–1240. doi:10.1093/bioinformatics/btz682.

11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020) 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', *Advances in Neural Information Processing Systems (NeurIPS)*, 33, pp. 9459–9474.

12. Li, Y., Li, S., Wang, S., Zhang, J., Jiang, H., Ma, Y., You, Y., Zhong, Z. and Zhang, H. (2023) 'ChatDoctor: A Medical Chat Model Fine-Tuned on LLaMA Model Using Medical Domain Knowledge', *arXiv preprint* arXiv:2303.14070.

13. Lin, C.Y. (2004) 'ROUGE: A Package for Automatic Evaluation of Summaries', *Proceedings of the ACL Workshop: Text Summarization Branches Out*, pp. 74–81. Association for Computational Linguistics.

14. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. and Dean, J. (2013) 'Distributed Representations of Words and Phrases and their Compositionality', *Advances in Neural Information Processing Systems (NeurIPS)*, 26.

15. OpenAI (2023) 'GPT-4 Technical Report', *arXiv preprint* arXiv:2303.08774.

16. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. and Schulman, J. (2022) 'Training language models to follow instructions with human feedback', *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

17. Pampari, A., Raghavan, P., Liang, J. and Peng, J. (2018) 'emrQA: A Large Corpus for Question Answering on Electronic Medical Records', *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 2357–2368.

18. Papineni, K., Roukos, S., Ward, T. and Zhu, W.J. (2002) 'BLEU: a Method for Automatic Evaluation of Machine Translation', *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 311–318.

19. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I. (2019) 'Language Models are Unsupervised Multitask Learners', *OpenAI Blog*, 1(8).

20. Rajkomar, A., Oren, E., Chen, K., Dai, A.M., Hajaj, N., Hardt, M., Liu, P.J., Liu, X., Marcus, J., Sun, M. and Sundberg, P. (2018) 'Scalable and accurate deep learning with electronic health records', *NPJ Digital Medicine*, 1(18). doi:10.1038/s41746-018-0029-1.

21. Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks', *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982–3992.

22. Roller, S., Dinan, E., Goyal, N., Ju, D., Williamson, M., Liu, Y., Xu, J., Ott, M., Shuster, K., Smith, E.M., Boureau, Y.L. and Weston, J. (2021) 'Recipes for Building an Open-Domain Chatbot', *Proceedings of the 16th Conference of the EACL: Main Volume*, pp. 300–325.

23. Sherstinsky, A. (2020) 'Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network', *Physica D: Nonlinear Phenomena*, 404, p. 132306. doi:10.1016/j.physd.2019.132306.

24. Singhal, K., Azizi, S., Tu, T., Mahdavi, S.S., Wei, J., Chung, H.W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S. and Payne, P. (2023) 'Large Language Models Encode Clinical Knowledge', *Nature*, 620, pp. 172–180. doi:10.1038/s41586-023-06291-2.

25. Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.T., Jin, A., Bos, T., Baker, L., Du, Y. and Li, Y. (2022) 'LaMDA: Language Models for Dialog Applications', *arXiv preprint* arXiv:2201.08239.

26. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F. and Rodriguez, A. (2023) 'LLaMA: Open and Efficient Foundation Language Models', *arXiv preprint* arXiv:2302.13971.

27. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) 'Attention Is All You Need', *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

28. Wirth, R. and Hipe, J. (2000) 'CRISP-DM: Towards a Standard Process Model for Data Mining', *Proceedings of the Fourth International Conference on the Practical Application of Knowledge Discovery and Data Mining*, pp. 29–39.

29. Zakka, C., Shad, R., Chaurasia, A., Dalal, A.R., Kim, J.L., Moor, M., Fong, R., Phillips, A., Rodman, A., Wu, M. and Rajpurkar, P. (2024) 'Almanac — Retrieval-Augmented Language Models for Clinical Medicine', *NEJM AI*, 1(2). doi:10.1056/AIoa2300068.

30. Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q. and Artzi, Y. (2020) 'BERTScore: Evaluating Text Generation with BERT', *Proceedings of the 8th International Conference on Learning Representations (ICLR)*, Addis Ababa.
