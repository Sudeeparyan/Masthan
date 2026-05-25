
# Dynamic Healthcare Personalization in AI Assistants: A Comparative Analysis of LSTM Fine-Tuning versus Retrieval-Augmented Generation

**MSc in Artificial Intelligence** Mastan Vali Shaik  
24226807  
National College of Ireland  

## Abstract
The aim of this study is to develop a personalised AI medical assistant which tracks a user's health status through conversation. This research will explore two ways to build the medical assistant: building a Long Short-Term Memory model with medical dialogue data, and building a Retrieval-Augmented Generation model which uses medical information retrieval to generate responses. The methods will be compared by the quality of the medical responses, the accuracy of the personalisation and the response time through an open medical question-answer data set. This study will establish which system is more effective for giving health advice, and offer insights on building accessible and personalised AI chatbots for health.

**Keywords:** LSTM, Retrieval-Augmented Generation, Healthcare Chatbot, Personalisation, Deep Learning

## 1. Introduction
Personal health care can be challenging. Many adults with chronic conditions such as diabetes, heart and lung disease, find it hard to manage their symptoms, medicines and lifestyle modifications. Our current health-care system revolves around seeing the doctor, but there's a need to fill the space in between appointments. A chatbot which can answer questions, learn from previous conversations and provide tailored advice can help fill this gap to manage patients between visits.

With the latest development in deep learning and natural language processing, there are two ways to train such a chatbot. One is fine-tuning, where a neural network is trained on a medical dataset so that it learns medical facts through the weights of the neural network. Long Short-Term Memory (LSTM) networks (Hochreiter and Schmidhuber, 1997) is a type of neural network that can learn from time series. A medical assistant with LSTM would learn facts from medical dialogues during the training phase and use all the facts it knows to respond to a question during the inference phase. The benefit of this approach is that all facts are memorised and the answer is quick. But it's expensive to change knowledge and may produce incorrect responses if the facts it knows are different from the facts it sees.

The other approach is Retrieval-Augmented Generation (RAG) by Lewis et al. (2020) which decouples knowledge and generation. The RAG system doesn't know facts, but at run-time it searches a knowledge base, and uses the most relevant documents to generate responses using a language model. This allows the system to access the most recent medical guidelines, medical history of the patient and medication interactions, without the need to retrain the language model. But it slows down the system and the quality of the response is dependent on the quality of the documents that are retrieved.

The two paradigms have been explored separately in health care. Rajkomar et al. (2018) demonstrated the use of deep learning models to predict events from electronic health record data and Singhal et al. (2023) showed that large language models with retrieval can perform at an expert level in medical reasoning tasks. But there is limited research directly comparing these two approaches with identical metrics, on the same use case for health care. This is important for developers of health assistants to understand which model is suitable for their tasks.

Our research will answer the question: "What is the performance of fine-tuned LSTM versus the Retrieval-Augmented Generation (RAG) system, in terms of accuracy, personalisation and efficiency, for developing a personalised AI health assistant?"

We will answer this question by building both of the systems using the same health question-answer dataset. The LSTM system will use health dialogues to learn to generate responses. The RAG system will use the same health information to build a vector database and retrieve information to generate responses. The approaches will be compared using the same metrics.

This study has three contributions. First, it offers a practical way of comparing fine-tuning vs retrieval-based methods for health personalisation. Second, it offers a mixed perspective on the benefits and drawbacks of each approach for particular types of health questions. Third, it provides a method for researchers to evaluate new methods of health assistants.

The rest of this paper is structured as follows. Section 2 discusses the existing studies of LSTM for clinical applications, RAG in medical domain, and studies comparing the two methods. Section 3 provides details on the research method of this study, including the data set, system designs, evaluation strategy, ethical considerations and project timelines.

## 2. Literature Review

### A. LSTM and Recurrent Networks in Medical Case
Clinical data is often temporal and recurrent neural networks (RNNs) are commonly applied to model this data, whether it is time-series or text. Hochreiter and Schmidhuber (1997) developed the LSTM network to overcome the vanishing gradient problem that existed in previous recurrent networks. Gating enables the network to keep information from the early part of the data sequence, and to remove information from later parts of the sequence, which is useful for medical data, where symptoms in the early part of a patient's medical history can influence their diagnosis later in the sequence.

While the paper by Hochreiter and Schmidhuber was theoretical, a recent review of the basics and practical use of LSTM was provided by Sherstinsky (2020). While Hochreiter and Schmidhuber described the gating mechanisms from a theoretical perspective, Sherstinsky reviewed the practical configuration and training of LSTM networks in various applications, and highlighted that the hyperparameters (e.g. size of the hidden layer and dropout rate) play an important role in clinical prediction tasks. Both papers suggest that LSTM is still a good approach for clinical time series, but Sherstinksy's paper suggests that it's not as easy as using LSTM.

These insights were further built upon by Rajkomar et al. (2018), who applied deep learning models, such as recurrent networks, for predicting in-hospital death, unexpected readmission and long length of stay with electronic health records of two large. They showed that deep models using raw clinical data are at least as effective as clinical scoring systems. Similarly, in another medical application, Li et al. (2023) adapted a large language model to medical dialogues to create ChatDoctor, a model that given medical questions, can respond with medical answers.

Rajkomar et al. and Li et al. worked on structured prediction tasks (such as predicting a score) and Li et al. worked on the more difficult task of generating free-text medical responses. These two studies show that it is beneficial to train models on medical data, but also that it's difficult to adapt models to new medical knowledge. This is a critical issue in medicine, where medical knowledge and practices are constantly evolving, and drives us to investigate the use of retrieval-based models that can use the latest medical knowledge at inference.

### B. Retrieval-Augmented Generation in Healthcare
Retrieval-based approaches for accessing external knowledge in language models have recently become more popular to avoid fine-tuning. At the NeurIPS, Lewis et al. (2020) proposed the RAG model to demonstrate how by using a dense passage retriever and a sequence-to-sequence generator, it is possible to generate more factual answers that can be easily updated compared to answers generated by a model that relies solely on model's pre-trained knowledge (CORE Rank: A*).

In contrast, Guu et al. (2020) developed the REALM model, which pre-trains a retrieval-augmented language model to use retrieval during pre-training (ICML, CORE Rank: A*). Lewis et al. used retrieval at inference time with a pre-trained retriever, whereas Guu et al. used retrieval during training to be able to learn what documents to retrieve. While both Lewis et al.'s and Guu et al.'s methods outperform non-retrieval methods in knowledge-intensive tasks, they are different in terms of simplicity: Lewis et al.'s method is simpler to implement and to modify for different tasks and domains, and is therefore more convenient for the medical domain, where resources are limited.

In the case of the medical domain, Singhal et al. (2023) built Med-PaLM, a large language model that generated expert-quality responses on medical board exams, by training the model with retrieval of clinical evidence They show that retrieval helps to reduce hallucination, that is, language models' tendency to generate plausible but false medical advice. On the other hand, Zakka et al. (2024) tackled the hallucination problem by developing a clinical language model called Almanac which augments the advice generated with a link to the medical evidence, enabling the clinician to check where the advice comes from. While Singhal et al. looked at medical benchmarks to assess medical accuracy, Zakka et al. focused on medical safety and reliability, by ensuring that the advice provided is documented. These two pieces of research show that retrieval-augmented systems are a safer option to use in clinical practice than generative models, as they offer an avenue for investigation.

### C. Fine-Tuning vs. Retrieval
There have been some preliminary comparisons of fine-tuning to retrieval-based systems. Vaswani et al. (2017) were the first to introduce the Transformer architecture used in most modern language models, which enables attention that is used in both fine-tuned and retrieval-based models. Their attention mechanism enables a model to calculate the importance of different tokens in the source, which is pertinent to this work because the fine-tuning approach (using LSTMs) and RAG approach both involve different pieces of health information. LSTMs have been largely superseded by Transformers for many natural language processing applications, but they are still incredibly effective with smaller datasets and low-resource scenarios, as is the case here.

In particular, Ben Abacha and Demner-Fushman (2019) tackled a medical question answering task by using retrieval and textual entailment on the MedQuAD medical question-answer pairs from popular health websites. They first used retrieval from a medical knowledge base to find potential answers, and then used a neural model to select the best one. This paper is significant because it shows a "retrieval first" approach to medical question answering can be achieved without the high cost of fine-tuning a model. However, Li et al. (2023) have demonstrated that fine-tuning of a sufficiently large language model with a sufficient amount of medical dialogue data can also generate good medical QA without retrieval. The proposed research will address the trade-off between the two (RAG and fine-tuned) strategies.

### D. Research Niche
There are studies that demonstrate the effectiveness of fine-tuning and RAG in health care. However, there is a lack of research that compares these methods, with the same data, metrics and task. The current state of the art is to either develop a fine-tuned model and report its performance, or develop a RAG system and compare it to non-retrieval systems. This work addresses this gap by developing the two approaches to build an AI chatbot health assistant, and by comparing these approaches to answer the question: Should the developer of a personalised health AI chatbot focus on fine-tuning the chatbot's model on medical data, or develop a retrieval system that can access external medical knowledge? The goal is a quantitative answer to this question, following a performance analysis with several metrics.

## 3. Research Method and Specification

### A. Research Method
We employ the cross-industry standard process for data mining (CRISP-DM, Wirth and Hipe, 2000), a widely adopted process model for data mining projects, which consists of six steps: business understanding, data understanding, data preparation, modelling, evaluation and deployment.

**Data Selection and Understanding.** The study will use the MedQuAD dataset (Ben Abacha and Demner-Fushman, 2019) which is an open access set of medical question-answer pairs from reliable sources such as the National Institutes of Health and the National Library of Medicine. This dataset contains around 47,000 question-answer pairs from a wide variety of health topics such as symptoms, treatment, prevention and risk factors. We will choose a sample of 5,000 pairs that are representative of questions about daily health management such as information on the treatment of chronic diseases, drugs, diet and other lifestyle factors.

**Data Preparation.** We will preprocess the question-answer pairs with some standard text processing: lowercasing, tokenisation with a tokeniser fine-tuned for medical language, cleaning them of artefacts (quotes, etc.) and split the dataset into a train set (70%) for model training, a validation set (15%) for hyperparameter optimisation and a test set (15%) for evaluation. For the LSTM approach, the questions and the answers will be mapped to sequences of numbers using word embeddings pre-trained on the medical corpus. For the RAG approach, the set of answers will be divided into passages of about 200 tokens and converted to dense vectors with a model of sentence embeddings.

**LSTM Fine-Tuning Pipeline.** The first RAG implementation will be an LSTM encoder-decoder. The encoder will convert the health question (as a sequence of embeddings) to a vector representation of the question. The decoder will generate a health answer from the encoder, word by word. We will have a two layer bidirectional LSTM encoder and two layer LSTM decoder (256 units) with attention on the encoder states. We will train our model using Adam optimiser with cross-entropy loss, learning rate 0.001 and decay schedule and stop training early on validation loss. We will also use dropout (0.3) between the layers to avoid over-fitting. Training will be done on a GPU with PyTorch.

*(Figure 1: LSTM encoder-decoder with attention mechanism that learns and remembers medical information to generate personalized responses. Components: Question, Your Profile (Age, Medicines, Health Conditions) -> LSTM Model (Learns & Remembers Medical Information) -> Answer)*

**RAG Pipeline.** Our second system will be a RAG system (Lewis et al., 2020). When the user queries the system with a health question, this question will be encoded as a vector by a sentence transformer. This vector will then be used to search a FAISS vector index of pre-encoded passages from the medical knowledge base. The most relevant k (k=5) passages will be used to construct a new prompt that will consist of the question and the passages. This will then be used to prompt a generator that will generate the answer. The retrieval system will use the all-MiniLM-L6-v2 sentence transformer as the encoder and an open-source language model as the generator that will be able to run on the given hardware.

*(Figure 2: RAG system that retrieves relevant medical passages from a knowledge base and generates evidence-based personalized responses. Components: Question -> Search Medical Knowledge Base; Your Profile (Age, Medicines, Health Conditions) -> Find Best Matching Answers -> Answer)*

**Personalization Mechanism.** Both systems will be personalised. The user profile will contain some medical data such as age, medical conditions, medications and symptoms. For the LSTM, the user profile will be a vector which will be concatenated with the vector of the question. For the RAG system, the user profile will be used to guide the search to find similar patients' profiles. This will help the system to offer specific advice to the patient.

### B. Research Resources

| Resource | Purpose |
| :--- | :--- |
| MedQuAD Dataset | Medical question-answer pairs for training and evaluation |
| Python 3.10+ | Primary programming language |
| PyTorch | LSTM model implementation and training |
| LangChain | RAG pipeline orchestration |
| FAISS | Vector similarity search for document retrieval |
| Sentence-Transformers | Text encoding for RAG retrieval |
| Hugging Face Transformers | Pre-trained language models |
| scikit-learn | Evaluation metrics and data splitting |
| Google Colab / Local GPU | Model training environment |
| Git/GitHub | Version control and reproducibility |

### C. Evaluation
* **Response Accuracy.** Measured by ROUGE-L and BLEU scores on generated responses vs ground truth responses. This is the longest common sub-sequence (ROUGE-L) and n-gram precision (BLEU).
* **Semantic Similarity.** Measured by BERTScore, which takes the meaning of the answer into account (using contextual embeddings), and identifies cases where the system produces the correct answer in a different way.
* **Personalisation Quality.** We will evaluate a sample of 100 test questions by human raters. We will request a 1-5 score for user relevance, specificity and appropriateness per response.
* **Computational Efficiency.** Time per inference (ms), model size (number of parameters, file size), memory inferences.
* **Hallucination Rate.** Measured by the proportion of responses that contain claims found in the training data and/or the documents. A medical expert will assess a random sample of 50 responses for each system.

### D. Ethical Considerations
* **Data Privacy.** MedQuAD data will be public medical data from government websites. We will not use any patient data. All personalisation profiles will be mock.
* **Medical Safety.** The AI assistant we will produce in this project is experimental and should not be used. The medical responses will be clearly labelled as NOT medical advice. In particular, hallucination rates will be used to warn against a potential for incorrect medical advice.
* **Bias and Fairness.** The MedQuAD data set is gathered from English US government health sites, and may be biased in terms of geography or language. The test user profiles for personalisation will have a variation of age, health and demographics to assess if there are differences between groups.
* **Transparency.** We will open source the code, configuration, evaluation and data. RAG is transparent about which source documents are used to answer a question.

### E. Project Plan

| Task | May 2026 | Jun 2026 | Jul 2026 | Aug 2026 | Sep 2026 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dataset collection & preprocessing | ✔ | | | | |
| Literature review refinement | ✔ | | | | |
| LSTM model design & training | | ✔ | | | |
| RAG pipeline implementation | | ✔ | | | |
| Personalisation layer | | | ✔ | | |
| Evaluation & comparison | | | ✔ | | |
| Error analysis & refinement | | | | ✔ | |
| Report writing | | | | ✔ | |
| Final review & submission | | | | | ✔ |

## 4. REFERENCES
1. Ben Abacha, A. and Demner-Fushman, D. (2019) 'A question-entailment approach to question answering', BMC Bioinformatics, 20(511). doi:10.1186/s12859-019-3119-4.
2. Guu, K., Lee, K., Tung, Z., Pasupat, P. and Chang, M.W. (2020) 'REALM: Retrieval-Augmented Language Model Pre-Training', Proceedings of the International Conference on Machine Learning (ICML).
3. Hochreiter, S. and Schmidhuber, J. (1997) 'Long Short-Term Memory', Neural Computation, 9(8), pp. 1735-1780. doi:10.1162/neco.1997.9.8.1735.
4. Lewis, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020) 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', Advances in Neural Information Processing Systems (NeurIPS), 33, pp. 9459-9474
5. Li, Y., Li, S., Wang, S., Zhang, J., Jiang, H., Ma, Y., You, Y., Zhong, Z. and Zhang, H. (2023) 'ChatDoctor: A Medical Chat Model Fine-Tuned on LLaMA Model Using Medical Domain Knowledge', arXiv preprint arXiv:2303.14070.
6. Rajkomar, A. et al. (2018) 'Scalable and accurate deep learning with electronic health records', NPJ Digital Medicine, 1(18). doi:10.1038/s41746-018-0029-1.
7. Sherstinsky, A. (2020) 'Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network', Physica D: Nonlinear Phenomena, 404, p. 132306. doi: 10.1016/j.physd.2019.132306.
8. Singhal, K. et al. (2023) 'Large Language Models Encode Clinical Knowledge', Nature, 620, pp. 172-180. doi:10.1038/s41586-023-06291-2.
9. Vaswani, A. et al. (2017) 'Attention Is All You Need', Advances in Neural Information Processing Systems (NeurIPS), 30.
10. Wirth, R. and Hipe, J. (2000) 'CRISP-DM: Towards a standard process model for data mining', Proceedings of the Fourth International Conference on the Practical Application of Knowledge Discovery and Data Mining, pp. 29-39.
11. Zakka, C. et al. (2024) 'Almanac - Retrieval-Augmented Language Models for Clinical Medicine', NEJM AI, 1(2). doi: 10.1056/Aloa2300068.
