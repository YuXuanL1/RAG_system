# RAG_system

## Introduction
This project is Comprehensive RAG competition hosted by META on the [AIcrowd website](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/meta-kdd-cup-24-crag-retrieval-summarization). There are three task in this competition, but we only focus on task 1. In this task, we are provided with up to five web pages for each question. While these web pages are likely, but not guaranteed, to be relevant. The objective of this task is to evaluate the answer generation capabilities of the RAG (Retrieval-Augmented Generation) systems.

## Tasks
- Analyze the provided web pages and remove irrelevant content.
- Implement at least one of the following retrieval models:
  - Vector Space Model (VSM)
  - Language Model (LM)
- Combining predictions from multiple models, such as TF-IDF, BM25, or JM-smoothing. 
- Use LLM to generate answers based on the retrieved information.
- Adopt machine learning toolkits to create a classifier that judges whether the retrieved information is relevant to the query.
- Experiment with different LLMs and settings, such as quantization to optimize performance and efficiency.
- Fine-tune the LLM on relevant datasets to further enhance its accuracy and adaptability for this specific task.

## Illustration of Framework
![螢幕擷取畫面 2025-03-16 160725](https://github.com/user-attachments/assets/5083161f-6643-4063-a817-0e1c644e447d)

## Methodology
Our approach is primarily based on modifications to the official rag_llama_baseline.py. The
following process description represents the method that achieved the highest score among
our multiple submissions.
1. **Preprocessing**

    - **HTML Parsing**  
      Each webpage is parsed to remove HTML tags and structure, retaining only the semantic text.
    
    - **Sentence Splitting**  
      Text is segmented into sentences using **Blingfire**, with each sentence treated as an individual chunk.
    
    - **Chunk Length Control**  
      To balance semantic integrity and efficiency, each chunk is limited to **≤ 2000 characters**.
    
    - **Embedding Computation**  
      All chunks and user queries are embedded using **SentenceTransformer** with the `all-MiniLM-L6-v2` model.

2. **Cosine Similarity**

    In the text retrieval and initial filtering phase, we first sort the text chunks based on cosine
    similarity and select the top 30 chunks with the highest scores as the candidate set for initial
    filtering. These text chunks are considered the most relevant to the query and will be used in
    subsequent processing steps.

3. **Reranking**

    - **CrossEncoder Scoring**  
      Use a **CrossEncoder model** to rescore the initially retrieved text chunks, capturing deeper semantic relationships between the query and content.
    
    - **Score Thresholding**  
      Only retain chunks with a **similarity score > 7.8**, effectively filtering out irrelevant or low-quality results.
    
    - **Top-k Selection**  
      Select the **top 10 highest-scoring chunks** to ensure both precision and efficiency.
    
    - **Purpose**  
      This step enhances semantic matching accuracy and prepares a high-quality input set for the **generation phase** with the LLM.

4. **Formatting Prompts**

    - **Query-wise Prompt Construction**  
      For each query, build a prompt using its top-ranked candidate chunks from the reranking step.
    
    - **Reference Chunk Inclusion**  
      Valid text chunks are included as **"reference documents"**, with ordered tags like `<DOC rank=1>`, `<DOC rank=2>`, etc.
    
    - **Context Length Control**  
      The total character count of context chunks is **capped at 6,000 characters** to meet LLM input constraints. Excess text is truncated.
    
    - **Fallback Handling**  
      If no valid chunks are retrieved, the prompt includes `"No Reference"` and instructs the LLM to respond with `"I don’t know."`
    
    - **Goal**  
      This structured format helps the LLM interpret chunk importance and generate more **context-aware, precise answers**.

5. **Generating Answers**

    - **System Prompt Components**  
      - Includes the **query**, its **timestamp (`query_time`)**, and **ranked reference documents**
      - Reference chunks are tagged (`<DOC rank=1>`, etc.) and ordered by importance
      - If no references are available, `"No Reference"` is inserted and the LLM is instructed to reply: `"I don’t know."`
    
    - **Instruction Constraints**  
      - Answers must be based **only on reference documents**
      - Limit responses to **≤ 50 words**
      - Avoid speculation or uncertain language
      - If no answer can be derived, explicitly reply `"I don’t know."`
    
    - **Prompt Formatting Tool**  
      Structured using the built-in `apply_chat_template()` function to fit LLM input format
    
    - **LLM Answer Generation**
    
      - **Model Used**: `Llama-3.2-1B-Instruct`
      - **Inputs**: Structured prompt + filtered reference documents + strict generation instructions
    
    - **Decoding Configuration**
      - `temperature = 0.1`: Reduces randomness, increases precision
      - `top_p = 0.9`: Allows sampling from the top 90% of token probabilities, ensuring diversity without sacrificing accuracy
    
    - **Goal**  
      Produce **factually grounded, concise, and accurate answers** strictly based on retrieved content.

6. **Post-Processing**

    After generating the answers, the system further inspects and processes the results to ensure
    that the final outputs meet the query requirements and maintain high reliability. During
    testing, we observed an increase in hallucinated responses when queries involved
    mathematical calculations. To mitigate this, we implemented a simple rule: when the query
    contains the word "average," the corresponding answer is directly set to "I don’t know."

## Results
We achieved continuous improvement with each attempt, and the above process represents
our highest-scoring approach, achieving a score of 0.003. (increased accuracy by 50%)

Our preprocessing method, while fast and convenient, also comes with some significant
drawbacks. 
- **Simplified Chunking Strategy**  
  - Used **only single-layer chunking**, lacking hierarchical context modeling
  - Did not implement **two-stage retrieval** (Child → Parent), which likely reduced chunk relevance and coherence

- **No External Knowledge Integration**  
  - Relied solely on the provided web search results
  - Lacked **public datasets** (e.g., Wikipedia, domain corpora), limiting performance on **background-knowledge-heavy queries**

> 💡 **Future Work**: Incorporating hierarchical retrieval, external datasets, and domain-adaptive prompting could significantly enhance answer quality and contextual relevance.


## Report
[WSM Group4 Report.pdf](https://github.com/user-attachments/files/19270001/WSM.Group4.Report.pdf)

