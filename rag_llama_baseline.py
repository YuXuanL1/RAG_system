import os
import numpy as np
import torch
import ray
from collections import defaultdict
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from numpy.linalg import norm
from llama_cpp import Llama
import json
import pickle
from typing import Dict, Any, List

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

#### CONFIG PARAMETERS ---
NUM_CONTEXT_SENTENCES = 20
MAX_CONTEXT_SENTENCE_LENGTH = 1000
MAX_CONTEXT_REFERENCES_LENGTH = 4000
AICROWD_SUBMISSION_BATCH_SIZE = 8
VLLM_TENSOR_PARALLEL_SIZE = 4
VLLM_GPU_MEMORY_UTILIZATION = 0.85
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128

def cos_sim(a, b):
    """Computes cosine similarity between two vectors."""
    return (a @ b.T) / (norm(a) * norm(b))

class ChunkExtractor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)

        if not text:
            return interaction_id, [""]

        sentences = sent_tokenize(text)
        sentences = [str(sentence) for sentence in sentences]
        chunk_docs = self.text_splitter.create_documents(sentences)
        chunks = [doc.page_content for doc in chunk_docs]
        
        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        chunk_dictionary = defaultdict(list)
        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)
            chunk_dictionary[interaction_id].extend(_chunks)

        return self._flatten_chunks(chunk_dictionary)

    def _flatten_chunks(self, chunk_dictionary):
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        return np.array(chunks), np.array(chunk_interaction_ids)

import torch.nn.functional as F
from typing import List, Tuple

# class ColBERTRetriever:
#     def __init__(self, sentence_model):
#         self.sentence_model = sentence_model
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#     def _tokenize_and_embed(self, text: str) -> torch.Tensor:
#         """
#         Tokenize text and get token-level embeddings
#         Returns: tensor of shape [seq_len, embedding_dim]
#         """
#         # Encode text using sentence_transformers
#         # This gives us the token embeddings directly
#         embeddings = self.sentence_model.encode(
#             text,
#             output_value="token_embeddings",
#             convert_to_tensor=True,
#             normalize_embeddings=True
#         )
        
#         # Remove batch dimension if present
#         if len(embeddings.shape) == 3:
#             embeddings = embeddings[0]
            
#         return embeddings.to(self.device)

# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm

# class AnswerAIColBERTRetriever:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_name = "answerdotai/answerai-colbert-small-v1"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
#         self.model.eval()
        
#     def _tokenize_text(self, text: str, is_query: bool = False) -> dict:
#         """Tokenize text with special tokens for query/document"""
#         max_length = 64 if is_query else 256
#         return self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         )
        
#     def _encode_text(self, text: str, is_query: bool = False) -> torch.Tensor:
#         """
#         Encode single text using AnswerAI ColBERT
#         Returns: tensor of token embeddings [seq_len, hidden_dim]
#         """
#         # Tokenize
#         inputs = self._tokenize_text(text, is_query)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
#         # Get embeddings
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
#             # Normalize embeddings
#             embeddings = F.normalize(embeddings, p=2, dim=1)
            
#         return embeddings

#     def get_dense_scores(self, query: str, chunks: List[str]) -> torch.Tensor:
#         """
#         Compute AnswerAI ColBERT similarity scores between query and chunks
#         """
#         # Encode query
#         q_embeddings = self._encode_text(query, is_query=True)
        
#         scores = []
#         # Process chunks in batches to avoid OOM
#         batch_size = 32
        
#         # Use tqdm for progress bar
#         for i in tqdm(range(0, len(chunks), batch_size), desc="Computing ColBERT scores"):
#             batch_chunks = chunks[i:i + batch_size]
#             batch_scores = []
            
#             # Get document embeddings for batch
#             for chunk in batch_chunks:
#                 d_embeddings = self._encode_text(chunk, is_query=False)
                
#                 # Compute maximum similarity for each query token
#                 # [query_len, doc_len]
#                 sim_matrix = torch.matmul(q_embeddings, d_embeddings.transpose(0, 1))
#                 # [query_len]
#                 max_sim = torch.max(sim_matrix, dim=1)[0]
#                 # Sum over query tokens
#                 score = torch.sum(max_sim).item()
#                 batch_scores.append(score)
                
#             scores.extend(batch_scores)
            
#         return torch.tensor(scores)


# try:
#     import faiss
#     import faiss.contrib.torch_utils
# except ImportError:
#     raise ImportError(
#         "Please install FAISS with: pip install faiss-gpu  # for GPU support\n"
#         "or: pip install faiss-cpu  # for CPU only"
#     )

# class FAISSRetriever:
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         self.model = AutoModel.from_pretrained("bert-base-uncased")
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()
#         self.model.eval()
#         self.faiss_index = None
#         self.chunk_embeddings = None

#     def _get_mean_embedding(self, text: str) -> np.ndarray:
#         """Get mean pooled embedding for text using BERT"""
#         inputs = self.tokenizer(
#             text, 
#             max_length=512,
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )
#         if torch.cuda.is_available():
#             inputs = {k: v.cuda() for k, v in inputs.items()}
            
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             embeddings = outputs.last_hidden_state.mean(dim=1)
            
#         return embeddings.cpu().numpy()

#     def build_index(self, chunks: List[str]):
#         """Build FAISS index from chunks"""
#         print(f"Starting to build FAISS index for {len(chunks)} chunks...")
        
#         # Get embeddings for all chunks
#         embeddings = []
#         batch_size = 32
#         for i in tqdm(range(0, len(chunks), batch_size), desc="Computing embeddings"):
#             batch = chunks[i:i + batch_size]
#             batch_embeddings = [self._get_mean_embedding(chunk) for chunk in batch]
#             embeddings.extend(batch_embeddings)
            
#         embeddings = np.vstack(embeddings)
#         print(f"Generated embeddings with shape: {embeddings.shape}")
        
#         # Build FAISS index
#         dimension = embeddings.shape[1]
#         print(f"Creating FAISS index with dimension {dimension}")
        
#         # Initialize index with GPU if available
#         if torch.cuda.is_available():
#             print("Using GPU for FAISS")
#             res = faiss.StandardGpuResources()
#             index = faiss.IndexFlatIP(dimension)
#             self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
#         else:
#             print("Using CPU for FAISS")
#             self.faiss_index = faiss.IndexFlatIP(dimension)
        
#         # Add vectors to index
#         print("Adding vectors to FAISS index...")
#         self.faiss_index.add(embeddings)
#         self.chunk_embeddings = embeddings
#         print("FAISS index built successfully")
        
#     def get_scores(self, query: str, k: int) -> np.ndarray:
#         """Get similarity scores for query"""
#         query_embedding = self._get_mean_embedding(query)
#         scores, _ = self.faiss_index.search(query_embedding, k)
#         return scores[0]  # Return scores for first (only) query

from gensim import corpora, models
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class LSIRetriever:
    def __init__(self, num_topics=100):
        self.num_topics = num_topics
        self.dictionary = None
        self.lsi_model = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and remove stopwords"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]
        
    def build_lsi_index(self, chunks: List[str]):
        """Build LSI model from document chunks"""
        # Preprocess all documents
        processed_chunks = [self.preprocess_text(chunk) for chunk in chunks]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_chunks)
        
        # Convert documents to BoW format
        corpus = [self.dictionary.doc2bow(chunk) for chunk in processed_chunks]
        
        # Train LSI model
        self.lsi_model = models.LsiModel(
            corpus, 
            id2word=self.dictionary,
            num_topics=self.num_topics
        )
        
        # Convert corpus to LSI space
        self.lsi_corpus = self.lsi_model[corpus]
        
        return self.lsi_corpus
    
    def get_query_vector(self, query: str):
        """Convert query to LSI space"""
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Convert to BoW
        query_bow = self.dictionary.doc2bow(processed_query)
        
        # Convert to LSI space
        return self.lsi_model[query_bow]
    
    def get_similarity_scores(self, query_vec, corpus_lsi):
        """Compute cosine similarity between query and documents in LSI space"""
        # Convert query vector to dense numpy array
        query_dense = np.zeros(self.num_topics)
        for idx, score in query_vec:
            query_dense[idx] = score
            
        # Convert corpus to dense matrix
        corpus_dense = np.zeros((len(corpus_lsi), self.num_topics))
        for i, doc in enumerate(corpus_lsi):
            for idx, score in doc:
                corpus_dense[i, idx] = score
                
        # Compute cosine similarity
        similarity = np.dot(corpus_dense, query_dense) / (
            np.linalg.norm(corpus_dense, axis=1) * np.linalg.norm(query_dense)
        )
        
        return similarity

class HybridRetriever:
    # def __init__(self, sentence_model):
    #     self.sentence_model = sentence_model
    #     self.colbert_retriever = ColBERTRetriever(sentence_model)

    def __init__(self, sentence_model):
        self.sentence_model = sentence_model
        # self.colbert_retriever = AnswerAIColBERTRetriever()
        # self.faiss_retriever = FAISSRetriever()
        self.lsi_retriever = LSIRetriever(num_topics=100)
        self.lsi_corpus = None


    # def build_bm25_index(self, chunks):
    #     """Builds BM25 index from chunks."""
    #     tokenized_corpus = [word_tokenize(chunk.lower()) for chunk in chunks]
    #     return BM25Okapi(tokenized_corpus), tokenized_corpus

    # compute cos sim
    def get_dense_scores(self, query_embedding, chunk_embeddings):
        """Computes dense retrieval scores using cosine similarity."""
        # Convert the list of scores to a numpy array
        scores = np.array([cos_sim(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings])
        return scores

    def hybrid_retrieve(self, query, chunks, chunk_embeddings=None, topk=NUM_CONTEXT_SENTENCES, k=60):
        # Note: chunk_embeddings parameter is kept for compatibility but not used
        # 檢查輸入
        if chunks is None or not isinstance(chunks, (list, np.ndarray)) or len(chunks) == 0:
            raise ValueError("Chunks must be a non-empty list of strings.")
        if chunk_embeddings is None:
            raise ValueError("chunk_embeddings cannot be None. Ensure they are properly generated.")

        # Dense retrieval using cos sim
        query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)
        if query_embedding is None:
            raise ValueError("Query embedding is None. Ensure the sentence model is properly initialized and input is valid.")
        
        dense_scores = self.get_dense_scores(query_embedding, chunk_embeddings)
        dense_scores = np.array(dense_scores)

        # Dense retrieval using ColBERT
        # dense_scores = self.colbert_retriever.get_dense_scores(query, chunks)

    ###########################################################################    
        # # Sparse retrieval (BM25)
        # bm25, _ = self.build_bm25_index(chunks)
        # tokenized_query = word_tokenize(query.lower())
        # sparse_scores = bm25.get_scores(tokenized_query)

        # # Sparse retrieval (LSI)
        # Initialize LSI if not already done
        if self.lsi_corpus is None:
            try:
                self.lsi_corpus = self.lsi_retriever.build_lsi_index(chunks)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LSI corpus: {e}")
        # Sparse retrieval using LSI
        query_vec = self.lsi_retriever.get_query_vector(query)
        sparse_scores = self.lsi_retriever.get_similarity_scores(query_vec, self.lsi_corpus)
        sparse_scores = np.array(sparse_scores)
        
        # Combine scores using RRF
        dense_ranks = {i: rank for rank, i in enumerate(np.argsort(-dense_scores), 1)}
        sparse_ranks = {i: rank for rank, i in enumerate(np.argsort(-np.array(sparse_scores)), 1)}
        
        rrf_scores = []
        for idx in range(len(chunks)):
            dense_rank = dense_ranks.get(idx, len(dense_ranks) + 1)
            sparse_rank = sparse_ranks.get(idx, len(sparse_ranks) + 1)
            rrf_score = (1 / (k + dense_rank)) + (1 / (k + sparse_rank))
            rrf_scores.append(rrf_score)
        
        # Get top-k chunks by RRF score
        top_indices = np.argsort(-np.array(rrf_scores))[:topk]
        return chunks[top_indices]


import atexit
from contextlib import contextmanager

class RAGModel:
    def __init__(self):
        self._llm = None
        self._sentence_model = None
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()
        self.hybrid_retriever = HybridRetriever(self.sentence_model)
        # Register cleanup on program exit
        atexit.register(self.cleanup)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = Llama(
                model_path=self.model_name,
                n_ctx=4096,
                n_threads=8,
                verbose=False
            )
        return self._llm

    @property
    def sentence_model(self):
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

            # model_name = "dunzhang/stella_en_400M_v5"  # 替換為您希望使用的模型名稱
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # self._sentence_model = AutoModel.from_pretrained(
            #     model_name,
            #     trust_remote_code=True  # 許可執行自訂代碼
            # )
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self._sentence_model.to(self.device)
            # self._sentence_model.eval()
        return self._sentence_model

    def cleanup(self):
        try:
            if hasattr(self, '_llm') and self._llm is not None:
                self._llm.close()  # 使用適當的清理方法
                self._llm = None
        except Exception as e:
            print(f"Error during cleanup: {e}")


    def initialize_models(self):
        """Initialize model paths and configurations"""
        # self.model_name = "tensorblock/Llama-3.2-8B-Instruct-GGUF"
        self.model_path = r"C:\Users\6yx\Downloads\Llama-3.2-8B-Instruct-Q8_0.gguf"
        self.model_name = r"C:\Users\6yx\Downloads\Llama-3.2-8B-Instruct-Q8_0.gguf"

        if not os.path.exists(self.model_name):
            raise Exception(f"Model weights not found at {self.model_name}")

    @contextmanager
    def model_context(self):
        """Context manager for safe model usage"""
        try:
            yield self.llm
        finally:
            self.cleanup()


    def generate_response(self, prompt: str, max_tokens=75, temperature=0.7, top_p=0.95):
        try:
            if not prompt or not isinstance(prompt, str):
                print("Invalid prompt:", prompt)  # 檢查 prompt
                return "I don't know"

            with self.model_context() as model:
                print("Generating with prompt:", prompt)  # 檢查實際使用的 prompt
                completion = model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    echo=False,
                    stop=["<|eot_id|>", "\n\n"]
                )
                print("Raw completion:", completion)  # 檢查原始回應

                if completion and "choices" in completion and len(completion["choices"]) > 0:
                    return completion["choices"][0]["text"].strip()
                print("Invalid completion structure:", completion)  # 檢查失敗原因
                return "I don't know"

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return "I don't know"

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        try:
            batch_interaction_ids = batch["interaction_id"]
            queries = batch["query"]
            batch_search_results = batch["search_results"]
            query_times = batch["query_time"]

            # Extract chunks
            chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                batch_interaction_ids, batch_search_results
            )

            # Retrieve top matches using hybrid retrieval
            batch_retrieval_results = []
            for idx, interaction_id in enumerate(batch_interaction_ids):
                query = queries[idx]
                
                # Filter chunks for current interaction
                relevant_mask = chunk_interaction_ids == interaction_id
                relevant_chunks = chunks[relevant_mask]

                # Calculate embeddings for the relevant chunks
                chunk_embeddings = self.calculate_embeddings(relevant_chunks)
                
                # Get hybrid retrieval results (no need for pre-computed embeddings)
                retrieval_results = self.hybrid_retriever.hybrid_retrieve(
                    query, 
                    relevant_chunks,
                    chunk_embeddings=chunk_embeddings
                )
                batch_retrieval_results.append(retrieval_results)

            # Generate responses
            formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)
            responses = []
            for prompt in formatted_prompts:
                response = self.generate_response(prompt)
                responses.append(response)

            return responses

        except Exception as e:
            print(f"Error in batch_generate_answer: {str(e)}")  # 加入更詳細的錯誤信息
            import traceback
            print(traceback.format_exc())  # 打印完整的錯誤堆疊
            return ["I don't know"] * len(queries)

    def calculate_embeddings(self, sentences):
        try:
            return self.sentence_model.encode(
                sentences=sentences,
                normalize_embeddings=True,
                batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
            )
        except Exception as e:
            print(f"Error in calculate_embeddings: {str(e)}")
            return np.array([])

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for idx, query in enumerate(queries):
            query_time = query_times[idx]
            retrieval_results = batch_retrieval_results[idx]

            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            
            user_message = f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n" \
                    f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>\n" \
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"

            formatted_prompts.append(prompt)

        return formatted_prompts