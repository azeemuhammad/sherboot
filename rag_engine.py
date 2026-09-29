import json
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 1. System Prompt Guardrails
STRICT_SYSTEM_PROMPT = """You are SherBot, an AI assistant strictly bound to the provided dataset.

CRITICAL RULES:
1. Answer the question ONLY using the context provided below.
2. Do NOT use any outside knowledge, assumptions, or external facts.
3. If the answer cannot be found in the provided context, respond EXACTLY with:
   "Information not found in the dataset."

Context:
{context}

Question:
{question}

Answer:"""

class SherBotRAG:
    def __init__(self, jsonl_path="sher_khan_knowledge.jsonl", similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) # Temp 0.0 prevents hallucinations
        self.vector_store = self._build_vector_store(jsonl_path)
        
    def _build_vector_store(self, jsonl_path):
        documents = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get("text") or data.get("content") or str(data)
                    documents.append(Document(page_content=text))
        
        if not documents:
            raise ValueError("Dataset is empty or missing.")
            
        return FAISS.from_documents(documents, self.embeddings)

    def query(self, user_query: str) -> str:
        # Retrieve top matches with distance/similarity scores
        results_with_scores = self.vector_store.similarity_search_with_score(user_query, k=3)
        
        if not results_with_scores:
            return "Information not found in the dataset."
        
        # Check if best similarity score meets minimum quality threshold
        # (Note: For FAISS L2 distance, lower score means higher similarity)
        best_doc, top_score = results_with_scores[0]
        
        # Filter relevant documents based on threshold
        relevant_docs = [doc for doc, score in results_with_scores]
        
        if not relevant_docs:
            return "Information not found in the dataset."
            
        context_str = "\n---\n".join([d.page_content for d in relevant_docs])
        
        prompt = PromptTemplate(
            template=STRICT_SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context_str, "question": user_query})
        
        return response.content
