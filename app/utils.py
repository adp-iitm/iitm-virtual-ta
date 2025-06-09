import os
from dotenv import load_dotenv
load_dotenv()

import pickle
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple
import logging
import google.generativeai as genai
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables - only load when needed
model = None
embeddings_cache = None
chunks_cache = None

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.error("GOOGLE_API_KEY environment variable not set")

KNOWLEDGE_BASE = {}

def load_vector_store() -> Tuple[List[Dict], np.ndarray, List[Dict], None]:
    """Load vector store with lazy loading and memory optimization"""
    global embeddings_cache, chunks_cache
    
    if chunks_cache is not None and embeddings_cache is not None:
        logger.info("Using cached vector store")
        return chunks_cache, embeddings_cache, [], None
    
    try:
        with open('embeddings/vector_store.pkl', 'rb') as f:
            vector_store = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise
    
    documents = vector_store['documents']
    embeddings = np.array(vector_store['embeddings'], dtype=np.float32)  # Use float32 to save memory
    metadatas = vector_store['metadatas']
    
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    chunks = [
        {
            'text': doc,
            'title': meta.get('title', 'No title'),
            'url': meta.get('source', ''),
            'metadata': meta
        }
        for doc, meta in zip(documents, metadatas)
    ]
    
    # Cache the results
    chunks_cache = chunks
    embeddings_cache = embeddings
    
    logger.info(f"Loaded {len(chunks)} chunks from vector store")
    
    return chunks, embeddings, metadatas, None

def get_embeddings_via_api(text: str) -> np.ndarray:
    """Get embeddings using a lightweight API instead of loading heavy models"""
    try:
        # Use Hugging Face Inference API (free tier)
        API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-base-en-v1.5"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"}
        
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            embedding = np.array(response.json(), dtype=np.float32)
            if embedding.ndim > 1:
                embedding = embedding[0]  # Take first embedding if batch
            return embedding
        else:
            logger.warning(f"HF API failed with status {response.status_code}")
            
    except Exception as e:
        logger.error(f"API embedding error: {str(e)}")
    
    # Fallback: Use a simple TF-IDF-like approach for basic similarity
    return create_simple_embedding(text)

def create_simple_embedding(text: str, vocab_size: int = 768) -> np.ndarray:
    """Create a simple hash-based embedding as fallback"""
    words = text.lower().split()
    embedding = np.zeros(vocab_size, dtype=np.float32)
    
    for i, word in enumerate(words[:100]):  # Limit to first 100 words
        hash_val = hash(word) % vocab_size
        embedding[hash_val] += 1.0 / (i + 1)  # Weight by position
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def get_top_k_chunks(question: str, chunks: List[Dict], embeddings: np.ndarray, k: int = 5) -> List[Dict]:
    """Get top k chunks with optimized similarity calculation"""
    try:
        # Try API-based embedding first
        question_embedding = get_embeddings_via_api(question)
        
        logger.info(f"Question embedding shape: {question_embedding.shape}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Ensure compatibility
        if question_embedding.shape[0] != embeddings.shape[1]:
            logger.warning("Embedding dimension mismatch, using fallback method")
            return get_top_k_chunks_fallback(question, chunks, k)
        
        if np.any(np.isnan(embeddings)) or np.any(np.isnan(question_embedding)):
            logger.error("NaN values found in embeddings")
            return get_top_k_chunks_fallback(question, chunks, k)
        
        # Calculate similarities in batches to save memory
        batch_size = 50
        similarities = []
        
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            batch_similarities = np.dot(batch_embeddings, question_embedding) / (
                np.linalg.norm(batch_embeddings, axis=1) * np.linalg.norm(question_embedding)
            )
            similarities.extend(batch_similarities)
        
        similarities = np.array(similarities)
        
        if np.any(np.isnan(similarities)):
            logger.error("NaN values found in similarities")
            return get_top_k_chunks_fallback(question, chunks, k)
        
        top_k_indices = np.argsort(similarities)[::-1]
        selected_chunks = []
        
        for idx in top_k_indices:
            if len(selected_chunks) >= k:
                break
            if similarities[idx] >= 0.2:  # Lower threshold for simple embeddings
                selected_chunks.append(chunks[idx])
        
        logger.info(f"Selected {len(selected_chunks)} chunks for question: {question}")
        return selected_chunks
        
    except Exception as e:
        logger.error(f"Error in get_top_k_chunks: {str(e)}")
        return get_top_k_chunks_fallback(question, chunks, k)

def get_top_k_chunks_fallback(question: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
    """Fallback method using simple text matching"""
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        text_lower = chunk['text'].lower()
        text_words = set(text_lower.split())
        
        # Calculate Jaccard similarity
        intersection = len(question_words & text_words)
        union = len(question_words | text_words)
        similarity = intersection / union if union > 0 else 0
        
        # Boost score if exact phrases match
        for word in question_words:
            if len(word) > 3 and word in text_lower:
                similarity += 0.1
        
        if similarity > 0.1:  # Minimum threshold
            scored_chunks.append((similarity, chunk))
    
    # Sort by similarity and return top k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    selected_chunks = [chunk for _, chunk in scored_chunks[:k]]
    
    logger.info(f"Fallback method selected {len(selected_chunks)} chunks")
    return selected_chunks

def process_image(image_path: str) -> str:
    """Process image with error handling"""
    try:
        image = Image.open(image_path)
        # Resize large images to save memory
        if image.size[0] > 1024 or image.size[1] > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return ""

def generate_answer(question: str, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """Generate answer with optimized memory usage"""
    links = []
    
    # Extract proper titles and URLs from chunks
    seen_urls = set()
    for chunk in chunks:
        url = chunk['metadata'].get('original_url', chunk.get('url', ''))
        if not url:
            url = chunk['metadata'].get('source', '')
        
        if not url.startswith('http'):
            continue

        title = chunk.get('title', 'No title')
        
        if title == "No title" or not title.strip():
            lines = chunk['text'].split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('URL:') and len(line) > 5:
                    title = line[:100]
                    break
            if title == "No title":
                url_parts = url.split('/')
                for part in reversed(url_parts):
                    if part and part != 't' and not part.isdigit():
                        title = part.replace('-', ' ').replace('_', ' ').title()
                        if len(title) > 5:
                            break
        
        if url and url not in seen_urls:
            links.append({"url": url, "text": title})
            seen_urls.add(url)
    
    logger.info(f"Generated {len(links)} links")

    # Check knowledge base first
    question_lower = question.lower().strip()
    if question_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[question_lower], links

    if not chunks:
        logger.info("No chunks found")
        return "I couldn't find relevant information. Please check the course materials.", links

    # Prepare context with memory optimization
    context_parts = []
    total_length = 0
    max_context = 1000  # Reduced context size
    
    for chunk in chunks:
        chunk_text = chunk['text'][:300]  # Limit each chunk
        if total_length + len(chunk_text) > max_context:
            break
        context_parts.append(chunk_text)
        total_length += len(chunk_text)
    
    context = "\n".join(context_parts)
    logger.info(f"Context length: {len(context)} characters")

    # Use Google Gemini API
    if not GOOGLE_API_KEY:
        return "API key not configured. Please check the course materials for more information.", links

    try:
        logger.info("Using Google Gemini API")
        genai.configure(api_key=GOOGLE_API_KEY)
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are a helpful Virtual TA for a Tools in Data Science course. 
Based on the following context, provide a concise, accurate answer to the student's question.
If the answer is not in the context, say so and suggest checking the course materials.

Context: {context}

Question: {question}

Answer:"""
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            answer = response.text.strip()
            logger.info("Generated answer successfully")
            return answer, links
        else:
            raise ValueError("No valid response from API")
            
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        
        if context:
            return "I found some relevant information but couldn't generate a complete answer. Please check the linked resources for more details.", links
        else:
            return "I couldn't find a specific answer to your question. Please check the course materials or ask your instructor for clarification.", links