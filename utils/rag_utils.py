# utils/rag_utils.py
"""
RAG (Retrieval-Augmented Generation) utilities for chunking, embedding, and retrieval
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from openai import OpenAI


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Try to find a sentence boundary
        if end < text_length:
            # Look for sentence endings
            for separator in ['. ', '.\n', '! ', '? ', '\n\n']:
                sep_index = text.rfind(separator, start, end)
                if sep_index != -1:
                    end = sep_index + len(separator)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < text_length else text_length
    
    return chunks


def get_embeddings_openai(texts: List[str], client: OpenAI, model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Get OpenAI embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        client: OpenAI client instance
        model: Embedding model to use
        
    Returns:
        np.ndarray: Array of embeddings
    """
    embeddings = []
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def create_tfidf_embeddings(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF embeddings for texts (fallback when OpenAI is not available).
    
    Args:
        texts: List of texts to embed
        
    Returns:
        tuple: (embeddings array, fitted vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings, vectorizer


def find_relevant_chunks(
    query: str,
    chunks: List[str],
    embeddings: np.ndarray,
    top_k: int = 3,
    client: Optional[OpenAI] = None,
    vectorizer: Optional[TfidfVectorizer] = None
) -> List[Tuple[str, float]]:
    """
    Find the most relevant chunks for a query.
    
    Args:
        query: User query
        chunks: List of document chunks
        embeddings: Pre-computed embeddings for chunks
        top_k: Number of top chunks to return
        client: OpenAI client (if using OpenAI embeddings)
        vectorizer: TF-IDF vectorizer (if using TF-IDF)
        
    Returns:
        List[Tuple[str, float]]: List of (chunk, similarity_score) tuples
    """
    if client:
        # Use OpenAI embeddings
        query_embedding = get_embeddings_openai([query], client)[0]
    elif vectorizer:
        # Use TF-IDF
        query_embedding = vectorizer.transform([query]).toarray()[0]
    else:
        raise ValueError("Either client or vectorizer must be provided")
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return chunks with scores
    results = [(chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0]
    
    return results


def create_context_from_chunks(relevant_chunks: List[Tuple[str, float]], max_tokens: int = 2000) -> str:
    """
    Create context string from relevant chunks, respecting token limits.
    
    Args:
        relevant_chunks: List of (chunk, score) tuples
        max_tokens: Maximum number of tokens for context
        
    Returns:
        str: Combined context string
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    context_parts = []
    total_tokens = 0
    
    for chunk, score in relevant_chunks:
        chunk_tokens = len(encoding.encode(chunk))
        if total_tokens + chunk_tokens <= max_tokens:
            context_parts.append(chunk)
            total_tokens += chunk_tokens
        else:
            # Truncate the chunk to fit
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:  # Only add if we have reasonable space
                truncated = encoding.decode(encoding.encode(chunk)[:remaining_tokens])
                context_parts.append(truncated)
            break
    
    return "\n\n".join(context_parts)


def update_document_index(
    file_content: str,
    file_metadata: Dict,
    existing_chunks: List[str],
    existing_metadata: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> Tuple[List[str], List[Dict]]:
    """
    Update the document index with new file content.
    
    Args:
        file_content: Content of the new file
        file_metadata: Metadata of the new file
        existing_chunks: Existing document chunks
        existing_metadata: Existing chunk metadata
        chunk_size: Size of chunks
        overlap: Overlap between chunks
        
    Returns:
        tuple: (updated_chunks, updated_metadata)
    """
    # Create chunks from new file
    new_chunks = chunk_text(file_content, chunk_size, overlap)
    
    # Create metadata for each chunk
    new_metadata = [
        {
            **file_metadata,
            "chunk_index": i,
            "chunk_total": len(new_chunks)
        }
        for i in range(len(new_chunks))
    ]
    
    # Combine with existing
    updated_chunks = existing_chunks + new_chunks
    updated_metadata = existing_metadata + new_metadata
    
    return updated_chunks, updated_metadata