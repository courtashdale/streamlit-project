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
    Split text into overlapping chunks with safety measures.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # For very large documents, use simpler chunking
    if len(text) > 500000:  # 500KB
        return chunk_text_simple(text, chunk_size, overlap)
    
    chunks = []
    start = 0
    text_length = len(text)
    max_chunks = min(5000, (text_length // chunk_size) + 100)  # Hard limit
    
    while start < text_length and len(chunks) < max_chunks:
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
        
        # Ensure we make progress
        new_start = end - overlap if end < text_length else text_length
        if new_start <= start:  # Prevent infinite loop
            new_start = start + max(1, chunk_size // 2)
        
        start = new_start
        
        # Progress indicator for large documents
        if len(chunks) % 100 == 0 and len(chunks) > 0:
            print(f"Processed {len(chunks)} chunks...")
    
    if len(chunks) >= max_chunks:
        print(f"Warning: Reached maximum chunk limit ({max_chunks}). Document may be truncated.")
    
    return chunks


def chunk_text_simple(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple chunking for very large documents - no sentence boundary detection.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    max_chunks = 2000  # Hard limit for very large docs
    
    while start < text_length and len(chunks) < max_chunks:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < text_length else text_length
        
        # Progress indicator
        if len(chunks) % 200 == 0 and len(chunks) > 0:
            print(f"Simple chunking: {len(chunks)} chunks processed...")
    
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
    
    # Process in smaller batches to avoid rate limits and timeouts
    batch_size = 20  # Reduced from 100
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            # Add timeout handling
            response = client.embeddings.create(
                input=batch,
                model=model,
                timeout=30  # 30 second timeout per batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            # Progress indicator
            if total_batches > 1:
                print(f"Processed batch {batch_num}/{total_batches}")
                
        except Exception as e:
            raise Exception(f"OpenAI embedding failed at batch {batch_num}/{total_batches}: {str(e)}")
    
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
    
    # Return chunks with scores (include all results, even with low similarity)
    results = [(chunks[i], similarities[i]) for i in top_indices]
    
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
    # Check if content is too large
    content_length = len(file_content)
    if content_length > 1000000:  # 1MB limit
        print(f"Warning: Large file detected ({content_length:,} characters). Processing may take time.")
    
    # Create chunks from new file
    print(f"Creating chunks from {file_metadata.get('filename', 'unknown file')}...")
    new_chunks = chunk_text(file_content, chunk_size, overlap)
    
    if not new_chunks:
        print("Warning: No chunks created from file content.")
        return existing_chunks, existing_metadata
    
    print(f"Successfully created {len(new_chunks)} chunks.")
    
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