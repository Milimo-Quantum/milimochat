from typing import List, Union, Dict, Any
import numpy as np
from numpy.linalg import norm
import json
import pickle
from datetime import datetime

from ollama_client import OllamaClient
from config import EMBEDDINGS_DIMENSION

class EmbeddingsManager:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.dimension = EMBEDDINGS_DIMENSION
        self.cache = {}

    async def get_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """Get embedding vector for text."""
        if use_cache and text in self.cache:
            return self.cache[text]

        try:
            # Get embedding with validation
            embedding = self.ollama_client.generate_embeddings(text)
            
            # Ensure proper dimensionality
            if not embedding or len(embedding) != self.dimension:
                print(f"Warning: Invalid embedding generated for text: {text[:50]}...")
                embedding = [0.0] * self.dimension  # Return zero vector as fallback
            
            if use_cache:
                self.cache[text] = embedding
            
            return embedding

        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def get_batch_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embeddings = []
        
        for text in texts:
            embedding = await self.get_embedding(text, use_cache)
            embeddings.append(embedding)
        
        return embeddings

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
            
            return float(similarity)

        except Exception as e:
            raise Exception(f"Failed to calculate similarity: {str(e)}")

    async def find_similar_texts(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find most similar texts to a query."""
        try:
            # Get query embedding
            query_embedding = await self.get_embedding(query)
            
            # Get embeddings for all texts
            text_embeddings = await self.get_batch_embeddings(texts)
            
            # Calculate similarities
            similarities = []
            for i, text_embedding in enumerate(text_embeddings):
                similarity = self.calculate_similarity(query_embedding, text_embedding)
                if similarity >= threshold:
                    similarities.append({
                        'text': texts[i],
                        'similarity': similarity
                    })
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            raise Exception(f"Failed to find similar texts: {str(e)}")

    def save_cache(self, filepath: str):
        """Save embedding cache to file."""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'dimension': self.dimension,
                'embeddings': self.cache
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            raise Exception(f"Failed to save cache: {str(e)}")

    def load_cache(self, filepath: str):
        """Load embedding cache from file."""
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify dimension matches
            if cache_data['dimension'] != self.dimension:
                raise ValueError("Cache embedding dimension mismatch")
            
            self.cache = cache_data['embeddings']

        except Exception as e:
            raise Exception(f"Failed to load cache: {str(e)}")

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache = {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            'cache_size': len(self.cache),
            'dimension': self.dimension,
            'memory_usage': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        try:
            # Pickle cache and get size
            return len(pickle.dumps(self.cache))
        except:
            return 0

    async def bulk_process_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> Dict[str, List[float]]:
        """Process multiple texts in batches."""
        results = {}
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.get_batch_embeddings(batch_texts)
            
            for text, embedding in zip(batch_texts, batch_embeddings):
                results[text] = embedding
            
            if show_progress:
                print(f"Processed batch {i//batch_size + 1}/{total_batches}")
        
        return results