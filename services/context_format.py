import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

class ContextFormatter:
    """Handles proper formatting and integration of memories into model context."""
    
    def __init__(self, max_context_length: int = 32768):
        self.max_context_length = max_context_length
        self.context_usage = {}
        
    def format_memories(
        self,
        short_term: List[Dict[str, Any]],
        long_term: List[Dict[str, Any]],
        query: str,
        include_temporal: bool = True,
        redundancy_filter: bool = True
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Format and merge memories for model context."""
        # Deduplicate memories first
        unique_short_term = self._deduplicate_memories(short_term)
        unique_long_term = self._deduplicate_memories(long_term)
        
        # Calculate relevance scores
        scored_short_term = self._score_memories(unique_short_term, query)
        scored_long_term = self._score_memories(unique_long_term, query)
        
        # Sort by relevance
        scored_short_term.sort(key=lambda x: x['relevance_score'], reverse=True)
        scored_long_term.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Prepare context messages
        context_messages = []
        context_metadata = {
            'total_memories': 0,
            'short_term_used': 0,
            'long_term_used': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add short-term memories with higher priority
        current_length = 0
        for memory in scored_short_term:
            if current_length + len(memory['content']) > self.max_context_length:
                break
                
            context_messages.append({
                'role': 'system',
                'content': f"Recent Memory [{memory['timestamp']}]: {memory['content']}"
            })
            current_length += len(memory['content'])
            context_metadata['short_term_used'] += 1
            context_metadata['total_memories'] += 1
            
        # Add long-term memories if space allows
        remaining_length = self.max_context_length - current_length
        for memory in scored_long_term:
            if current_length + len(memory['content']) > self.max_context_length:
                break
                
            context_messages.append({
                'role': 'system',
                'content': f"Past Memory [{memory['timestamp']}]: {memory['content']}"
            })
            current_length += len(memory['content'])
            context_metadata['long_term_used'] += 1
            context_metadata['total_memories'] += 1
        
        # Track context usage
        usage_id = self._generate_usage_id(query)
        self.context_usage[usage_id] = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'metadata': context_metadata,
            'memory_count': len(context_messages)
        }
        
        return context_messages, context_metadata
        
    def _deduplicate_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate memories based on content similarity."""
        unique_memories = []
        seen_content = set()
        
        for memory in memories:
            # Skip invalid/null memories
            if not memory or not isinstance(memory, dict):
                continue
                
            # Validate required fields
            required_fields = {'content', 'timestamp'}
            if not required_fields.issubset(memory.keys()):
                continue
                
            # Create a normalized version of content for comparison
            normalized_content = self._normalize_content(memory['content'])
            
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_memories.append(memory)
                
        return unique_memories
        
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        # Remove whitespace and convert to lowercase
        return ' '.join(content.lower().split())
        
    def _score_memories(
        self,
        memories: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Score memories based on relevance to query."""
        scored_memories = []
        
        for memory in memories:
            # Calculate temporal relevance (recent = more relevant)
            temporal_score = self._calculate_temporal_score(memory.get('timestamp'))
            
            # Calculate semantic relevance if embeddings available
            semantic_score = self._calculate_semantic_score(
                memory.get('embedding'),
                memory.get('query_embedding')
            )
            
            # Combine scores (weight can be adjusted)
            relevance_score = (temporal_score * 0.3) + (semantic_score * 0.7)
            
            memory_copy = memory.copy()
            memory_copy['relevance_score'] = relevance_score
            scored_memories.append(memory_copy)
            
        return scored_memories
        
    def _calculate_temporal_score(self, timestamp: Optional[str]) -> float:
        """Calculate temporal relevance score."""
        if not timestamp:
            return 0.5  # Default score for memories without timestamp
            
        try:
            memory_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            time_diff = (now - memory_time).total_seconds()
            
            # Exponential decay based on time difference
            return np.exp(-time_diff / (24 * 3600))  # 24 hours decay factor
        except Exception:
            return 0.5
            
    def _calculate_semantic_score(
        self,
        memory_embedding: Optional[List[float]],
        query_embedding: Optional[List[float]]
    ) -> float:
        """Calculate semantic similarity score."""
        if not memory_embedding or not query_embedding:
            return 0.5  # Default score if embeddings not available
            
        try:
            # Cosine similarity between embeddings
            memory_vec = np.array(memory_embedding)
            query_vec = np.array(query_embedding)
            
            similarity = np.dot(memory_vec, query_vec) / (
                np.linalg.norm(memory_vec) * np.linalg.norm(query_vec)
            )
            
            return float(similarity)
        except Exception:
            return 0.5
            
    def _generate_usage_id(self, query: str) -> str:
        """Generate unique ID for context usage tracking."""
        timestamp = datetime.now().isoformat()
        return f"ctx_{timestamp}_{hash(query)}"
        
    def get_context_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about context usage."""
        if not self.context_usage:
            return {
                'total_queries': 0,
                'avg_memories_used': 0,
                'avg_short_term': 0,
                'avg_long_term': 0
            }
            
        total_queries = len(self.context_usage)
        total_memories = sum(
            usage['memory_count'] for usage in self.context_usage.values()
        )
        total_short_term = sum(
            usage['metadata']['short_term_used']
            for usage in self.context_usage.values()
        )
        total_long_term = sum(
            usage['metadata']['long_term_used']
            for usage in self.context_usage.values()
        )
        
        return {
            'total_queries': total_queries,
            'avg_memories_used': total_memories / total_queries,
            'avg_short_term': total_short_term / total_queries,
            'avg_long_term': total_long_term / total_queries
        }
        
    def validate_context(
        self,
        context_messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Validate and potentially prune context messages."""
        total_length = sum(len(msg['content']) for msg in context_messages)
        
        if total_length <= self.max_context_length:
            return context_messages, {'pruned': False, 'total_length': total_length}
            
        # If context exceeds limit, prune while maintaining message integrity
        pruned_messages = []
        current_length = 0
        
        for msg in reversed(context_messages):  # Start from most recent
            msg_length = len(msg['content'])
            if current_length + msg_length <= self.max_context_length:
                pruned_messages.append(msg)
                current_length += msg_length
                
        return list(reversed(pruned_messages)), {
            'pruned': True,
            'original_length': total_length,
            'pruned_length': current_length
        }