import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import OrderedDict

class ContextIntegrator:
    def __init__(self, max_context_length: int = 32768):
        """Initialize context integrator with maximum context length."""
        self.max_context_length = max_context_length
        self.seen_memory_keys = set()
        self.context_usage_stats = {}
        
    def format_memory_for_context(
        self,
        memory: Dict[str, Any],
        memory_type: str = "long_term"
    ) -> Dict[str, Any]:
        """Format a memory entry for context inclusion with proper structure."""
        try:
            # Basic validation
            required_fields = {'content', 'created_at'}
            if not all(field in memory for field in required_fields):
                raise ValueError("Memory missing required fields")

            # Format timestamp consistently
            timestamp = datetime.fromisoformat(memory['created_at'])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Generate memory key if not present
            memory_key = memory.get('key', f"{memory_type}_{timestamp.timestamp()}")

            # Structure the memory with clear hierarchy
            formatted_memory = {
                'key': memory_key,
                'type': memory_type,
                'content': memory['content'],
                'timestamp': formatted_time,
                'metadata': {
                    'original_timestamp': memory['created_at'],
                    'memory_type': memory_type,
                    'source': memory.get('source', 'unknown'),
                    'relevance_score': memory.get('relevance_score', 1.0),
                    'vector_id': memory.get('vector_id'),
                    'context_used': memory.get('context_used', False)
                }
            }

            # Add optional fields if present
            if 'embedding' in memory:
                formatted_memory['metadata']['has_embedding'] = True
            
            if 'chunk_index' in memory:
                formatted_memory['metadata']['chunk_index'] = memory['chunk_index']

            return formatted_memory
            
        except Exception as e:
            print(f"Error formatting memory: {str(e)}")
            return None

    def prioritize_memories(
        self,
        memories: List[Dict[str, Any]],
        current_context: str,
        query_embedding: Optional[np.ndarray] = None,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Prioritize memories using semantic similarity and robust scoring."""
        try:
            scored_memories = []
            
            for memory in memories:
                if not isinstance(memory, dict):
                    continue

                # Skip recently used memories
                memory_key = memory.get('key')
                if memory_key in self.seen_memory_keys:
                    continue

                # Base score with semantic boost for long-term memories
                base_score = memory.get('relevance_score', 0.0)
                if memory.get('type') == 'long_term' and 'embedding' in memory:
                    base_score = 0.0  # Remove relevance threshold

                # Recency decay adjustment
                timestamp = datetime.fromisoformat(memory['created_at'])
                hours_diff = (datetime.now() - timestamp).total_seconds() / 3600
                recency_score = np.exp(-hours_diff/480)  # 20-day half-life for longer retention

                # Semantic similarity scoring
                context_score = 0.0
                exact_match = False
                if query_embedding is not None and memory.get('embedding'):
                    try:
                        # Cosine similarity for vectorized memories
                        memory_embedding = np.array(memory['embedding'])
                        context_score = np.dot(query_embedding, memory_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                        )
                        
                        # Check for exact phrase match in content
                        if memory.get('content') and current_context:
                            exact_match = current_context.lower() in memory['content'].lower()
                            if exact_match:
                                context_score = max(context_score, 1.0)  # Max score for exact matches
                    except Exception as e:
                        print(f"Embedding error: {str(e)}")
                elif memory.get('content') and current_context:
                    # Fallback to BM25-style text matching
                    memory_tokens = memory['content'].lower().split()
                    context_tokens = current_context.lower().split()
                    common_tokens = set(memory_tokens) & set(context_tokens)
                    context_score = sum(
                        (memory_tokens.count(t) * context_tokens.count(t)) /
                        (len(memory_tokens) + len(context_tokens))
                        for t in common_tokens
                    )
                    # Boost exact matches
                    if current_context.lower() in memory['content'].lower():
                        context_score = 1.0

                # Combined scoring with adjusted weights
                final_score = context_score * 1.0  # Base scoring purely on semantic relevance
                
                # Apply boosts
                if exact_match:
                    final_score *= 3.0  # Triple score for exact matches
                    # Additional boost for phrase matches
                    if current_context.lower().strip() in memory['content'].lower().strip():
                        final_score *= 1.5
                elif memory.get('type') == 'long_term' and context_score > 0.5:
                    final_score *= 1.25  # Smaller boost for good matches

                scored_memories.append({
                    **memory,
                    'priority_score': final_score,
                    'exact_match': exact_match
                })

            # Sort by final score
            scored_memories.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return scored_memories

        except Exception as e:
            print(f"Error prioritizing memories: {str(e)}")
            return memories

    def deduplicate_memories(
        self,
        memories: List[Dict[str, Any]],
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar memories."""
        try:
            if not memories:
                return []

            unique_memories = []
            seen_contents = set()

            for memory in memories:
                if not memory:  # Skip null entries
                    continue
                content = memory.get('content', '').strip()
                
                # Skip empty content
                if not content:
                    continue

                # Generate a simplified version for comparison
                simplified = ' '.join(content.lower().split())
                
                # Check for duplicates or near-duplicates
                is_duplicate = False
                for seen_content in seen_contents:
                    # Calculate similarity
                    similarity = self._calculate_text_similarity(simplified, seen_content)
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_memories.append(memory)
                    seen_contents.add(simplified)

            return unique_memories

        except Exception as e:
            print(f"Error deduplicating memories: {str(e)}")
            return memories

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using character trigrams."""
        try:
            # Generate trigrams
            def get_trigrams(text):
                return set(text[i:i+3] for i in range(len(text)-2))
            
            trigrams1 = get_trigrams(text1)
            trigrams2 = get_trigrams(text2)
            
            # Calculate Jaccard similarity
            intersection = len(trigrams1.intersection(trigrams2))
            union = len(trigrams1.union(trigrams2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating text similarity: {str(e)}")
            return 0.0

    def merge_context(
        self,
        short_term_memories: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
        current_context: str,
        max_memories: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Merge short-term and long-term memories with proper context integration."""
        try:
            merged_context = []
            context_metadata = {
                'total_memories': 0,
                'short_term_used': 0,
                'long_term_used': 0,
                'timestamp': datetime.now().isoformat()
            }

            # Format all memories consistently
            # Validate and format memories
            def is_valid_memory(mem):
                return isinstance(mem, dict) and \
                    'content' in mem and \
                    'created_at' in mem and \
                    isinstance(mem['content'], str) and \
                    len(mem['content'].strip()) > 0

            formatted_short_term = [
                self.format_memory_for_context(mem, "short_term")
                for mem in short_term_memories
                if is_valid_memory(mem)
            ]
            
            formatted_long_term = [
                self.format_memory_for_context(mem, "long_term")
                for mem in long_term_memories
                if is_valid_memory(mem)
            ]

            # Deduplicate each set with relaxed thresholds for long-term
            unique_short_term = self.deduplicate_memories(formatted_short_term)
            unique_long_term = self.deduplicate_memories(
                formatted_long_term,
                similarity_threshold=0.95  # Higher threshold preserves more variations
            )

            # Prioritize with enhanced weights for vector matches
            prioritized_short_term = self.prioritize_memories(
                unique_short_term,
                current_context,
                vector_weight=0.5  # Reduced emphasis on temporal recency
            )
            prioritized_long_term = self.prioritize_memories(
                unique_long_term,
                current_context,
                vector_weight=0.8  # Increased weight for semantic matches
            )

            # Allocate 80% of slots to long-term memories, 20% to short-term
            long_term_slots = min(
                len(prioritized_long_term),
                int(max_memories * 0.8)
            )
            short_term_slots = max_memories - long_term_slots

            # Add long-term memories first
            for memory in prioritized_long_term[:long_term_slots]:
                merged_context.append(memory)
                self.seen_memory_keys.add(memory['key'])
                context_metadata['long_term_used'] += 1
                context_metadata['total_memories'] += 1

            # Add short-term memories if space remains
            for memory in prioritized_short_term[:short_term_slots]:
                merged_context.append(memory)
                self.seen_memory_keys.add(memory['key'])
                context_metadata['short_term_used'] += 1
                context_metadata['total_memories'] += 1

            # Update metadata
            context_metadata['total_memories'] = len(merged_context)
            context_metadata['total_tokens'] = self._estimate_tokens(merged_context)

            # Sort final context by timestamp
            merged_context.sort(
                key=lambda x: datetime.fromisoformat(x['metadata']['original_timestamp'])
            )

            return merged_context, context_metadata

        except Exception as e:
            print(f"Error merging context: {str(e)}")
            return [], {'error': str(e)}

    def _estimate_tokens(self, memories: List[Dict[str, Any]]) -> int:
        """Estimate token count for memories (rough approximation)."""
        try:
            total_chars = sum(len(str(memory)) for memory in memories)
            # Rough estimate: 1 token ≈ 4 characters
            return total_chars // 4
        except Exception as e:
            print(f"Error estimating tokens: {str(e)}")
            return 0

    def format_context_for_prompt(
        self,
        merged_context: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """Format merged context into a prompt-friendly format."""
        try:
            context_parts = [
                "RELEVANT CONTEXT:",
                f"[Using {metadata['total_memories']} memories: "
                f"{metadata['short_term_used']} short-term, "
                f"{metadata['long_term_used']} long-term]"
            ]

            # Add memories in chronological order
            for memory in merged_context:
                # Format based on memory type
                memory_type = memory['type']
                timestamp = memory['timestamp']
                content = memory['content']
                
                if memory_type == "short_term":
                    prefix = "Recent Memory"
                else:
                    prefix = "Past Memory"
                
                context_parts.append(
                    f"\n{prefix} [{timestamp}]: {content}"
                )

            return "\n".join(context_parts)

        except Exception as e:
            print(f"Error formatting context for prompt: {str(e)}")
            return "Error retrieving context"

    def validate_context(
        self,
        context: str,
        max_length: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Validate and truncate context if necessary."""
        try:
            validation_info = {
                'original_length': len(context),
                'is_truncated': False,
                'token_estimate': len(context) // 4,  # Rough estimate
                'timestamp': datetime.now().isoformat()
            }

            if not max_length:
                max_length = self.max_context_length

            if len(context) > max_length:
                # Truncate while keeping complete memories
                lines = context.split('\n')
                truncated_lines = []
                current_length = 0
                
                for line in lines:
                    if current_length + len(line) + 1 <= max_length:
                        truncated_lines.append(line)
                        current_length += len(line) + 1
                    else:
                        break

                context = '\n'.join(truncated_lines)
                validation_info['is_truncated'] = True
                validation_info['final_length'] = len(context)

            # Validate format
            validation_info['has_header'] = context.startswith("RELEVANT CONTEXT:")
            validation_info['memory_count'] = context.count("Memory [")

            return context, validation_info

        except Exception as e:
            print(f"Error validating context: {str(e)}")
            return context, {'error': str(e)}

    def track_context_usage(
        self,
        context_id: str,
        usage_info: Dict[str, Any]
    ) -> None:
        """Track context usage for optimization."""
        try:
            if context_id not in self.context_usage_stats:
                self.context_usage_stats[context_id] = []
            
            usage_info['timestamp'] = datetime.now().isoformat()
            self.context_usage_stats[context_id].append(usage_info)

            # Prune old stats if needed
            if len(self.context_usage_stats) > 1000:
                oldest_key = min(
                    self.context_usage_stats.keys(),
                    key=lambda k: self.context_usage_stats[k][0]['timestamp']
                )
                del self.context_usage_stats[oldest_key]

        except Exception as e:
            print(f"Error tracking context usage: {str(e)}")

    def clear_context_tracking(self) -> None:
        """Clear context tracking data."""
        self.seen_memory_keys.clear()
        self.context_usage_stats.clear()
