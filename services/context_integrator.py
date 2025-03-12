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
        """Prioritize memories using semantic similarity and robust scoring with enhanced entity detection."""
        try:
            scored_memories = []
            import re
            
            # Enhanced entity detection for important information
            # Look for potential proper nouns, names, or important entities
            def extract_entities(text):
                # Look for potential proper nouns (capitalized words)
                proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
                
                # Look for potential entity patterns
                # Check for name patterns (first name + last name)
                name_patterns = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
                
                # "My name is" pattern
                name_intro_patterns = re.findall(r'(?:my name is|I am|I\'m|call me) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)', text, re.IGNORECASE)
                extracted_names = [match.split(" ")[-1] for match in name_intro_patterns if match]
                
                # Combine all unique entities
                all_entities = set(proper_nouns + name_patterns + extracted_names)
                return all_entities
            
            # Extract entities from current context
            context_entities = extract_entities(current_context) if current_context else set()
            
            for memory in memories:
                if not isinstance(memory, dict):
                    continue

                # Skip recently used memories unless they contain important entities or have high relevance
                memory_key = memory.get('key')
                memory_content = memory.get('content', '')
                memory_entities = extract_entities(memory_content)
                
                # Check if memory contains any entities from current context
                entity_match = bool(context_entities and memory_entities and (context_entities & memory_entities))

                # Only skip if no entity match and not high relevance
                if memory_key in self.seen_memory_keys and not entity_match:
                    # Check if this is a high-relevance memory that should override the skip
                    if memory.get('relevance_score', 0.0) < 0.95 and not memory.get('match_type') == 'exact':
                        continue

                # Base score with source-specific adjustments
                base_score = memory.get('relevance_score', 0.0)
                
                # Different handling based on memory source
                memory_source = memory.get('source', memory.get('type', 'unknown'))
                if memory_source == 'message':
                    # Messages get full weight on semantic relevance
                    base_score = 0.0
                elif memory_source == 'long_term' and 'embedding' in memory:
                    base_score = 0.0  # Remove relevance threshold for long-term memories with embeddings
                elif memory_source == 'file_chunk':
                    # File chunks get a small boost as they often contain valuable information
                    base_score = 0.1

                # Enhanced recency scoring with source-specific adjustments
                created_at = memory.get('created_at', memory.get('timestamp'))
                if not created_at:
                    recency_score = 0.5  # Default for memories without timestamp
                else:
                    try:
                        timestamp = datetime.fromisoformat(created_at)
                        hours_diff = (datetime.now() - timestamp).total_seconds() / 3600
                        
                        # Different decay rates based on source
                        if memory_source == 'message':
                            # Messages have shorter relevance decay (14-day half-life instead of 7)
                            recency_score = np.exp(-hours_diff/336)  # Slower decay - 14 days
                        else:
                            # Even longer half-life for other memories (60-day half-life instead of 30)
                            recency_score = np.exp(-hours_diff/1440)  # Slower decay - 60 days
                    except (ValueError, TypeError):
                        recency_score = 0.5  # Default if timestamp parsing fails
                
                # Comprehensive semantic similarity scoring with fallbacks
                context_score = 0.0
                exact_match = False
                phrase_match = False
                entity_matched = False
                
                # If memory already has a similarity score from database retrieval, use it
                if memory.get('similarity') is not None:
                    context_score = memory['similarity']
                    
                    # If this is an exact match from the database, mark it
                    if memory.get('match_type') == 'exact':
                        exact_match = True
                    elif memory.get('match_type') == 'keyword' or memory.get('match_type') == 'phrase':
                        phrase_match = True
                
                # Try vector similarity if available and no preset similarity
                elif query_embedding is not None and memory.get('embedding'):
                    try:
                        # Cosine similarity for vectorized memories
                        memory_embedding = np.array(memory['embedding'])
                        context_score = np.dot(query_embedding, memory_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                        )
                    except Exception as e:
                        print(f"Embedding error: {str(e)}")
                
                # Check for entity matches which are high-value
                if context_entities and memory_entities:
                    matching_entities = context_entities & memory_entities
                    if matching_entities:
                        entity_matched = True
                        # Strong boost for entity matches
                        entity_score = 0.9 * len(matching_entities) / max(len(context_entities), 1)
                        context_score = max(context_score, entity_score)
                
                # Always check for text matches regardless of embedding availability
                if memory_content and current_context and not exact_match:
                    # Check for exact phrase match
                    memory_content_lower = memory_content.lower()
                    context_lower = current_context.lower()
                    
                    # Complete phrase match
                    if context_lower in memory_content_lower:
                        exact_match = True
                        context_score = max(context_score, 1.0)  # Max score for exact matches
                    
                    # Name detection - special handling for "my name is" pattern
                    name_match = re.search(r'my name is ([A-Za-z]+(?: [A-Za-z]+)?)', memory_content_lower)
                    if name_match and name_match.group(1) in context_lower:
                        exact_match = True  # Name matches are treated as exact
                        context_score = max(context_score, 1.0)
                    
                    # Look for name in current context
                    name_in_context = re.search(r'name(?:s)? (?:is|are) ([A-Za-z]+(?: [A-Za-z]+)?)', context_lower)
                    if name_in_context and name_in_context.group(1) in memory_content_lower:
                        exact_match = True  # Name matches are treated as exact
                        context_score = max(context_score, 1.0)
                    
                    # Partial phrase matches
                    elif len(context_lower) > 4:  # Only for substantial queries
                        # Check for sentence fragments
                        for fragment in self._get_phrases(context_lower):
                            if len(fragment) > 4 and fragment in memory_content_lower:
                                phrase_match = True
                                context_score = max(context_score, 0.85)  # Increased score for phrase matches
                    
                    # No vector or phrase match, fall back to token-based matching
                    if context_score < 0.4:
                        memory_tokens = memory_content_lower.split()
                        context_tokens = context_lower.split()
                        
                        if memory_tokens and context_tokens:
                            common_tokens = set(memory_tokens) & set(context_tokens)
                            if common_tokens:
                                token_score = sum(
                                    (memory_tokens.count(t) * context_tokens.count(t)) /
                                    (len(memory_tokens) + len(context_tokens))
                                    for t in common_tokens
                                ) * 2.5  # Increased scaling for token matching
                                
                                context_score = max(context_score, token_score)

                # Comprehensive scoring with source-specific adjustments
                # Base weight is now purely on semantic relevance
                final_score = context_score
                
                # Add boosting factors based on match type and source
                if exact_match:
                    boost_factor = 5.0  # Stronger boost for exact matches
                    
                    # Extra boost for exact matches in important sources
                    if memory_source == 'message':
                        if memory.get('role') == 'user':
                            boost_factor *= 1.5  # User messages are more important
                    
                    final_score *= boost_factor
                    
                elif entity_matched:
                    boost_factor = 4.0  # Strong boost for entity matches
                    final_score *= boost_factor
                    
                elif phrase_match:
                    boost_factor = 2.0  # Increased boost for phrase matches
                    final_score *= boost_factor
                    
                elif context_score > 0.5:
                    # Decent semantic match gets modest boost
                    boost_factor = 1.5  # Increased boost
                    
                    # File chunks and long-term memories with good scores get extra boost
                    if memory_source in ['file_chunk', 'long_term']:
                        boost_factor = 2.0  # Increased boost for long-term memories
                        
                    final_score *= boost_factor
                
                # Apply recency influence (20% of score is based on recency - reduced from 30%)
                final_score = (final_score * 0.8) + (recency_score * 0.2)
                
                # Additional critical pattern detection
                # Check for important information patterns in memory
                important_patterns = [
                    (r'\bmy name\b.*?\b([A-Za-z]+(?:\s[A-Za-z]+)?)\b', 5.0),  # Name pattern
                    (r'\bcall me\b.*?\b([A-Za-z]+(?:\s[A-Za-z]+)?)\b', 5.0),  # Alternative name pattern
                    (r'\bI am\b.*?\b([A-Za-z]+(?:\s[A-Za-z]+)?)\b', 2.0),     # Identity pattern
                    (r'\b(?:my|our) (?:phone|contact|email|address)\b.*?(\S+@\S+|\d[\d\s-]{8,}|\d+\s\w+\s\w+)', 3.0),  # Contact info
                    (r'\bprefer(?:s|ence)?\b', 2.0),  # Preference pattern
                    (r'\balways\b|\bnever\b', 1.5),   # Strong preference indicators
                    (r'\bimportant\b|\bcritical\b|\bessential\b', 2.5)  # Explicit importance indicators
                ]
                
                for pattern, boost in important_patterns:
                    if re.search(pattern, memory_content, re.IGNORECASE):
                        final_score *= boost  # Apply multiplicative boost for critical information
                
                # Additional metadata for debugging and analysis
                memory_with_score = {
                    **memory,
                    'priority_score': final_score,
                    'semantic_score': context_score,
                    'recency_score': recency_score,
                    'exact_match': exact_match,
                    'phrase_match': phrase_match,
                    'entity_matched': entity_matched,
                    'source': memory_source
                }
                
                scored_memories.append(memory_with_score)

            # Sort by final score
            scored_memories.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return scored_memories

        except Exception as e:
            print(f"Error prioritizing memories: {str(e)}")
            return memories
    
    def _get_phrases(self, text: str, min_phrase_length: int = 3) -> List[str]:
        """Extract meaningful phrases from text for better matching."""
        words = text.split()
        phrases = []
        
        # Single words (if long enough)
        phrases.extend([w for w in words if len(w) >= min_phrase_length])
        
        # 2-word phrases
        if len(words) >= 2:
            phrases.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
            
        # 3-word phrases
        if len(words) >= 3:
            phrases.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])
            
        # 4-word phrases for longer text
        if len(words) >= 4:
            phrases.extend([' '.join(words[i:i+4]) for i in range(len(words)-3)])
            
        return phrases

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
        max_memories: int = 25  # Increased from 10 to allow for more memories
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Merge short-term and long-term memories with enhanced context integration."""
        try:
            merged_context = []
            context_metadata = {
                'total_memories': 0,
                'short_term_used': 0,
                'long_term_used': 0,
                'exact_matches': 0,
                'entity_matches': 0,
                'timestamp': datetime.now().isoformat()
            }

            # Format all memories consistently
            # Validate and format memories with improved validation
            def is_valid_memory(mem):
                if not isinstance(mem, dict):
                    return False
                    
                # Require key fields
                if not all(k in mem for k in ['content', 'created_at']):
                    return False
                    
                # Validate content is meaningful
                if not isinstance(mem.get('content'), str) or len(mem.get('content', '').strip()) < 2:
                    return False
                    
                # Ensure created_at is a valid timestamp
                try:
                    if isinstance(mem.get('created_at'), str):
                        datetime.fromisoformat(mem['created_at'])
                    return True
                except (ValueError, TypeError):
                    return False

            # Apply consistent formatting with source tracking
            formatted_short_term = []
            for mem in short_term_memories:
                if is_valid_memory(mem):
                    formatted = self.format_memory_for_context(mem, "short_term")
                    if formatted:
                        # Track source and match_type if available
                        if 'match_type' in mem:
                            formatted['metadata']['match_type'] = mem['match_type']
                        if 'source' in mem:
                            formatted['metadata']['source'] = mem['source']
                        formatted_short_term.append(formatted)
            
            formatted_long_term = []
            for mem in long_term_memories:
                if is_valid_memory(mem):
                    formatted = self.format_memory_for_context(mem, "long_term")
                    if formatted:
                        # Track source and match_type if available
                        if 'match_type' in mem:
                            formatted['metadata']['match_type'] = mem['match_type']
                        if 'source' in mem:
                            formatted['metadata']['source'] = mem['source']
                        formatted_long_term.append(formatted)

            # Special handling for critical information patterns
            import re
            critical_info_patterns = [
                r'\bmy name is\b',
                r'\bcall me\b',
                r'\b(?:I am|I\'m)\b.*\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b',
                r'\bprefer(?:s|ence)?\b',
                r'\balways\b|\bnever\b',
                r'\bimportant\b|\bcritical\b|\bessential\b'
            ]
            
            # Extract critical memories first for priority inclusion
            critical_short_term = []
            critical_long_term = []
            normal_short_term = []
            normal_long_term = []
            
            # Process short-term memories
            for memory in formatted_short_term:
                content = memory.get('content', '').lower()
                # Check if this contains critical information
                is_critical = any(re.search(pattern, content, re.IGNORECASE) for pattern in critical_info_patterns)
                
                if is_critical:
                    critical_short_term.append(memory)
                else:
                    normal_short_term.append(memory)
            
            # Process long-term memories
            for memory in formatted_long_term:
                content = memory.get('content', '').lower()
                # Check if this contains critical information
                is_critical = any(re.search(pattern, content, re.IGNORECASE) for pattern in critical_info_patterns)
                
                if is_critical:
                    critical_long_term.append(memory)
                else:
                    normal_long_term.append(memory)

            # Deduplicate each category with appropriate thresholds
            unique_critical_short = self.deduplicate_memories(critical_short_term, similarity_threshold=0.9)
            unique_critical_long = self.deduplicate_memories(critical_long_term, similarity_threshold=0.9)
            unique_normal_short = self.deduplicate_memories(normal_short_term, similarity_threshold=0.85)
            unique_normal_long = self.deduplicate_memories(normal_long_term, similarity_threshold=0.95)

            # Prioritize memories by relevance to current context
            prioritized_critical_short = self.prioritize_memories(
                unique_critical_short,
                current_context,
                vector_weight=0.3  # Low vector weight for critical info - preserve regardless of vector match
            )
            prioritized_critical_long = self.prioritize_memories(
                unique_critical_long,
                current_context,
                vector_weight=0.3  # Low vector weight for critical info - preserve regardless of vector match
            )
            prioritized_normal_short = self.prioritize_memories(
                unique_normal_short,
                current_context,
                vector_weight=0.6  # More balanced for normal short-term
            )
            prioritized_normal_long = self.prioritize_memories(
                unique_normal_long,
                current_context,
                vector_weight=0.8  # Higher emphasis on vector relevance for normal long-term
            )

            # Calculate memory allocation with guaranteed slots for critical information
            # Always include all critical information regardless of total slots
            critical_memories = prioritized_critical_short + prioritized_critical_long
            critical_count = len(critical_memories)
            
            # Remaining slots
            remaining_slots = max(0, max_memories - critical_count)
            
            # Allocate remaining slots with priority to long-term memories
            # 25% for short-term, 75% for long-term (adjusted from 20/80)
            normal_long_slots = min(len(prioritized_normal_long), int(remaining_slots * 0.75))
            normal_short_slots = min(len(prioritized_normal_short), remaining_slots - normal_long_slots)

            # First add all critical memories (guaranteed inclusion)
            for memory in critical_memories:
                merged_context.append(memory)
                self.seen_memory_keys.add(memory.get('key', ''))
                
                # Update metadata counters
                if memory.get('type') == 'short_term':
                    context_metadata['short_term_used'] += 1
                else:
                    context_metadata['long_term_used'] += 1
                    
                # Track exact match stats
                if memory.get('metadata', {}).get('match_type') == 'exact':
                    context_metadata['exact_matches'] += 1
                if memory.get('metadata', {}).get('entity_matched', False):
                    context_metadata['entity_matches'] += 1
            
            # Add normal long-term memories based on priority score
            for memory in prioritized_normal_long[:normal_long_slots]:
                merged_context.append(memory)
                self.seen_memory_keys.add(memory.get('key', ''))
                context_metadata['long_term_used'] += 1
                
                # Track match stats
                if memory.get('metadata', {}).get('match_type') == 'exact':
                    context_metadata['exact_matches'] += 1
                if memory.get('metadata', {}).get('entity_matched', False):
                    context_metadata['entity_matches'] += 1

            # Add normal short-term memories if space remains
            for memory in prioritized_normal_short[:normal_short_slots]:
                merged_context.append(memory)
                self.seen_memory_keys.add(memory.get('key', ''))
                context_metadata['short_term_used'] += 1
                
                # Track match stats
                if memory.get('metadata', {}).get('match_type') == 'exact':
                    context_metadata['exact_matches'] += 1
                if memory.get('metadata', {}).get('entity_matched', False):
                    context_metadata['entity_matches'] += 1

            # Update total memories metadata
            context_metadata['total_memories'] = len(merged_context)
            context_metadata['critical_info_count'] = critical_count
            context_metadata['total_tokens'] = self._estimate_tokens(merged_context)

            # Sort final context by timestamp - chronological order helps model understand context timeline
            merged_context.sort(
                key=lambda x: datetime.fromisoformat(x['metadata'].get('original_timestamp', 
                                                                       datetime.now().isoformat()))
            )

            # Add detailed usage statistics
            context_metadata['memory_types'] = {
                'exact_match': sum(1 for m in merged_context if m.get('metadata', {}).get('match_type') == 'exact'),
                'semantic_match': sum(1 for m in merged_context if m.get('metadata', {}).get('match_type') == 'vector'),
                'keyword_match': sum(1 for m in merged_context if m.get('metadata', {}).get('match_type') == 'keyword'),
                'entity_match': sum(1 for m in merged_context if m.get('metadata', {}).get('entity_matched', False)),
            }

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
        """Format merged context into a prompt-friendly format with enhanced semantic structure."""
        try:
            import re
            
            # Start with header information
            context_parts = [
                "RELEVANT CONTEXT:",
                f"[Retrieved {metadata['total_memories']} memories: "
                f"{metadata.get('critical_info_count', 0)} critical, "
                f"{metadata['short_term_used']} recent, "
                f"{metadata['long_term_used']} historical]"
            ]
            
            # Group memories by type for better organization
            critical_memories = []
            recent_memories = []
            historical_memories = []
            
            # Helper functions to detect critical info
            def is_critical_info(content):
                critical_patterns = [
                    r'\bmy name is\b',
                    r'\bcall me\b',
                    r'\bI am\b|\bI\'m\b',
                    r'\bprefer(?:s|ence)?\b',
                    r'\balways\b|\bnever\b',
                    r'\bimportant\b|\bcritical\b|\bessential\b'
                ]
                return any(re.search(pattern, content, re.IGNORECASE) for pattern in critical_patterns)
            
            # Helper function to extract name from memory
            def extract_name(content):
                name_match = re.search(r'(?:my name is|I am|I\'m|call me) ([A-Za-z]+(?: [A-Za-z]+)?)', content, re.IGNORECASE)
                if name_match:
                    return name_match.group(1)
                return None
            
            # Categorize memories
            for memory in merged_context:
                memory_type = memory['type']
                content = memory['content']
                
                # Check for critical information first
                if is_critical_info(content) or memory.get('metadata', {}).get('match_type') == 'exact':
                    critical_memories.append(memory)
                elif memory_type == "short_term":
                    recent_memories.append(memory)
                else:
                    historical_memories.append(memory)
            
            # Format critical memories first and with special highlighting
            if critical_memories:
                context_parts.append("\n--- CRITICAL INFORMATION ---")
                
                # Process name information first (highest priority)
                name_memories = [m for m in critical_memories if extract_name(m['content'])]
                other_critical = [m for m in critical_memories if not extract_name(m['content'])]
                
                for memory in name_memories:
                    timestamp = memory['timestamp']
                    formatted_name = extract_name(memory['content'])
                    
                    if formatted_name:
                        context_parts.append(
                            f"\nUSER IDENTITY [{timestamp}]: {memory['content']}"
                        )
                
                # Format other critical information
                for memory in other_critical:
                    timestamp = memory['timestamp']
                    context_parts.append(
                        f"\nIMPORTANT [{timestamp}]: {memory['content']}"
                    )
            
            # Format recent memories (short-term)
            if recent_memories:
                context_parts.append("\n--- RECENT CONTEXT ---")
                for memory in recent_memories:
                    timestamp = memory['timestamp']
                    context_parts.append(
                        f"\nRecent Memory [{timestamp}]: {memory['content']}"
                    )
            
            # Format historical memories (long-term)
            if historical_memories:
                context_parts.append("\n--- HISTORICAL CONTEXT ---")
                for memory in historical_memories:
                    timestamp = memory['timestamp']
                    
                    # Add source attribution if available
                    source_info = ""
                    if memory.get('metadata', {}).get('source'):
                        source = memory['metadata']['source']
                        source_info = f" [Source: {source}]"
                    
                    context_parts.append(
                        f"\nHistorical Memory [{timestamp}]{source_info}: {memory['content']}"
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
