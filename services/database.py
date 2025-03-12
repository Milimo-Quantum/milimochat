import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
import json
from pathlib import Path
from contextlib import contextmanager
import uuid
import numpy as np
import pickle
import base64

from config import MEMORY_DB_PATH
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("database.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: Path = MEMORY_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database with required tables including vector storage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create core tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    created_at DATETIME,
                    last_accessed DATETIME,
                    access_count INTEGER DEFAULT 1,
                    embedding BLOB,
                    session_id TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # Create messages table for chat history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Create memory table for long-term memory with vector support
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME,
                    metadata TEXT,
                    session_id TEXT,
                    chunk_index INTEGER,
                    vector_id TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Create vector store table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vector_store (
                    id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create file store table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_store (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content BLOB,
                    file_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Create settings table for user preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    settings TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)

            # Create datetime_context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datetime_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    timezone TEXT NOT NULL,
                    formatted_string TEXT NOT NULL
                )
            """)

            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_session
                ON memory(key, created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_store_created
                ON vector_store(created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_store_session
                ON file_store(session_id, created_at DESC)
            """)
            
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            print(f"Database connection error: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def save_vector(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a vector to the vector store."""
        try:
            vector_id = str(uuid.uuid4())
            vector_blob = self._serialize_vector(vector)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO vector_store (id, vector, metadata, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        vector_id,
                        vector_blob,
                        json.dumps(metadata) if metadata else None,
                        datetime.now().isoformat()
                    )
                )
                conn.commit()
            return vector_id
        except Exception as e:
            print(f"Error saving vector: {str(e)}")
            raise

    def get_similar_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar vectors using cosine similarity."""
        try:
            results = []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, vector, metadata FROM vector_store")
                
                for row in cursor.fetchall():
                    vector = self._deserialize_vector(row['vector'])
                    similarity = self._cosine_similarity(query_vector, vector)
                    
                    if similarity >= threshold:
                        results.append({
                            'id': row['id'],
                            'similarity': similarity,
                            'metadata': json.loads(row['metadata']) if row['metadata'] else None
                        })
            
            # Sort by similarity and return top-k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error finding similar vectors: {str(e)}")
            return []

    def save_file(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a file to the file store."""
        try:
            file_id = str(uuid.uuid4())
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO file_store 
                    (id, filename, content, file_type, metadata, created_at, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        filename,
                        content,
                        file_type,
                        json.dumps(metadata) if metadata else None,
                        datetime.now().isoformat(),
                        session_id
                    )
                )
                conn.commit()
            return file_id
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a file from the file store."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM file_store WHERE id = ?",
                    (file_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row['id'],
                        'filename': row['filename'],
                        'content': row['content'],
                        'file_type': row['file_type'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                        'created_at': row['created_at'],
                        'session_id': row['session_id']
                    }
                return None
        except Exception as e:
            print(f"Error retrieving file: {str(e)}")
            return None

    def save_memory_with_vector(
        self,
        key: str,
        content: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: Optional[int] = None
    ) -> int:
        """Save a memory entry with associated vector and validation."""
        # Validate required fields
        key = key.strip()
        content = content.strip()
        if not key or not content:
            raise ValueError("Memory requires both key and content fields (non-empty after trimming)")
            
        # Validate and prepare metadata
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Memory metadata must be a dictionary")
            
        # Ensure timestamp exists and is valid
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
            
        try:
            datetime.fromisoformat(metadata['timestamp'])
        except (ValueError, TypeError):
            metadata['timestamp'] = datetime.now().isoformat()
        try:
            vector_id = self.save_vector(vector, metadata)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO memory 
                    (key, content, embedding, metadata, created_at, last_accessed, chunk_index, vector_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        content,
                        self._serialize_vector(vector),
                        json.dumps(metadata) if metadata else None,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        chunk_index,
                        vector_id
                    )
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Error saving memory with vector: {str(e)}")
            raise

    def get_memory_by_similarity(
        self,
        query_vector: List[float],
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        search_messages: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve memories based on vector similarity with enhanced message search."""
        try:
            memories = []
            
            # Search vector store for memory entries
            query = """
                SELECT m.*, v.vector
                FROM memory m
                JOIN vector_store v ON m.vector_id = v.id
                """
            params = []
            
            if session_id:
                query += " WHERE m.session_id = ?"
                params.append(session_id)
            
            logger.info(f"Executing memory query: {query} with params: {params}") # Log query
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    vector = self._deserialize_vector(row['vector'])
                    similarity = self._cosine_similarity(query_vector, vector)
                    
                    if similarity >= min_score:
                        memories.append({
                            'key': row['key'],
                            'content': row['content'],
                            'created_at': row['created_at'],
                            'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                            'similarity': similarity,
                            'chunk_index': row['chunk_index'],
                            'source': 'memory'
                        })
            
            # Also search message history if enabled
            if search_messages:
                message_query = """
                    SELECT m.*, v.id as vector_id, v.vector
                    FROM messages m
                    JOIN vector_store v ON json_extract(m.metadata, '$.vector_id') = v.id
                    WHERE m.session_id = ?
                """
                
                logger.info(f"Executing messages query: {message_query} with params: [{session_id}]") # Log query
                
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(message_query, (session_id,))
                    
                    for row in cursor.fetchall():
                        try:
                            if row['vector']:
                                vector = self._deserialize_vector(row['vector'])
                                similarity = self._cosine_similarity(query_vector, vector)
                                
                                if similarity >= min_score:
                                    metadata = {}
                                    if row['metadata']:
                                        try:
                                            metadata = json.loads(row['metadata'])
                                        except:
                                            pass
                                    
                                    memories.append({
                                        'key': f"message_{row['id']}",
                                        'content': row['content'],
                                        'created_at': row['timestamp'],
                                        'metadata': metadata,
                                        'similarity': similarity,
                                        'role': row['role'],
                                        'source': 'message'
                                    })
                        except Exception as e:
                            logger.error(f"Error processing message row: {str(e)}")
            
            # Sort by similarity and return top-k
            memories.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Log retrieval statistics
            logger.info(f"Retrieved {len(memories)} memories/messages with {len([m for m in memories if m.get('source') == 'message'])} from messages")
            if memories:
                logger.info(f"Top memory similarity: {memories[0]['similarity']}, content: {memories[0]['content'][:100]}...")
            
            return memories[:top_k]
        except Exception as e:
            print(f"Error retrieving memories by similarity: {str(e)}")
            return []

    def _serialize_vector(self, vector: List[float]) -> bytes:
        """Convert vector to bytes for storage."""
        return pickle.dumps(np.array(vector))

    def _deserialize_vector(self, vector_bytes: bytes) -> List[float]:
        """Convert stored bytes back to vector."""
        return pickle.loads(vector_bytes).tolist()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors with zero safety."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0  # Handle zero vectors gracefully
        return np.dot(v1, v2) / norm_product

    # Preserve all existing methods
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one with robust error handling."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute(
                        """
                        SELECT id FROM sessions 
                        WHERE id = ? AND is_active = 1
                        """, 
                        (session_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        cursor.execute(
                            """
                            UPDATE sessions 
                            SET last_accessed = ?
                            WHERE id = ?
                            """,
                            (datetime.now().isoformat(), session_id)
                        )
                        conn.commit()
                        return session_id
                
                new_session_id = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO sessions (id, created_at, last_accessed, is_active)
                    VALUES (?, ?, ?, 1)
                    """,
                    (
                        new_session_id,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    )
                )
                conn.commit()
                return new_session_id
                
        except Exception as e:
            print(f"Error managing session: {str(e)}")
            raise

    def save_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> int:
        """Save a chat message to the database."""
        try:
            # Ensure metadata is JSON serializable
            if metadata:
                try:
                    metadata = json.loads(json.dumps(metadata))
                except (TypeError, json.JSONDecodeError):
                    print("Warning: Could not serialize metadata, storing as empty dict")
                    metadata = {}
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        content,
                        json.dumps(metadata) if metadata else None,
                        datetime.now().isoformat()
                    )
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Error saving message: {str(e)}")
            return -1

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve chat messages with improved error handling."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT id, role, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (session_id,))
                messages = []
                
                for row in cursor.fetchall():
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else None
                        messages.append({
                            'id': row['id'],
                            'role': row['role'],
                            'content': row['content'],
                            'timestamp': row['timestamp'],
                            'metadata': metadata
                        })
                    except Exception as e:
                        print(f"Error processing message row: {str(e)}")
                        continue
                
                return messages
        except Exception as e:
            print(f"Error retrieving messages: {str(e)}")
            return []

    def save_memory(
        self,
        key: str,
        content: str,
        embedding: Optional[Union[List[float], bytes]] = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """Save a memory entry with validation and default values."""
        # Validate and clean required fields
        key = key.strip()
        content = content.strip()
        if not key or not content:
            raise ValueError("Memory requires both key and content fields (non-empty after trimming)")
            
        # Validate metadata structure
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Memory metadata must be a dictionary")
            
        # Enforce required memory schema
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
            
        if not isinstance(metadata.get('timestamp'), str):
            raise ValueError("Memory timestamp must be ISO format string")
            
        # Validate timestamp format
        try:
            datetime.fromisoformat(metadata['timestamp'])
        except (ValueError, TypeError):
            raise ValueError("Invalid timestamp format, must be ISO 8601")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if metadata is None:
                    metadata = {}
                metadata_json = json.dumps(metadata)

                # Handle embedding serialization
                embedding_blob = None
                if embedding is not None:
                    if isinstance(embedding, bytes):
                        embedding_blob = embedding
                    else:
                        try:
                            embedding_blob = self._serialize_vector(embedding)
                        except Exception as e:
                            print(f"Error serializing embedding: {str(e)}")

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO memory
                    (key, content, embedding, metadata, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        content,
                        embedding_blob,
                        metadata_json,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    )
                )

                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Error saving memory: {str(e)}")
            return -1
    
    def get_all_long_term_memories(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all long-term memories for a session with validation."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT
                        m.key,
                        m.content,
                        m.created_at,
                        m.last_accessed,
                        m.metadata
                    FROM memory m
                    WHERE m.key LIKE ?
                    OR (m.metadata IS NOT NULL
                        AND (
                            json_extract(m.metadata, '$.session_id') = ?
                            OR json_extract(m.metadata, '$.persistent_id') = ?
                        ))
                    ORDER BY m.created_at DESC
                """, (f"{session_id}%", session_id, session_id))
                
                valid_memories = []
                for row in cursor.fetchall():
                    try:
                        # Validate required fields
                        if not all(row[key] for key in ['key', 'content', 'created_at']):
                            print(f"Skipping invalid memory - missing required fields: {row['key']}")
                            continue
                            
                        # Process metadata
                        metadata_str = row['metadata'] or '{}'
                        metadata = json.loads(metadata_str)
                        
                        # Ensure timestamp exists and is valid
                        if 'timestamp' not in metadata:
                            metadata['timestamp'] = row['created_at']
                            
                        try:
                            datetime.fromisoformat(metadata['timestamp'])
                        except (ValueError, TypeError):
                            metadata['timestamp'] = datetime.now().isoformat()
                        
                        # Validate and build memory object with defaults
                        memory = {
                            'key': row['key'],
                            'content': row['content'],
                            'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                            'created_at': row['created_at'],
                            'last_accessed': row['last_accessed'],
                            'metadata': metadata
                        }
                        
                        # Skip invalid memories that are missing critical fields
                        if not memory['key'] or not memory['content']:
                            continue
                        valid_memories.append(memory)
                    except Exception as e:
                        print(f"Error processing memory row: {str(e)}")
                        continue
                
                return valid_memories
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []

    def update_datetime_context(self) -> str:
        """Update and return the current datetime context."""
        try:
            current_time = datetime.now()
            local_timezone = current_time.astimezone().tzname()
            
            formatted_context = (
                "CURRENT DATE AND TIME INFORMATION:\n"
                f"• Current Time: {current_time.strftime('%I:%M %p')}\n"
                f"• Current Date: {current_time.strftime('%A, %B %d, %Y')}\n"
                f"• Day of Week: {current_time.strftime('%A')}\n"
                f"• Month: {current_time.strftime('%B')}\n"
                f"• Year: {current_time.strftime('%Y')}\n"
                f"• Timezone: {local_timezone}\n"
            )

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO datetime_context (timestamp, timezone, formatted_string)
                    VALUES (?, ?, ?)
                """, (current_time.isoformat(), local_timezone, formatted_context))
                conn.commit()

            return formatted_context
        except Exception as e:
            print(f"Error updating datetime context: {str(e)}")
            return ""

    def get_current_datetime_context(self) -> str:
        """Get the most recent datetime context."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT formatted_string
                    FROM datetime_context
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                result = cursor.fetchone()
                
                if not result:
                    return self.update_datetime_context()
                
                return result['formatted_string']
        except Exception as e:
            print(f"Error getting datetime context: {str(e)}")
            return self.update_datetime_context()

    def delete_messages(self, session_id: str) -> None:
        """Delete all messages for a session."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.commit()
        except Exception as e:
            print(f"Error deleting messages: {str(e)}")

    def delete_message(self, message_id: int) -> None:
        """Delete a specific message."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                conn.commit()
        except Exception as e:
            print(f"Error deleting message: {str(e)}")

    def delete_memory(self, key: str) -> None:
        """Delete memory with complete session and relationship handling."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # First get memory details with flexible key matching
                    cursor.execute("""
                        SELECT m.*, v.id as vector_store_id, f.id as file_store_id
                        FROM memory m
                        LEFT JOIN vector_store v ON m.vector_id = v.id
                        LEFT JOIN file_store f ON (
                            m.metadata IS NOT NULL AND 
                            json_extract(m.metadata, '$.file_id') = f.id
                        )
                        WHERE m.key = ? 
                        OR m.key LIKE ? 
                        OR m.key LIKE ?
                        OR (
                            m.metadata IS NOT NULL AND (
                                json_extract(m.metadata, '$.memory_key') = ? OR
                                json_extract(m.metadata, '$.memory_key') LIKE ? OR
                                json_extract(m.metadata, '$.memory_key') LIKE ?
                            )
                        )
                    """, (
                        key,                    # Exact match
                        f"{key}_%",            # Prefix match
                        f"%_{key}",            # Suffix match
                        key,                    # Metadata exact match
                        f"{key}_%",            # Metadata prefix match
                        f"%_{key}"             # Metadata suffix match
                    ))
                    
                    memory_records = cursor.fetchall()
                    
                    if not memory_records:
                        print(f"No memory found for key: {key}")
                        return
                    
                    for record in memory_records:
                        # Delete associated vector store entry
                        if record['vector_store_id']:
                            cursor.execute(
                                "DELETE FROM vector_store WHERE id = ?",
                                (record['vector_store_id'],)
                            )
                        
                        # Delete associated file store entry
                        if record['file_store_id']:
                            cursor.execute(
                                "DELETE FROM file_store WHERE id = ?",
                                (record['file_store_id'],)
                            )
                        
                        # Delete any chunk entries
                        if record['chunk_index'] is not None:
                            cursor.execute(
                                """
                                DELETE FROM memory 
                                WHERE key LIKE ? AND chunk_index IS NOT NULL
                                """,
                                (f"{record['key']}%",)
                            )
                        
                        # Delete the main memory entry
                        cursor.execute(
                            "DELETE FROM memory WHERE id = ?",
                            (record['id'],)
                        )
                    
                    # Cleanup any orphaned entries
                    cursor.execute("""
                        DELETE FROM vector_store 
                        WHERE id NOT IN (SELECT DISTINCT vector_id FROM memory WHERE vector_id IS NOT NULL)
                    """)
                    
                    cursor.execute("""
                        DELETE FROM file_store 
                        WHERE id NOT IN (
                            SELECT DISTINCT json_extract(metadata, '$.file_id') 
                            FROM memory 
                            WHERE metadata IS NOT NULL AND json_extract(metadata, '$.file_id') IS NOT NULL
                        )
                    """)
                    
                    conn.commit()
                    print(f"Successfully deleted memory and related entries for key: {key}")
                    
                except Exception as e:
                    conn.rollback()
                    print(f"Error during memory deletion transaction: {str(e)}")
                    raise
                    
        except Exception as e:
            print(f"Error in delete_memory: {str(e)}")
            raise

        finally:
            # Ensure connection is closed
            if 'conn' in locals():
                if conn:
                    conn.close()

    def delete_memory_with_vectors(self, memory_key: str, vector_id: Optional[str], session_id: str) -> None:
        """Delete memory with associated vectors and metadata."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # Delete vector if exists
                    if vector_id:
                        cursor.execute(
                            "DELETE FROM vector_store WHERE id = ?",
                            (vector_id,)
                        )
                    
                    # Delete memory entry
                    cursor.execute(
                        """DELETE FROM memory 
                           WHERE key = ? AND 
                           (session_id = ? OR metadata LIKE ?)""",
                        (memory_key, session_id, f"%{session_id}%")
                    )
                    
                    # Commit transaction
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise Exception(f"Failed to delete memory: {str(e)}")
                    
        except Exception as e:
            raise Exception(f"Database error during memory deletion: {str(e)}")

    def delete_memory_complete(self, key: str, session_id: str) -> None:
        """Complete memory deletion with robust key resolution."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # Find all related memory records
                    cursor.execute("""
                        SELECT m.id, m.key, m.metadata, m.vector_id
                        FROM memory m
                        WHERE m.key = ?
                        OR m.key LIKE ?
                        OR (
                            m.metadata IS NOT NULL AND (
                                json_extract(m.metadata, '$.memory_key') = ? OR
                                json_extract(m.metadata, '$.file_key') = ? OR
                                json_extract(m.metadata, '$.key') = ?
                            )
                        )
                        AND (m.session_id = ? OR json_extract(m.metadata, '$.session_id') = ?)
                    """, (key, f"%{key}%", key, key, key, session_id, session_id))
                    
                    records = cursor.fetchall()
                    
                    for record in records:
                        # Delete vector store entry if exists
                        if record['vector_id']:
                            cursor.execute(
                                "DELETE FROM vector_store WHERE id = ?", 
                                (record['vector_id'],)
                            )
                        
                        # Delete file store entries if exist
                        if record['metadata']:
                            try:
                                metadata = json.loads(record['metadata'])
                                if file_id := metadata.get('file_id'):
                                    cursor.execute(
                                        "DELETE FROM file_store WHERE id = ?",
                                        (file_id,)
                                    )
                            except json.JSONDecodeError:
                                pass
                        
                        # Delete the memory record
                        cursor.execute(
                            "DELETE FROM memory WHERE id = ?",
                            (record['id'],)
                        )
                        
                        # Delete any related chunks
                        cursor.execute(
                            "DELETE FROM memory WHERE key LIKE ? AND chunk_index IS NOT NULL",
                            (f"{record['key']}%",)
                        )
                    
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise Exception(f"Transaction failed: {str(e)}")
                    
        except Exception as e:
            print(f"Error in delete_memory_complete: {str(e)}")
            raise

    def clear_all_session_memory(self, session_id: str) -> None:
        """Clear all memory entries for a session with improved transaction handling."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Start transaction
                    cursor.execute("BEGIN TRANSACTION")
                    
                    # Get vector_ids to delete
                    cursor.execute(
                        "SELECT vector_id FROM memory WHERE key LIKE ?",
                        (f"{session_id}_%",)
                    )
                    vector_ids = [row['vector_id'] for row in cursor.fetchall() if row['vector_id']]
                    
                    # Delete from vector store
                    if vector_ids:
                        cursor.executemany(
                            "DELETE FROM vector_store WHERE id = ?",
                            [(vid,) for vid in vector_ids]
                        )
                    
                    # Delete from memory table
                    cursor.execute(
                        "DELETE FROM memory WHERE key LIKE ?",
                        (f"{session_id}_%",)
                    )
                    
                    # Delete from messages table
                    cursor.execute(
                        "DELETE FROM messages WHERE session_id = ?",
                        (session_id,)
                    )
                    
                    # Delete from file store
                    cursor.execute(
                        "DELETE FROM file_store WHERE session_id = ?",
                        (session_id,)
                    )
                    
                    # Commit transaction
                    conn.commit()
                    print(f"Successfully cleared all memory for session {session_id}")
                except Exception as e:
                    conn.rollback()
                    raise Exception(f"Failed to clear session memory: {str(e)}")
        except Exception as e:
            print(f"Error clearing session memory: {str(e)}")
            raise

    def clean_old_memories(self, days: int = 30) -> None:
        """Clean up old memory entries that haven't been accessed recently."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get vector_ids to delete
                cursor.execute(
                    """
                    SELECT vector_id FROM memory
                    WHERE last_accessed < datetime('now', ?)
                    """,
                    (f'-{days} days',)
                )
                vector_ids = [row['vector_id'] for row in cursor.fetchall() if row['vector_id']]
                
                # Delete from vector store
                if vector_ids:
                    cursor.executemany(
                        "DELETE FROM vector_store WHERE id = ?",
                        [(vid,) for vid in vector_ids]
                    )
                
                # Delete old memories
                cursor.execute(
                    """
                    DELETE FROM memory
                    WHERE last_accessed < datetime('now', ?)
                    """,
                    (f'-{days} days',)
                )
                
                # Delete old files
                cursor.execute(
                    """
                    DELETE FROM file_store
                    WHERE created_at < datetime('now', ?)
                    """,
                    (f'-{days} days',)
                )
                
                conn.commit()
        except Exception as e:
            print(f"Error cleaning old memories: {str(e)}")

    def get_memory_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {
                    'total_memories': 0,
                    'total_vectors': 0,
                    'total_files': 0,
                    'avg_vector_size': 0,
                    'memory_usage_bytes': 0
                }

                # Session filtering parameters
                mem_condition = "WHERE session_id = ?" if session_id else ""
                file_condition = "WHERE session_id = ?" if session_id else ""
                params = (session_id,) if session_id else ()

                # Get memory counts
                cursor.execute(
                    f"SELECT COUNT(*) as count FROM memory {mem_condition}",
                    params
                )
                stats['total_memories'] = cursor.fetchone()['count']

                # Get session-associated vector counts
                if session_id:
                    cursor.execute("""
                        SELECT COUNT(DISTINCT v.id) as count
                        FROM vector_store v
                        JOIN memory m ON v.id = m.vector_id
                        WHERE m.session_id = ?
                    """, (session_id,))
                else:
                    cursor.execute("SELECT COUNT(*) as count FROM vector_store")
                stats['total_vectors'] = cursor.fetchone()['count']

                # Get file counts
                cursor.execute(
                    f"SELECT COUNT(*) as count FROM file_store {file_condition}",
                    params
                )
                stats['total_files'] = cursor.fetchone()['count']
                
                # Calculate average vector size
                if session_id:
                    cursor.execute("""
                        SELECT AVG(LENGTH(v.vector)) as avg_size
                        FROM vector_store v
                        JOIN memory m ON v.id = m.vector_id
                        WHERE m.session_id = ?
                    """, (session_id,))
                else:
                    cursor.execute("SELECT AVG(LENGTH(vector)) as avg_size FROM vector_store")
                result = cursor.fetchone()
                stats['avg_vector_size'] = result['avg_size'] if result['avg_size'] else 0
                
                # Calculate total memory usage
                if session_id:
                    cursor.execute("""
                        SELECT SUM(LENGTH(v.vector)) as total
                        FROM vector_store v
                        JOIN memory m ON v.id = m.vector_id
                        WHERE m.session_id = ?
                    """, (session_id,))
                    vector_size = cursor.fetchone()['total'] or 0
                    
                    cursor.execute("""
                        SELECT SUM(LENGTH(f.content)) as total
                        FROM file_store f
                        WHERE f.session_id = ?
                    """, (session_id,))
                    file_size = cursor.fetchone()['total'] or 0
                else:
                    cursor.execute("SELECT SUM(LENGTH(vector)) as total FROM vector_store")
                    vector_size = cursor.fetchone()['total'] or 0
                    
                    cursor.execute("SELECT SUM(LENGTH(content)) as total FROM file_store")
                    file_size = cursor.fetchone()['total'] or 0
                
                stats['memory_usage_bytes'] = vector_size + file_size
                
                return stats
        except Exception as e:
            print(f"Error getting memory stats: {str(e)}")
            return {
                'total_memories': 0,
                'total_vectors': 0,
                'total_files': 0,
                'avg_vector_size': 0,
                'memory_usage_bytes': 0
            }
