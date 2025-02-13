import os
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import mimetypes
import json
from PIL import Image
import io
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
import tiktoken
import pdfplumber

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, ERROR_MESSAGES, EMBED_MODEL, VISION_MODEL, IMAGE_PROMPT
from utils.ollama_client import OllamaClient

class FileProcessor:
    def __init__(self):
        self.allowed_extensions = ALLOWED_EXTENSIONS
        self.max_file_size = MAX_FILE_SIZE
        self.ollama_client = OllamaClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.token_splitter = TokenTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            encoding_name="cl100k_base"  # OpenAI's encoding
        )

    def validate_file(self, file) -> Tuple[bool, Optional[str]]:
        """Validate file size and type."""
        if file.size > self.max_file_size:
            return False, ERROR_MESSAGES["file_too_large"]
        
        file_ext = Path(file.name).suffix.lower()
        allowed_exts = [ext for exts in self.allowed_extensions.values() for ext in exts]
        
        if file_ext not in allowed_exts:
            return False, ERROR_MESSAGES["invalid_file_type"]
        
        return True, None

    def get_file_type(self, filename: str) -> Optional[str]:
        """Determine file type based on extension."""
        ext = Path(filename).suffix.lower()
        for file_type, extensions in self.allowed_extensions.items():
            if ext in extensions:
                return file_type
        return None

    async def process_image(self, file) -> Dict[str, Any]:
        """Process image files with enhanced metadata extraction."""
        try:
            image = Image.open(file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large while maintaining aspect ratio
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Generate image analysis using /api/generate endpoint
            response_content = "No analysis available." # Initialize with a default value
            try:
                async for chunk in self.ollama_client.generate_completion( # Changed to generate_completion
                    model=VISION_MODEL, # Using VISION_MODEL
                    prompt=IMAGE_PROMPT, # Basic prompt for now
                    images=[img_str] # Sending base64 image
                ):
                    response_content += chunk
            except Exception as e:
                response_content = f"Error generating image analysis: {str(e)}" # Keep error message in case of failure

            return {
                'type': 'image',
                'format': 'base64',
                'data': img_str,
                'analysis': response_content, # Store analysis instead of embedding
                'metadata': {
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format,
                    'created_at': datetime.now().isoformat(),
                    'filename': file.name,
                    'mime_type': mimetypes.guess_type(file.name)[0]
                }
            }
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    async def process_document(self, file, progress_callback=None) -> Dict[str, Any]:
        """Process document with improved content extraction and embedding generation."""
        try:
            async def update_progress(current: int, total: int, message: str):
                if progress_callback:
                    progress_callback.write(message)
                else:
                    print(f"{message}: {current}/{total}")

            await update_progress(0, 1, "Analyzing document structure")

            # First extract text content
            if file.type == "application/pdf":
                text_content = await self._extract_pdf_text(file)
            else:
                text_content = file.read().decode('utf-8')

            # Verify content extraction
            if not text_content:
                raise ValueError("Failed to extract document content")

            # Generate chunks with explicit error handling
            try:
                chunks = self._chunk_text(text_content)
                await update_progress(1, 3, f"Generated {len(chunks)} chunks")
            except Exception as e:
                print(f"Chunking error: {str(e)}")
                chunks = [text_content]

            # Initialize embedding generation
            chunk_embeddings = []
            embedding_errors = []
            
            # Process chunks in batches with error tracking
            batch_size = 5
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = []
                
                for chunk in batch_chunks:
                    try:
                        embedding = await self.ollama_client.generate_embeddings(
                            text=chunk,
                            model=EMBED_MODEL
                        )
                        
                        # Convert embedding to serializable format
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        batch_embeddings.append(embedding)
                        
                    except Exception as e:
                        print(f"Embedding error for chunk {i}: {str(e)}")
                        embedding_errors.append(str(e))
                        batch_embeddings.append(None)
                    
                    await update_progress(
                        i + len(batch_embeddings),
                        total_chunks,
                        f"Generated embeddings for chunk {i + len(batch_embeddings)} of {total_chunks}"
                    )
                
                chunk_embeddings.extend(batch_embeddings)

            await update_progress(total_chunks, total_chunks, "Processing complete")

            # Create document summary from first chunk
            document_summary = chunks[0][:500] if chunks else ""

            # Prepare response with improved error handling
            response = {
                'type': 'document',
                'format': Path(file.name).suffix[1:],
                'summary': document_summary,
                'chunks': chunks,
                'chunk_embeddings': chunk_embeddings,
                'metadata': {
                    'filename': file.name,
                    'size': file.size,
                    'mime_type': file.type,
                    'created_at': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'embedding_count': len([e for e in chunk_embeddings if e is not None]),
                    'embedding_errors': embedding_errors if embedding_errors else None,
                },
                'vector_context': {
                    'type': 'document_chunks',
                    'chunk_count': len(chunks),
                    'embedding_count': len([e for e in chunk_embeddings if e is not None]),
                    'timestamp': datetime.now().isoformat()
                }
            }

            if embedding_errors:
                print(f"Warning: {len(embedding_errors)} chunks failed embedding generation")
                
            return response

        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")

    async def _extract_pdf_text(self, file) -> str:
        """Extract text from PDF with improved error handling."""
        try:
            with pdfplumber.open(file) as pdf:
                text_parts = []
                for page in pdf.pages:
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    except Exception as e:
                        print(f"Error extracting page text: {str(e)}")
                        continue
                
                return "\n\n".join(text_parts)
                
        except Exception as e:
            print(f"PDF extraction error: {str(e)}")
            try:
                # Fallback to raw text extraction
                return file.read().decode('utf-8', errors='ignore')
            except Exception as e2:
                print(f"Fallback extraction failed: {str(e2)}")
                return ""

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text using both character and token-based splitting."""
        try:
            # First split by characters
            char_chunks = self.text_splitter.split_text(text)
            
            # Then refine with token-based splitting
            token_chunks = []
            for chunk in char_chunks:
                token_chunks.extend(self.token_splitter.split_text(chunk))
            
            return token_chunks
        except Exception as e:
            print(f"Error chunking text: {str(e)}")
            return [text]  # Return single chunk if splitting fails

    async def _extract_pdf_text(self, file) -> str:
        """Extract text from PDF file with async support."""
        try:
            with pdfplumber.open(file) as pdf:
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
                return "\n".join(text)
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            return ""

    async def process_file(self, file) -> Dict[str, Any]:
        """Process uploaded file based on its type with RAG enhancements."""
        # Validate file
        is_valid, error = self.validate_file(file)
        if not is_valid:
            raise ValueError(error)
        
        # Determine file type
        file_type = self.get_file_type(file.name)
        if not file_type:
            raise ValueError(ERROR_MESSAGES["invalid_file_type"])
        
        # Process based on type
        if file_type == 'image':
            return await self.process_image(file)
        elif file_type == 'document':
            return await self.process_document(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def get_file_preview(self, processed_file: Dict[str, Any]) -> Optional[str]:
        """Generate HTML preview for the processed file."""
        file_type = processed_file['type']
        
        if file_type == 'image':
            img_data = processed_file['data']
            return f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; border-radius: 0.5rem;">'
        
        elif file_type == 'document':
            format_type = processed_file['format']
            preview_html = []
            
            if format_type in ['csv', 'xlsx']:
                # Create a preview table for spreadsheets
                df = pd.DataFrame(processed_file['content']['data'])
                preview_html.append(
                    df.head().to_html(
                        classes=['dataframe'],
                        index=False,
                        escape=False,
                        border=0
                    )
                )
                
            elif format_type == 'txt':
                # Show first few lines of text
                content = processed_file['content']
                preview_lines = '\n'.join(content.split('\n')[:5])
                preview_html.append(
                    f'<pre style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 0.5rem;">{preview_lines}</pre>'
                )
                
            elif format_type == 'pdf':
                preview_html.append('<div class="pdf-preview">PDF document (preview not available)</div>')
                
            # Add metadata display
            if 'metadata' in processed_file:
                preview_html.append(
                    f"""<div class="metadata-section" style="margin-top: 1rem;">
                        <strong>File Information:</strong><br/>
                        Size: {processed_file['metadata'].get('size', 'N/A')}<br/>
                        Type: {processed_file['metadata'].get('mime_type', 'N/A')}<br/>
                        Created: {processed_file['metadata'].get('created_at', 'N/A')}
                    </div>"""
                )
                
            return '\n'.join(preview_html)
        
        return None

    async def get_relevant_chunks(self, query: str, processed_file: Dict[str, Any], top_k: int = 3) -> List[str]:
        """Retrieve most relevant chunks with proper async handling."""
        if 'chunks' not in processed_file or 'embeddings' not in processed_file:
            return []

        try:
            # Generate query embedding
            query_embedding = await self.ollama_client.generate_embeddings(
                text=query,
                model=EMBED_MODEL
            )
            
            # Ensure embeddings are lists
            query_embedding_list = query_embedding if isinstance(query_embedding, list) else query_embedding.tolist()
            
            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(processed_file['embeddings']):
                chunk_embedding_list = chunk_embedding if isinstance(chunk_embedding, list) else chunk_embedding.tolist()
                
                similarity = np.dot(query_embedding_list, chunk_embedding_list) / (
                    np.linalg.norm(query_embedding_list) * np.linalg.norm(chunk_embedding_list)
                )
                similarities.append((similarity, i))
            
            # Sort by similarity and get top-k chunks
            similarities.sort(reverse=True)
            relevant_chunks = [
                processed_file['chunks'][idx]
                for _, idx in similarities[:top_k]
            ]
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error getting relevant chunks: {str(e)}")
            return []

    def format_for_chat(self, processed_file: Dict[str, Any]) -> str:
        """Format processed file data for chat context."""
        file_type = processed_file['type']
        
        if file_type == 'image':
            return f"[Image uploaded - {processed_file['metadata']['size'][0]}x{processed_file['metadata']['size'][1]} {processed_file['metadata']['format']}]"
        
        elif file_type == 'document':
            format_type = processed_file['format']
            if format_type in ['csv', 'xlsx']:
                summary = processed_file['content']['summary']
                return f"[{format_type.upper()} document uploaded - {summary['rows']} rows, {summary['columns']} columns]"
            else:
                return f"[Document uploaded - {format_type.upper()}, {processed_file['metadata']['size']} bytes]"