import json
import requests
from typing import Dict, List, AsyncGenerator, Optional, Any, Union
import httpx
import numpy as np
from config import (
    OLLAMA_BASE_URL,
    ERROR_MESSAGES,
    MODEL_PARAMETERS,
    DEFAULT_MODEL_PARAMS,
    EMBED_MODEL
)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")

    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch models: {str(e)}")

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get model info: {str(e)}")

    async def generate_chat_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a chat response with RAG support."""
        endpoint = f"{self.base_url}/api/chat"

        # Extract known parameters with RAG optimization
        options = {
            "num_ctx": kwargs.get('num_ctx', MODEL_PARAMETERS["context_length"]),
            "num_thread": kwargs.get('num_thread', DEFAULT_MODEL_PARAMS.get("num_thread", 8)),
            "temperature": kwargs.get('temperature', DEFAULT_MODEL_PARAMS["temperature"]),
            "top_p": kwargs.get('top_p', DEFAULT_MODEL_PARAMS["top_p"]),
            "presence_penalty": kwargs.get('presence_penalty', DEFAULT_MODEL_PARAMS["presence_penalty"]),
            "frequency_penalty": kwargs.get('frequency_penalty', DEFAULT_MODEL_PARAMS["frequency_penalty"]),
        }

        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream('POST', endpoint, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise httpx.HTTPError(f"HTTP {response.status_code}: {error_text.decode()}")

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)

                            if 'error' in data:
                                raise Exception(data['error'])

                            if data.get("done", False):
                                return

                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content

                        except json.JSONDecodeError as e:
                            print(f"Failed to decode JSON: {line}")
                            continue

            except httpx.TimeoutException:
                raise Exception("Request timed out while waiting for the model response")
            except httpx.HTTPError as e:
                raise Exception(f"HTTP error occurred: {str(e)}")
            except Exception as e:
                raise Exception(f"Error generating chat response: {str(e)}")

    async def generate_completion(
        self,
        model: str,
        prompt: str,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a completion with RAG optimization."""
        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream('POST', endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("done"):
                                break
                            if response_text := data.get("response"):
                                yield response_text
                        except json.JSONDecodeError:
                            continue
            except httpx.HTTPError as e:
                error_text = await response.aread() # Read error response content
                logger.error(f"Completion generation failed: {str(e)}, Response: {error_text.decode()}") # Log error response
                raise Exception(f"Completion generation failed: {str(e)}")

    async def generate_embeddings(self, text: Union[str, List[str]], model: str = EMBED_MODEL) -> Union[List[float], List[List[float]]]:
        """Generate embeddings with proper async handling."""
        try:
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text

            embeddings = []
            batch_size = 32

            async with httpx.AsyncClient() as client:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = []

                    for t in batch:
                        response = await client.post(
                            f"{self.base_url}/api/embed",
                            json={
                                "model": model,
                                "input": t,
                                "options": {
                                    "num_ctx": MODEL_PARAMETERS["context_length"],
                                    "num_thread": DEFAULT_MODEL_PARAMS["num_thread"]
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        #logger.info(f"Ollama API response: {result}") # Log the full API response
                        batch_embeddings.append(result.get("embedding", []))

                    embeddings.extend(batch_embeddings)

            # Convert to numpy array and then to list for serialization
            embeddings_array = np.array(embeddings)
            final_embeddings = embeddings_array.tolist() if len(texts) > 1 else embeddings_array[0].tolist()
            logger.info(f"Generated embeddings of shape: {embeddings_array.shape}")  # Log embedding shape
            logger.info(f"Input text for embedding: {texts[:1]}") # Log first input text
            return final_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")  # Log error with logger
            raise

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get available parameters for a model."""
        try:
            model_info = self.get_model_info(model_name)
            return model_info.get("parameters", {})
        except Exception as e:
            raise Exception(f"Failed to get model parameters: {str(e)}")

    async def load_model(self, model_name: str) -> None:
        """Preload a model into memory."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": ""  # Empty prompt just loads the model
                    }
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise Exception(f"Failed to load model: {str(e)}")

    async def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": "0"
                    }
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise Exception(f"Failed to unload model: {str(e)}")

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Get list of currently running models."""
        try:
            response = requests.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get running models: {str(e)}")

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            raise Exception(f"Failed to calculate similarity: {str(e)}")

    async def batch_generate_embeddings(
    self,
    texts: List[str],
    batch_size: int = 32,
    model: str = EMBED_MODEL
) -> List[List[float]]:
        """Generate embeddings in batches with progress tracking."""
        try:
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                for text in batch:
                    embedding = await self.generate_embeddings(text, model)
                    batch_embeddings.append(embedding)
                all_embeddings.extend(batch_embeddings)
                print(f"Processed batch {i//batch_size + 1}/{total_batches}")

            return all_embeddings
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")