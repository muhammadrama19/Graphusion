from typing_extensions import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, Any
from langchain_core.language_models.llms import LLM
import json
import os
import requests

try:  # Optional dependency for Gemini-based runs
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime when feature is used
    genai = None
    types = None
    GOOGLE_GENAI_AVAILABLE = False

# Default Ollama endpoint
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class Triple(TypedDict):
    s: Annotated[str, ..., "Subject of the extracted Knowledge Graph Triple"]
    p: Annotated[str, ..., "Relation of the extracted Knowledge Graph Triple"]
    o: Annotated[str, ..., "Object of the extracted Knowledge Graph Triple"]


class Triples(BaseModel):
    triples: List[Triple]


class KnowledgeGraphLLM(LLM):
    model_name: str
    max_tokens: int
    provider: str = "auto"  # "openai", "google", "ollama", or "auto" for detection
    ollama_base_url: str = OLLAMA_BASE_URL

    @property
    def _llm_type(self) -> str:
        return f"Candidate Triple Extraction Chain based on {self.model_name}"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        provider = self._resolve_provider()
        
        if provider == "google":
            return self._call_gemini(prompt)
        elif provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return self._call_openai(prompt)

    def _call_openai(self, prompt: str) -> str:
        llm = ChatOpenAI(model=self.model_name, max_tokens=self.max_tokens)
        llm = llm.with_structured_output(Triples)
        stream_response = llm.stream(prompt)
        response = self.get_last_chunk(stream_response)

        if not hasattr(response, 'triples'):
            return "None"
        return json.dumps(response.triples).replace('\n', '')

    def _call_gemini(self, prompt: str) -> str:
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError(
                "google-genai is required for Gemini models. Install it via: pip install google-genai"
            )
        
        # Initialize the client (uses GOOGLE_API_KEY env var automatically)
        client = genai.Client()
        
        # Create the structured output schema for triples
        triple_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "triples": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "s": types.Schema(type=types.Type.STRING, description="Subject of the extracted Knowledge Graph Triple"),
                            "p": types.Schema(type=types.Type.STRING, description="Relation of the extracted Knowledge Graph Triple"),
                            "o": types.Schema(type=types.Type.STRING, description="Object of the extracted Knowledge Graph Triple"),
                        },
                        required=["s", "p", "o"],
                    ),
                ),
            },
            required=["triples"],
        )
        
        # Generate content with structured output
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
                response_schema=triple_schema,
            ),
        )
        
        # Parse the JSON response
        try:
            result = json.loads(response.text)
            if "triples" not in result:
                return "None"
            return json.dumps(result["triples"]).replace('\n', '')
        except (json.JSONDecodeError, AttributeError):
            return "None"

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API running on localhost."""
        # Extract actual model name if prefixed with "ollama:"
        model_name = self.model_name
        if model_name.lower().startswith("ollama:"):
            model_name = model_name[7:]  # Remove "ollama:" prefix
        
        # Build the system prompt for structured JSON output
        system_prompt = """You are a knowledge graph extraction assistant. Extract triples from the given text.
You MUST respond with valid JSON only, in this exact format:
{"triples": [{"s": "subject", "p": "predicate", "o": "object"}, ...]}
Do not include any other text, explanation, or markdown formatting."""
        
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Extract the message content
            content = result.get("message", {}).get("content", "")
            
            # Parse the JSON response
            try:
                parsed = json.loads(content)
                if "triples" not in parsed:
                    return "None"
                return json.dumps(parsed["triples"]).replace('\n', '')
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{[\s\S]*"triples"[\s\S]*\}', content)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        return json.dumps(parsed.get("triples", [])).replace('\n', '')
                    except json.JSONDecodeError:
                        pass
                return "None"
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running (ollama serve) and the URL is correct."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def _resolve_provider(self) -> str:
        if self.provider != "auto":
            return self.provider
        lower_name = self.model_name.lower()
        if lower_name.startswith("gemini"):
            return "google"
        if lower_name.startswith("ollama:"):
            return "ollama"
        return "openai"

    @staticmethod
    def get_last_chunk(stream_response):
        last_chunk = None
        for chunk in stream_response:
            last_chunk = chunk
        return last_chunk