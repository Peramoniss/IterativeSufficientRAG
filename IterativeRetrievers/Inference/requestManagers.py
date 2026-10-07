import requests
import time
import subprocess
import ollama
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from typing import override
import httpx
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import traceback
 
class RequestManager(ABC):
    def __init__(self, url:str, timeout:int, api_key:str):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
 
    @abstractmethod
    def query_model(self, messages: list, model: str, temperature: float = 0.0, **kwargs):
        """
        Send `messages` to `model` and return the response.
        Concrete implementations define their own extra kwargs
        (e.g. streaming, thinking mode, max tokens, context length).
        """
        raise NotImplementedError
   
    def complete(self, prompt, model: str, temperature: float = 0.0, **kwargs):
        raise NotImplementedError
   
 
class OllamaRequestManager(RequestManager):
    def __init__(self, url:str="http://localhost:11434", timeout:int=120, api_key:str="ollama"):
        super().__init__(url=url, api_key=api_key, timeout=timeout)
        self.client = ollama.Client(
            host=url,
            timeout=timeout,  # seconds
            headers={
                'Authorization': f'Bearer {api_key}'
            }
        )
 
    # Caso se depare com o erro: "ConnectError: [Errno 111] Connection refused", execute essa função para reiniciar
    def start_ollama(self):
        """Inicia um servidor local do ollama, que funciona em http://localhost:11434/api"""
        _ = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
 
        time.sleep(3)
 
    @override
    def query_model(self, messages, model: str = "qwen3.5:4b", think=False, temperature: float = 0, context_length: int = 4096) -> ollama.ChatResponse:
        """Gera resposta com o modelo de LLM"""
        # Gera resposta usando o modelo definido para as mensagens acumuladas no chat até então
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                think=think, # True, False, 'low', 'medium', 'high'
                options={"temperature": temperature, "num_ctx": context_length}
            )
        except Exception as e:
            # traceback.print_exc()
            return None
 
        return response["message"]["content"], response["prompt_eval_count"], response["eval_count"]
   
    @override
    def complete(self, prompt, model: str = "qwen3.5:4b", temperature: float = 0, context_length: int = 8192):
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                options={"temperature": temperature, "num_ctx": context_length}
            )
        except Exception as e:
            print("Exception in complete:", e)
            return None

        return response["response"], response.get("prompt_eval_count"), response.get("eval_count")
 
    def query_model_requests(self, messages, model, temperature=0.0, timeout=180, context_length:int = 512, log=True):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
 
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": context_length}
        }
 
        try:
            if log:
                print(f"Sending request to {model}...")
 
            response = requests.post(self.url, headers=headers, json=payload, timeout=timeout)
 
            if response.status_code != 200:
                if log:
                    print(f"Server Error: {response.status_code} - {response.text}")
                return None
 
            return response.json()
        except requests.exceptions.Timeout:
            if log:
                print("Error: The request timed out (the model took too long to load).")
            try:
                stop_payload = {
                    "model": model,
                    "keep_alive": 0
                }
                # Use a very short timeout here; we just want to fire the signal
                requests.post("http://localhost:11434/api/chat", json=stop_payload, timeout=10)
                if log:
                    print(f"Successfully sent stop signal. GPU cleared.")
            except Exception as stop_err:
                if log:
                    print(f"Failed to send stop signal: {stop_err}")
        except Exception as e:
            if log:
                print(f"API Request Error: {e}")
            return None
 
class ExternalRequestManager(RequestManager):
    def __init__(self, url:str, api_key:str, timeout:int=120, http_client=None):
        super().__init__(url=url, api_key=api_key, timeout=timeout)
        if http_client == None:
            http_client = httpx.Client(timeout=httpx.Timeout(timeout))
        else:
            http_client.timeout = httpx.Timeout(timeout)
 
        self.client = OpenAI(
            api_key=api_key,
            base_url=url,
            default_headers={
                "api-key": api_key  # Matches your HTTP snippet header
            },
            # timeout=timeout,
            http_client=http_client
        )
 
    @override
    def query_model(self, messages, model:str, max_response_tokens=2000, temperature=0.0, force_json = False):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0000000000, #0.0000000000 1.0
            max_tokens=max_response_tokens,
            top_p=0.0001,
            frequency_penalty=0,
            presence_penalty=0,
            seed = 42,
            response_format= { "type": "json_object" } if force_json else None
        )
        usage = response.usage
 
        answer = response.choices[0].message.content
        return answer, usage.prompt_tokens, usage.completion_tokens
   
    @override
    def complete(self, messages, model:str, max_response_tokens=2000, temperature=0.0): # DEPRECATED FOR MOST COMMERCIAL MODELS
        response = self.client.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_response_tokens,
            top_p=0.01,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={ "type": "json_object" }
        )
        return response.choices[0]
 
 
class FlanT5RequestManager(RequestManager):
    """
    Local FLAN-T5 request manager for IRCoT reproduction.
 
    FLAN-T5 uses relative position biases, so it can process sequences
    longer than its 512-token fine-tuning length without erroring --
    quality may degrade gradually past that point, but there's no hard
    architectural wall the way there is with absolute position embeddings.
    `max_input_tokens` defaults well above 512 accordingly; tune it down
    if you hit memory issues (encoder self-attention is O(n^2) in memory).
    """
 
    def __init__(self, model_name: str = "google/flan-t5-xl", device: str = None,
                 max_input_tokens: int = 4096,
                 url: str = "", api_key: str = "", timeout: int = 120):
        super().__init__(url=url, api_key=api_key, timeout=timeout)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Override the tokenizer's default 512 cap so truncation=True
        # downstream doesn't silently clip below what we actually want.
        self.tokenizer.model_max_length = max_input_tokens
 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.bfloat16).to(self.device)
        self.model.eval()
 
    def _flatten_messages(self, messages: list) -> str:
        if isinstance(messages, str):
            return messages
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            else:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)
 
    @override
    def query_model(self, messages, model: str = None, temperature: float = 0.0,
                     max_response_tokens: int = 512, **kwargs):
        prompt = self._flatten_messages(messages)
        return self.complete(prompt, model=model, temperature=temperature,
                              max_response_tokens=max_response_tokens, **kwargs)
 
    @override
    def complete(self, prompt: str, model: str = None, temperature: float = 0.0,
                 max_response_tokens: int = 512, **kwargs):
        try:
            raw_len = self.tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"].shape[-1]
            if raw_len > self.max_input_tokens:
                print(f"Warning: prompt is {raw_len} tokens, truncating to "
                      f"{self.max_input_tokens}. Consider trimming exemplars/docs "
                      f"if this happens often, since content is lost from the end.")
 
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=self.max_input_tokens).to(self.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
 
            do_sample = temperature > 0.0
            gen_kwargs = {"max_new_tokens": max_response_tokens, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = temperature
 
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
 
            completion_tokens = output_ids.shape[-1]
            content = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
 
            return content, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"FLAN-T5 generation error: {e}")
            return None