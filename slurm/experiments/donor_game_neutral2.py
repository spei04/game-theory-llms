# imports
import random
import json
import time
import numpy as np
import os
import pandas as pd
from collections import Counter
from threading import Lock, Thread
import threading
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import textwrap
#from openai import InternalServerError

import openai
import anthropic
from anthropic import InternalServerError
from google import genai as google_genai
from google.genai import types as genai_types

import re
import difflib
import datetime
from dotenv import load_dotenv
load_dotenv()

# Force CUDA paths for bitsandbytes on the cluster (must be set before torch/transformers import)
# Skipped gracefully on CPU-only nodes (e.g. API runs on <cpu-partition> partition)
import ctypes

cuda_path = "/home/software/cuda/11.3/lib64/libcudart.so.11.0"
try:
    # 1. Manually load the library into the process address space
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)
    # 2. Set the environment variables for child threads
    os.environ["CUDA_HOME"] = "/home/software/cuda/11.3"
    os.environ["LD_LIBRARY_PATH"] = "/home/software/cuda/11.3/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["BNB_CUDA_VERSION"] = "113"
    print(f"[CUDA] Loaded {cuda_path} (GPU mode)")
except OSError:
    print(f"[CUDA] {cuda_path} not found — running in CPU/API mode")

# 3. Now it is safe to import transformers/bitsandbytes (only needed for local GPU models)
try:
    import torch
except ImportError:
    torch = None

# Expose HUGGINGFACE_TOKEN as HF_TOKEN so transformers/huggingface_hub pick it up automatically
# (avoids calling login() repeatedly which hits the whoami rate limit)
if not os.environ.get('HF_TOKEN') and os.environ.get('HUGGINGFACE_TOKEN'):
    os.environ['HF_TOKEN'] = os.environ['HUGGINGFACE_TOKEN']

# Set API keys
ANTHROPIC_API_KEY=os.environ.get('ANTHROPIC_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
google_client = google_genai.Client(api_key=GOOGLE_API_KEY)

# Ollama client for local open-source models
# Install: https://ollama.com  |  Pull a model: ollama pull llama3.2
# Latest recommended models: llama3.2, llama3.3, phi4, qwen2.5, deepseek-r1, gemma3
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
ollama_client = openai.OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

# vLLM server client (Apptainer-hosted OpenAI-compatible endpoint)
# Set VLLM_SERVER_HOST to point at the running vLLM server (default: localhost:8000)
VLLM_SERVER_HOST = os.environ.get('VLLM_SERVER_HOST', 'http://localhost:8000')
VLLM_SERVER_MODEL = os.environ.get('VLLM_SERVER_MODEL', '')
vllm_server_client = openai.OpenAI(base_url=f"{VLLM_SERVER_HOST}/v1", api_key="vllm")

# Groq client for cloud-hosted open-source models (fast inference, free tier available)
# Get a free API key at: https://console.groq.com
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
groq_client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

# HuggingFace Inference API client (OpenAI-compatible router)
# Uses HF_TOKEN (or HUGGINGFACE_TOKEN) — same token as local HF models
HF_INFERENCE_API_KEY = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN', '')
hf_inference_client = openai.OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_INFERENCE_API_KEY,
)

# Groq-hosted open-source models (set model_str to one of these to use Groq)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",       # Meta Llama 3.3 70B — best quality
    "llama-3.1-8b-instant",          # Meta Llama 3.1 8B  — fastest / cheapest
    "deepseek-r1-distill-llama-70b", # DeepSeek R1 distilled — strong reasoning
    "mixtral-8x7b-32768",            # Mistral MoE — long context (32k)
    "gemma2-9b-it",                  # Google Gemma 2 9B
    "qwen-qwq-32b",                  # Qwen QwQ 32B — reasoning model
]

# vllm models — native FP8/BF16 inference on GPU (proper quantization support)
# Install: pip install vllm
VLLM_MODELS = {
    "Qwen/Qwen3-32B-FP8",    # FP8 quantized, ~32GB VRAM on L40S (Qwen3ForCausalLM)
    "Qwen/Qwen3.5-27B-FP8",  # FP8 quantized — requires vLLM with Qwen3_5 support
}
_vllm_engine = None
_vllm_lock = threading.Lock()

def _get_vllm_engine():
    """Lazy-load vllm LLM engine once and reuse."""
    global _vllm_engine
    if _vllm_engine is None:
        with _vllm_lock:
            if _vllm_engine is None:
                from vllm import LLM
                print(f"Loading {model_str} via vllm (may download model)...")
                _vllm_engine = LLM(model=model_str, dtype="auto", gpu_memory_utilization=0.85)
                print(f"{model_str} loaded.")
    return _vllm_engine

# GGUF models loaded via llama-cpp-python (CPU or GPU via CUBLAS)
# Install: pip install llama-cpp-python
# For GPU on cluster: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
LLAMACPP_MODELS = {
    "qwen3.5-27b": {
        "repo_id": "unsloth/Qwen3.5-27B-GGUF",
        "filename": "Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf",  # ~16GB (was BF16 ~54GB)
    },
}
_llamacpp_model = None
_llamacpp_lock = threading.Lock()

def _get_llamacpp_model():
    """Lazy-load the llama-cpp model once and reuse across all calls."""
    global _llamacpp_model
    if _llamacpp_model is None:
        with _llamacpp_lock:
            if _llamacpp_model is None:
                from llama_cpp import Llama
                cfg = LLAMACPP_MODELS[model_str]
                print(f"Loading {model_str} from {cfg['repo_id']} (may download GGUF)...")
                _llamacpp_model = Llama.from_pretrained(
                    repo_id=cfg["repo_id"],
                    filename=cfg["filename"],
                    n_ctx=4096,
                    n_gpu_layers=-1,  # offload all layers to GPU
                    verbose=False,
                )
                print(f"{model_str} loaded.")
    return _llamacpp_model

# HuggingFace local models (runs on MPS/GPU/CPU via transformers pipeline)
# On a cluster: set HF_TOKEN env var in your job script — transformers picks it up automatically
# Locally: HUGGINGFACE_TOKEN in .env is used
HF_MODELS = [
    "google/gemma-2-2b",           # Gemma 2 2B base model
    "google/gemma-2-2b-it",        # Gemma 2 2B instruction-tuned
    "Qwen/Qwen2.5-7B-Instruct",    # Qwen 2.5 7B BF16 (~14GB VRAM, no quantization needed)
]
_hf_pipeline = None
_hf_pipeline_lock = threading.Lock()
_hf_inference_lock = threading.Lock()  # Serialize inference to avoid OOM on large models

# AWQ quantized models via autoawq (no CUDA compilation needed)
# Install: pip install autoawq
AWQ_MODELS = {
    "qwen2.5-7b-awq":  "Qwen/Qwen2.5-7B-Instruct-AWQ",   # ~4GB VRAM
    "qwen2.5-14b-awq": "Qwen/Qwen2.5-14B-Instruct-AWQ",  # ~8GB VRAM
    "qwen2.5-32b-awq": "Qwen/Qwen2.5-32B-Instruct-AWQ",  # ~18GB VRAM
}
_awq_model = None
_awq_tokenizer = None
_awq_lock = threading.Lock()
_awq_inference_lock = threading.Lock()

def _get_awq_model():
    """Lazy-load AWQ quantized model via autoawq directly (bypasses transformers AWQ backend)."""
    global _awq_model, _awq_tokenizer
    if _awq_model is None:
        with _awq_lock:
            if _awq_model is None:
                from awq import AutoAWQForCausalLM
                from transformers import AutoTokenizer
                model_id = AWQ_MODELS[model_str]
                hf_token = os.environ.get('HF_TOKEN', '')
                print(f"Loading {model_id} (AWQ via autoawq, may download ~4GB)...")
                _awq_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
                _awq_model = AutoAWQForCausalLM.from_quantized(
                    model_id, fuse_layers=True, token=hf_token
                )
                print(f"{model_id} loaded.")
    return _awq_model, _awq_tokenizer

# 4-bit quantized HuggingFace models via bitsandbytes (requires GPU + bitsandbytes)
# Install: pip install bitsandbytes
# On cluster: module load cuda/11.3 before running
BNB_MODELS = {
    "qwen3.5-27b-4bit":  "Qwen/Qwen3.5-27B",              # ~14GB VRAM (no separate Instruct variant)
    "qwen2.5-72b-4bit":  "Qwen/Qwen2.5-72B-Instruct",   # ~38GB VRAM (A100 40/80GB)
    "qwen2.5-32b-4bit":  "Qwen/Qwen2.5-32B-Instruct",   # ~18GB VRAM
    "qwen2.5-14b-4bit":  "Qwen/Qwen2.5-14B-Instruct",   # ~8GB VRAM
}
_bnb_model = None
_bnb_tokenizer = None
_bnb_lock = threading.Lock()

def _get_bnb_model():
    """Lazy-load a 4-bit quantized HuggingFace model via bitsandbytes."""
    global _bnb_model, _bnb_tokenizer
    if _bnb_model is None:
        with _bnb_lock:
            if _bnb_model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                import torch
                model_id = BNB_MODELS[model_str]
                hf_token = os.environ.get('HF_TOKEN', '')
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print(f"Loading {model_id} in 4-bit (may download ~38GB on first run)...")
                _bnb_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
                _bnb_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=hf_token,
                )
                print(f"{model_id} loaded.")
    return _bnb_model, _bnb_tokenizer

def _get_hf_pipeline():
    """Lazy-load the HuggingFace pipeline once and reuse across all calls."""
    global _hf_pipeline
    if _hf_pipeline is None:
        with _hf_pipeline_lock:
            if _hf_pipeline is None:
                from transformers import pipeline as hf_pipeline_fn
                import torch
                hf_token = os.environ.get('HF_TOKEN', '')
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
                print(f"Loading {model_str} on {device} (first call — may download model)...")
                # Pass token directly — avoids login() which rate-limits on whoami
                # Use device_map="auto" for large models (e.g. 27B FP8) to distribute layers across GPU memory
                import torch
                if device == "cuda":
                    _hf_pipeline = hf_pipeline_fn("text-generation", model=model_str, device_map="auto",
                                                   torch_dtype="auto", token=hf_token)
                else:
                    _hf_pipeline = hf_pipeline_fn("text-generation", model=model_str, device=device, token=hf_token)
                # Clear conflicting max_length from model's generation config to suppress warning
                if hasattr(_hf_pipeline.model, 'generation_config'):
                    _hf_pipeline.model.generation_config.max_length = None
                print(f"{model_str} loaded.")
    return _hf_pipeline

# Token usage tracking (thread-safe)
_token_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
_token_lock = threading.Lock()
COST_CAP_USD = float(os.environ.get("COST_CAP", "10.0"))  # Abort the run if estimated spend exceeds this amount

class CostCapExceeded(Exception):
    pass

def _track_usage(input_tokens: int, output_tokens: int):
    """Accumulate token counts across all API calls; raise if cost cap exceeded."""
    with _token_lock:
        _token_usage["input_tokens"] += input_tokens
        _token_usage["output_tokens"] += output_tokens
        _token_usage["api_calls"] += 1
        costs = _TOKEN_COSTS.get(model_str, {"input": 0.0, "output": 0.0})
        total_cost = (
            _token_usage["input_tokens"]  / 1_000_000 * costs["input"] +
            _token_usage["output_tokens"] / 1_000_000 * costs["output"]
        )
        if total_cost >= COST_CAP_USD:
            raise CostCapExceeded(
                f"Cost cap of ${COST_CAP_USD:.2f} reached (estimated ${total_cost:.4f}). Aborting run."
            )

# Approximate cost per million tokens (USD) for each model
_TOKEN_COSTS = {
    "claude-3-5-sonnet":             {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-6":             {"input": 3.00,  "output": 15.00},
    "claude-3-sonnet":               {"input": 3.00,  "output": 15.00},
    "claude-3-opus":                 {"input": 15.00, "output": 75.00},
    "claude-3-haiku":                {"input": 0.25,  "output": 1.25},
    "gpt-4o":                        {"input": 2.50,  "output": 10.00},
    "gpt-4":                         {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":                 {"input": 0.50,  "output": 1.50},
    "gemini-1.5-flash":              {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":                {"input": 1.25,  "output": 5.00},
    "llama-3.3-70b-versatile":       {"input": 0.59,  "output": 0.79},
    "llama-3.1-8b-instant":          {"input": 0.05,  "output": 0.08},
    "deepseek-r1-distill-llama-70b": {"input": 0.75,  "output": 0.99},
    "mixtral-8x7b-32768":            {"input": 0.24,  "output": 0.24},
    "gemma2-9b-it":                  {"input": 0.20,  "output": 0.20},
    "qwen-qwq-32b":                  {"input": 0.29,  "output": 0.39},
}

def print_token_summary():
    """Print a summary of tokens used and estimated cost."""
    costs = _TOKEN_COSTS.get(model_str, {"input": 0.0, "output": 0.0})
    input_cost  = _token_usage["input_tokens"]  / 1_000_000 * costs["input"]
    output_cost = _token_usage["output_tokens"] / 1_000_000 * costs["output"]
    total_cost  = input_cost + output_cost
    print("\n" + "="*50)
    print(f"  TOKEN USAGE SUMMARY  (model: {model_str})")
    print("="*50)
    print(f"  API calls    : {_token_usage['api_calls']}")
    print(f"  Input tokens : {_token_usage['input_tokens']:,}")
    print(f"  Output tokens: {_token_usage['output_tokens']:,}")
    print(f"  Total tokens : {_token_usage['input_tokens'] + _token_usage['output_tokens']:,}")
    print(f"  Est. cost    : ${total_cost:.4f} USD")
    print("="*50)

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(checkpoint_path, generation_idx, agents, simulation_data,
                    all_strategies_by_generation, prev_gen_strategies,
                    all_average_final_resources, all_donations, all_final_scores,
                    all_final_reputations, conditional_survival):
    """Atomically save simulation state after a completed generation."""
    checkpoint = {
        "generation_idx": generation_idx,
        "agents": [asdict(a) for a in agents],
        "simulation_data": simulation_data.to_dict(),
        "all_strategies_by_generation": all_strategies_by_generation,
        "prev_gen_strategies": prev_gen_strategies,
        "all_average_final_resources": all_average_final_resources,
        "all_donations": all_donations,
        "all_final_scores": all_final_scores,
        "all_final_reputations": all_final_reputations,
        "conditional_survival": conditional_survival,
        "token_usage": dict(_token_usage),
    }
    os.makedirs(os.path.dirname(checkpoint_path) or "simulation_results", exist_ok=True)
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(checkpoint, f)
    os.replace(tmp_path, checkpoint_path)  # atomic write
    print(f"  [Checkpoint saved → generation {generation_idx + 1} complete]")

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and reconstruct Agent and SimulationData objects."""
    with open(checkpoint_path) as f:
        cp = json.load(f)

    agents = []
    for d in cp["agents"]:
        agent = Agent(
            name=d["name"],
            resources=d["resources"],
            reputation=d["reputation"],
            total_donated=d["total_donated"],
            potential_donated=d["potential_donated"],
            history=d["history"],
            strategy=d["strategy"],
            strategy_justification=d["strategy_justification"],
            strategy_response=d["strategy_response"],
            total_final_score=d["total_final_score"],
            average_reputation=d["average_reputation"],
            traces=d["traces"],
            old_traces=d["old_traces"],
            punishment=d["punishment"],
            invalid_responses=d["invalid_responses"],
        )
        agents.append(agent)

    sim_data_dict = cp["simulation_data"]
    simulation_data = SimulationData(
        hyperparameters=sim_data_dict["hyperparameters"],
        agents_data=sim_data_dict["agents_data"],
        generation_stats=sim_data_dict["generation_stats"],
        generation_summaries=sim_data_dict["generation_summaries"],
    )

    with _token_lock:
        _token_usage.update(cp["token_usage"])

    return (
        cp["generation_idx"],
        agents,
        simulation_data,
        cp["all_strategies_by_generation"],
        cp["prev_gen_strategies"],
        cp["all_average_final_resources"],
        cp["all_donations"],
        cp["all_final_scores"],
        cp["all_final_reputations"],
        cp["conditional_survival"],
    )

# Create a global lock
print_lock = threading.Lock()
model_lock = threading.Lock()  # Add model lock

# Rate limiting class
class RateLimiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_request_time = {}
        
        # Rate limits (requests per minute) — set for paid API tiers.
        self.rate_limits = {
            "gpt-4o": 500,  # Tier-1 paid: 500 RPM
            "gpt-4": 500,
            "gpt-3.5-turbo": 500,
            "o1-mini": 500,
        }
    
    def wait_if_needed(self, model_name):
        """Wait if needed to respect rate limits for GPT models only"""
        if model_name not in self.rate_limits:
            return  # No rate limiting for non-GPT models (Claude, Gemini, etc.)
        
        rate_limit = self.rate_limits[model_name]
        min_interval = 60.0 / rate_limit  # Seconds between requests
        
        with self.lock:
            current_time = time.time()
            last_time = self.last_request_time.get(model_name, 0)
            
            time_since_last = current_time - last_time
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                print(f"Rate limiting {model_name}: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            self.last_request_time[model_name] = time.time()

# Global rate limiter instance
rate_limiter = RateLimiter()

@dataclass
class Agent:
    name: str
    resources: int
    reputation: float
    total_donated: int = 0
    # round_donated: float = 0.0  # Remove to match og_code.py
    potential_donated: int = 0
    history: list = field(default_factory=list)
    strategy: str = ""
    strategy_justification: str = ""
    strategy_response: str = ""  # Store the full LLM response for strategy generation
    total_final_score: int = 0
    average_reputation: float = 0
    traces: list[list[str]] = field(default_factory=lambda: [[]])
    old_traces: list[list[str]] = field(default_factory=lambda: [[]])
    punishment: int = 0
    invalid_responses: int = 0  # Add counter for invalid responses
    savings: float = 0.0  # Accumulated resources from donations received; cannot be spent
    inspired_by: str = ""         # Name of the agent declared as inspiration (any "Inherited from:" mention)
    inherited_from: str = ""      # Name of the agent whose strategy was copied exactly (generation 2+)
    inherited_rank: int = 0       # Rank (1=best) of the copied strategy among survivors
    inherited_strategy: str = ""  # The actual strategy text that was copied
    top_ranked_agent_name: str = ""  # Name of the rank-1 survivor from the previous generation

    def donate(self, amount):
        """ Handle the donation process for the agent """
        if 0 <= amount <= self.resources:
            self.resources -= amount
            self.total_donated += amount
        self.potential_donated += self.resources + amount

@dataclass
class SimulationData:
    hyperparameters: dict
    agents_data: list = field(default_factory=list)
    generation_stats: list = field(default_factory=list)
    generation_summaries: list = field(default_factory=list) 

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'agents_data': self.agents_data,
            'generation_stats': self.generation_stats,
            'generation_summaries': self.generation_summaries,  
        }
# Set model name and load model
#model_str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
#model_str = "google/gemma-3-4b-it"
#model_str = "gpt-4o"
#model_str = "claude-3-haiku"
#model_str = "gemini-1.5-flash"
#model_str = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
#model_str = "mlx-community/Phi-2-2.7B-4bit"
# === Groq open-source models (requires GROQ_API_KEY in .env) ===
#model_str = "llama-3.3-70b-versatile"
#model_str = "llama-3.1-8b-instant"
#model_str = "deepseek-r1-distill-llama-70b"
#model_str = "mixtral-8x7b-32768"
#model_str = "qwen-qwq-32b"
# === HuggingFace local models (requires HUGGINGFACE_TOKEN in .env or HF_TOKEN env var) ===
#model_str = "google/gemma-2-2b-it"
#model_str = "google/gemma-2-2b"  # base model — does not follow instruction format
#model_str = "gemma2-9b-it"       # decommissioned by Groq
# === GGUF models via llama-cpp-python (requires: pip install llama-cpp-python) ===
# For GPU on cluster: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
#model_str = "qwen3.5-27b"         # Qwen 3.5 27B GGUF (unsloth/Qwen3.5-27B-GGUF)
# === 4-bit quantized HuggingFace models (requires GPU + bitsandbytes) ===
# On cluster: module load cuda/11.3 && pip install bitsandbytes
# Model selected via MODEL_STR env var (set in SLURM script) — falls back to qwen2.5-72b-4bit
model_str = os.environ.get('MODEL_STR', 'qwen2.5-72b-4bit')
# === API models (no GPU, use <cpu-partition> partition) ===
# gemini-1.5-flash | gpt-4o | claude-3-5-sonnet
# === GPU models (<gpu-partition> + bitsandbytes) ===
# qwen2.5-72b-4bit | qwen3.5-27b-4bit (needs Python 3.10+)

@dataclass
class AgentRoundData:
    agent_name: str
    round_number: int
    game_number: int
    paired_with: str
    current_generation: int
    resources: int
    donated: float
    received: float
    strategy: str
    strategy_justification: str
    reputation: float
    is_donor: bool
    traces: list[list[str]]
    history: list
    justification: str = ""
    punished: bool = False

def _normalize_strategy(s: str) -> str:
    """Strip metadata and normalize whitespace for similarity comparison."""
    s = re.sub(r'\n*\s*Inherited from:\s*\S[^\n]*', '', s, flags=re.IGNORECASE)
    s = s.strip().rstrip('.*:**').strip()
    return re.sub(r'\s+', ' ', s).lower()

def strategy_is_unmodified(new_strategy: str, inherited_strategy: str, threshold: float = 0.92) -> bool:
    """Return True if the new strategy is essentially the same as the inherited one."""
    ratio = difflib.SequenceMatcher(
        None,
        _normalize_strategy(new_strategy),
        _normalize_strategy(inherited_strategy),
    ).ratio()
    return ratio >= threshold

def parse_strategy_output(output):
    parts = output.split("My strategy will be", 1)
    if len(parts) == 2:
        strategy_justification = parts[0].strip()
        strategy = "My strategy will be " + parts[1].strip()
    else:
        strategy_justification = ""
        strategy = output.strip()
    return strategy_justification, strategy

def generate_strategy(agent_name: str, generationNumber: int, inherited_strategies: str,
                      ranked_survivor_names: list = None, survivor_strategy_map: dict = None) -> str:
    if punishment_mechanism == "costly_punishment":
      punishment_text = f"You may also choose to punish the recipient by spending x units to take away {punishmentLoss}x of their resources. Bear in mind that others may punish you too."
    elif punishment_mechanism == "partner_choice":
      punishment_text = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
    else:
      punishment_text = ""
    
    # Strategy generation prompt should match the original donor_game_new framing.
    # Baseline behavior: agents only see their current partner's immediately preceding round.
    history_text = """As a donor, you will receive limited information about the recipient and their recent interactions:
                You will see what the recipient (let's call them A) did in the immediately preceding round only.
                Example:
                You are paired with A.
                You know: In round 3, A donated X% to B.
                Remember: This is all the information you have about A. You don't know what A did in rounds 1 or 2, only in round 3."""
    
    if generationNumber == 1:
      prompt = (
        f"Your name is {agent_name}. How will you play? "
        f"{history_text}"
        f"{punishment_text}"
        "In the very first round, you get no info about the receiver's past behavior. "
        "Think about how to split your 10 units in a way that helps you personally but also keeps things stable over many rounds. "
        "Before writing your final plan, briefly reason step by step about what works well here. "
        "Then write a strategy that starts with the words: My strategy will be…"
      )
      strategy_output = promptLLM(prompt)
      strategy_justification, strategy = parse_strategy_output(strategy_output)
      print(f"\n{agent_name} Strategy: {strategy} ")
      return strategy_justification, strategy, strategy_output, "", "", 0, ""
    else:
      prompt = (
        f"Your name is {agent_name}. How will you play? "
        "Before you decide, you will be shown the strategies + final scores of the best-performing 50% from the previous generation. "
        f"{inherited_strategies}"
        f"{history_text}"
        f"{punishment_text}"
        "In the very first round, you get no info about the receiver's past behavior. "
        "Think about how to split your 10 units in a way that helps you personally but also keeps things stable over many rounds. "
        "Before writing your final plan, briefly reason step by step about what works well here — especially how you might improve upon the strategies you see. "
        "Then write a strategy that starts with the words: My strategy will be… "
        "If your strategy borrows from or is inspired by one of the previous agents shown above, add on a new line: Inherited from: [agent_name]"
      )
    strategy_output = promptLLM(prompt)
    strategy_justification, strategy = parse_strategy_output(strategy_output)

    # Parse which agent was referenced, then check for unmodified adoption
    inspired_by = ""
    inherited_from = ""
    inherited_rank = 0
    inherited_strategy = ""
    m = re.search(r'Inherited from:\s*(\S+)', strategy_output)
    if m:
        candidate = m.group(1).strip().rstrip('.')
        if (survivor_strategy_map or {}).get(candidate):
            inspired_by = candidate  # always record any declared inspiration
        candidate_strategy = (survivor_strategy_map or {}).get(candidate, "")
        # Only record inheritance if the strategy was adopted completely without modification
        if candidate_strategy and strategy_is_unmodified(strategy, candidate_strategy):
            inherited_from = candidate
            if ranked_survivor_names:
                try:
                    inherited_rank = ranked_survivor_names.index(inherited_from) + 1
                except ValueError:
                    inherited_rank = 0
            inherited_strategy = candidate_strategy

    inherited_label = f"copied from {inherited_from} (rank {inherited_rank})" if inherited_from else (f"inspired by {inspired_by}" if inspired_by else "original")
    print(f"\n{agent_name} Strategy: {strategy} [{inherited_label}]")
    return strategy_justification, strategy, strategy_output, inspired_by, inherited_from, inherited_rank, inherited_strategy

def initializeAgents(numAgents: int, initialEndowment: int, generationNumber: int,
                     inherited_strategies: str, ranked_survivor_names: list = None,
                     survivor_strategy_map: dict = None, top_ranked_agent: str = "") -> list:
    agents = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(numAgents):
            name = f"{generationNumber}_{i+1}"
            futures.append(executor.submit(generate_strategy, str(name), generationNumber,
                                           inherited_strategies, ranked_survivor_names,
                                           survivor_strategy_map))

        # Collect results and create agents
        for i, future in enumerate(futures):
            strategy_justification, new_strategy, strategy_output, inspired_by, inherited_from, inherited_rank, inherited_strategy = future.result()
            name = f"{generationNumber}_{i+1}"
            agent = Agent(name=name, reputation=False, resources=initialEndowment,
                          strategy=new_strategy, strategy_justification=strategy_justification,
                          strategy_response=strategy_output,
                          inspired_by=inspired_by,
                          inherited_from=inherited_from, inherited_rank=inherited_rank,
                          inherited_strategy=inherited_strategy,
                          top_ranked_agent_name=top_ranked_agent)
            agents.append(agent)

    random.shuffle(agents)
    return agents

def bipartiteRoundRobin(agents):
  num_agents = len(agents)
  assert num_agents % 2 == 0, "Number of agents must be even."
  group_A = agents[:num_agents // 2]
  group_B = agents[num_agents // 2:]
  rounds = []
  toggle_roles = False
  # We rotate group B around group A, group A is static in this example
  for i in range(len(group_A)):
    # Rotate group B
    rotated_group_B = group_B[-i:] + group_B[:-i]
    if toggle_roles:
      round_pairings = list(zip(rotated_group_B, group_A))
    else:
      round_pairings = list(zip(group_A, rotated_group_B))
    rounds.append(round_pairings)
    toggle_roles = not toggle_roles
  return rounds

def extendRounds(original_rounds):
    extended_rounds = original_rounds.copy()

    for round in original_rounds:
        reversed_round = [(b, a) for a, b in round]
        extended_rounds.append(reversed_round)

    return extended_rounds

def calculate_received_amount(punishment_mechanism, refused, cooperationGain, response, punishmentLoss, action=None):
    if punishment_mechanism == "partner_choice":
        return cooperationGain * response if not refused else 0
    elif punishment_mechanism == "costly_punishment":
        if action is None:
            raise ValueError("Action must be specified for costly_punishment mechanism")
        if action == 'donate':
            return cooperationGain * response
        elif action == 'punish':
            return -punishmentLoss * response
        else:
            raise ValueError(f"Unknown action for costly_punishment: {action}")
    elif punishment_mechanism == 'none':
        return cooperationGain * response
    else:
        raise ValueError(f"Unknown punishment mechanism: {punishment_mechanism}")

def handle_pairing_thread_safe(donor, recipient, round_index, generation, game_number, agent_locks, donation_records, agent_updates, round_start_resources):
    action_info = ""
    donor_data = None
    recipient_data = None
    punished = False
    action = 'donate'
    justification = ""
    response = 0

    # Baseline behavior: donors only see their current partner's immediately preceding round.
    recipient_behavior = ""
    if recipient.traces and recipient.traces[-1]:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            last_action = last_trace[-1] if last_trace else ""
            recipient_behavior = f"In the previous round, {recipient.name} did the following: {last_action}"
        else:
            recipient_behavior = f"In the previous round, {recipient.name} did the following: {str(last_trace)}"

    with agent_locks[donor.name], agent_locks[recipient.name]:
        prompt = donorPrompt(
            donor, generation, round_index + 1, recipient
        )

        valid_response = False
        max_attempts = 10
        attempts = 0
        full_response = ""

        while not valid_response and attempts < max_attempts:
            try:
                #print(f"\nPrompt for {donor.name}:")
                #print(prompt)
                #print(f"{'='*50}\n")
                
                full_response = promptLLM(prompt, timeout=30)
                #print(full_response)
                parts = full_response.split('Answer:', 1)
                
                if len(parts) == 2:
                    justification = parts[0].replace('Justification:', '').strip()
                    answer_part = parts[1].strip()

                    if punishment_mechanism == "partner_choice":
                        if "refuse" in answer_part.lower():
                            action = 'refuse'
                            response = 0
                            valid_response = True
                        else:
                            match = re.search(r'^\s*(\d+(?:\.\d+)?)', answer_part)
                            if match:
                                action = 'donate'
                                response = float(match.group(1))
                                valid_response = True

                    elif punishment_mechanism == "costly_punishment":
                        match = re.search(r'(donate|punish).*?(\d+(?:[.,]\d+)?)', answer_part, re.IGNORECASE)
                        if match:
                            action = match.group(1).lower()
                            response = float(match.group(2).replace(',', '.'))
                            valid_response = True

                    else:  # No punishment mechanism
                        match = re.search(r'^\s*(\d+(?:\.\d+)?)', answer_part)
                        if match:
                            action = 'donate'
                            response = float(match.group(1))
                            valid_response = True

                if not valid_response:
                    print(f"Invalid response from {donor.name} in round {round_index + 1}. Retrying...")
                    attempts += 1
            except ValueError:
                print(f"Invalid numerical response from {donor.name} in round {round_index + 1}")
                print(full_response)
                attempts += 1
            except TimeoutError:
                print(f"LLM call timed out for {donor.name} in round {round_index + 1}")
                attempts += 1

        if not valid_response:
            print(f"Failed to get a valid response from {donor.name} after {max_attempts} attempts")
            action = 'donate'
            response = 0
            print(f"Defaulting to donating 0 units")

    if action == 'refuse':
        action_info = (
            f"{donor.name} refused to play with {recipient.name}.\n"
            f"Resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f} \n"
            f"Recipient traces: {recipient_behavior} \n"
            #f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
        )
        new_trace = recipient.traces[-1].copy() if recipient.traces else []
        new_trace.append(f"In round {round_index + 1}, {donor.name} refused to play with {recipient.name}.")
        donor.traces.append(new_trace)
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
            f"You refused to play."
            f"{get_history_text(recipient.traces, history_mode)}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
            f"They refused to play."
            f"{get_history_text(donor.traces, history_mode)}"
        )
    elif 0 <= response:
        if response > donor.resources:
            print(f"  [{donor.name}] Donation {response} exceeds endowment {donor.resources:.2f} — capping at {donor.resources:.2f}")
            response = donor.resources
        if action == 'donate':
            resources_before = donor.resources
            percentage_donated = response / resources_before if resources_before != 0 else 1
            donor.resources -= response
            donor.total_donated += response
            donor.potential_donated += donor.resources + response
            recipient.savings += cooperationGain * response
            action_info = (
                f"\n{donor.name}: -{response} ({percentage_donated:.2%}) and {recipient.name}: +{cooperationGain * response}.\n"
                f"Previous: {donor.name} round allowance: {resources_before:.2f}, {recipient.name} savings: {recipient.savings-(cooperationGain*response):.2f}.\n"
                f"New: {donor.name} round allowance: {donor.resources:.2f}, {recipient.name} savings: {recipient.savings:.2f}.\n"
                #f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
            )

            new_trace = recipient.traces[-1].copy() if recipient.traces else []
            new_trace.append(f"In round {round_index + 1}, {donor.name} donated {percentage_donated * 100:.2f}% of their resources to {recipient.name}.")
            donor.traces.append(new_trace)

            donor_history = (
                f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
                f"You gave up {response} units, and they received {cooperationGain * response} units."
                f"{get_history_text(recipient.traces, history_mode)}"
            )

            recipient_history = (
                f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
                f"They gave up {response} units, and you received {cooperationGain * response} units."
                f"{get_history_text(donor.traces, history_mode)}"
            )

            if donor.reputation == False:
                donor.reputation = percentage_donated
            else:
                donor.reputation = ((1 - abs(percentage_donated - recipient.reputation)) + discounted_value * donor.reputation) / (1 + discounted_value)

            # donor.round_donated += response  # Remove to match og_code.py

    elif action == 'punish':
        punished = True
        # Calculate percentage using resources BEFORE punishment
        resources_before = donor.resources
        percentage_donated = response / resources_before if resources_before != 0 else 1
        donor.resources -= response
        donor.total_donated += response
        donor.potential_donated += donor.resources + response
        recipient.savings = max(0, recipient.savings - punishmentLoss * response)
        action_info = (
            f"{donor.name}: -{response} ({percentage_donated:.2%}) and {recipient.name}: - {punishmentLoss * response}.\n"
            f"Previous: {donor.name} round allowance: {resources_before:.2f}, {recipient.name} savings: {recipient.savings+(punishmentLoss*response):.2f}."
            f"New: {donor.name} round allowance: {donor.resources:.2f}, {recipient.name} savings: {recipient.savings:.2f}.\n"
            f"Recipient traces: {recipient_behavior} \n"
            #f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
        )

        new_trace = recipient.traces[-1].copy() if recipient.traces else []
        new_trace.append(f"In round {round_index + 1}, {donor.name} punished {recipient.name} by spending {response} units to take away {punishmentLoss * response} units from their resources.")
        donor.traces.append(new_trace)

        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
            f"You punished them by giving up {response} units to take away {punishmentLoss * response} units from them."
            f"{get_history_text(recipient.traces, history_mode)}"
        )

        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
            f"They punished you by giving up {response} units to take away {punishmentLoss * response} units from you."
            f"{get_history_text(donor.traces, history_mode)}"
        )

    else:
        action_info = (
            f"{donor.name} attempted an invalid action.\n"
            f"Resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f} \n"
            f"Recipient traces: {recipient_behavior} \n"
            #f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
        )
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
            f"You attempted an invalid action."
            f"{get_history_text(recipient.traces, history_mode)}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
            f"They attempted an invalid action."
            f"{get_history_text(donor.traces, history_mode)}"
        )

    donor.history.append(donor_history)
    recipient.history.append(recipient_history)

    donor_data = AgentRoundData(
            agent_name=donor.name,
            round_number=round_index + 1,
            game_number=game_number,
            paired_with=recipient.name,
            current_generation=generation,
            resources=donor.resources,
            donated=response if action != 'refuse' else 0,
            received=0,
            strategy=donor.strategy,
            strategy_justification=donor.strategy_justification,
            reputation=donor.reputation,
            is_donor=True,
            traces=donor.traces,
            history=donor.history,
            justification=justification,
            punished=punished,
        )
    recipient_data = AgentRoundData(
            agent_name=recipient.name,
            round_number=round_index + 1,
            game_number=game_number,
            paired_with=donor.name,
            current_generation=generation,
            resources=recipient.resources,
            donated=0,
            received=calculate_received_amount(punishment_mechanism, action == 'refuse', cooperationGain, response, punishmentLoss, action),
            strategy=recipient.strategy,
            strategy_justification=recipient.strategy_justification,
            reputation=recipient.reputation,
            is_donor=False,
            traces=recipient.traces,
            history=recipient.history,
            punished=False,
            justification="",  # Recipients don't provide justifications
        )

    return action_info, donor_data, recipient_data

def donorGame(agents: list, rounds: list, generation: int, simulation_data: SimulationData) -> (list, list, list):
    fullHistory = []
    donation_records = Queue()
    agent_updates = Queue()

    # Create locks for each agent
    agent_locks = {agent.name: Lock() for agent in agents}

    def play_game(game_number, game_rounds):
        round_results = {i: [] for i in range(len(game_rounds))}
        
        # Track cumulative scores separately
        cumulative_scores = {agent.name: 0 for agent in agents}
        
        # Track donation percentages per round
        round_donation_percentages = []

        for round_index, round_pairings in enumerate(game_rounds):
            # Each round: flush previous round's remaining allowance into savings, then give fresh 10 units
            for agent in agents:
                agent.savings += agent.resources
                agent.resources = initial_endowment
            # Track resources at start of round for percentage calculations
            round_start_resources = {agent.name: agent.resources for agent in agents}
            #for agent in agents:
                #agent.round_donated = 0.0
            
            print(f"Starting Round {round_index + 1} - Game {game_number} - Generation {generation}")
            
            if round_index == 0:
                # Initialize traces for the first round
                for agent in agents:
                    agent.traces = [[f"{agent.name} did not have any previous interactions."]]

            # Reduce concurrent workers to avoid rate limits -> originally max_workers=min(len(round_pairings), 10)
            with ThreadPoolExecutor(max_workers=min(len(round_pairings), 10)) as executor: 
                futures = []
                for donor, recipient in round_pairings:
                    if round_index > 0:
                        donor.traces.append(recipient.traces[-1].copy())
                    future = executor.submit(
                        handle_pairing_thread_safe,
                        donor, recipient, round_index, generation, game_number,
                        agent_locks, donation_records, agent_updates, round_start_resources
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    action_info, donor_data, recipient_data = future.result()
                    if action_info:
                        round_results[round_index].append(action_info)
                    if donor_data and recipient_data:
                        simulation_data.agents_data.append(asdict(donor_data))
                        simulation_data.agents_data.append(asdict(recipient_data))

            # Calculate donation percentages for this round
            round_donations = []
            """for agent in agents:
                if agent.round_donated > 0:
                    # Calculate percentage based on resources at start of round
                    start_resources = round_start_resources[agent.name]
                    donation_percentage = (agent.round_donated / start_resources) * 100 if start_resources > 0 else 0
                    round_donations.append(donation_percentage)"""
            
            if round_donations:
                avg_donation_pct = np.mean(round_donations)
                round_donation_percentages.append(avg_donation_pct)
            else:
                round_donation_percentages.append(0)

            # Add total wealth (savings + remaining round allowance) to cumulative scores
            for agent in agents:
                cumulative_scores[agent.name] += agent.savings + agent.resources

        return round_results, cumulative_scores, round_donation_percentages

    # Play the first game
    game1_results, game1_scores, game1_donations = play_game(1, rounds)

    # Compile results for Game 1
    for round_index in range(len(rounds)):
        fullHistory.append(f"Round {round_index + 1} (Game 1):\n")
        fullHistory.extend(game1_results[round_index])

    # Apply updates after all threads have completed
    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    # Calculate and print average total wealth for Game 1
    average_resources_game1 = sum(agent.savings + agent.resources for agent in agents) / len(agents)

    # Store Game 1 final reputations
    game1_reputations = {agent.name: agent.reputation for agent in agents}

    # Reset for Game 2: flush remaining round allowance into savings, clear traces (full independence)
    for agent in agents:
        agent.savings += agent.resources  # carry unspent round allowance into savings
        agent.resources = 0  # reset; will be set fresh at start of each round
        agent_generation = int(agent.name.split('_')[0])
        if agent_generation < generation:  # This is a surviving agent
            agent.reputation = agent.average_reputation  # Use the average reputation from previous generation
        else:
            agent.reputation = False
        agent.traces.clear()  # Game 2 is independent — no traces carry over
        agent.history.clear()

    # Generate pairings for Game 2
    reversed_rounds = [[tuple(reversed(pair)) for pair in round_pairings] for round_pairings in rounds]

    # Play the second game
    game2_results, game2_scores, game2_donations = play_game(2, reversed_rounds)

    # Compile results for Game 2
    for round_index in range(len(reversed_rounds)):
        fullHistory.append(f"Round {round_index + 1} (Game 2):\n")
        fullHistory.extend(game2_results[round_index])

    # Apply updates after all threads have completed
    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    # Calculate and print average total wealth for Game 2
    average_resources_game2 = sum(agent.savings + agent.resources for agent in agents) / len(agents)

    # Calculate final scores and reputations (total accumulated savings across both games)
    for agent in agents:
        agent.total_final_score = agent.savings + agent.resources
        agent.average_reputation = (game1_reputations[agent.name] + agent.reputation) / 2 if agent.reputation is not False else game1_reputations[agent.name]

    overall_average_resources = (average_resources_game1 + average_resources_game2) / 2
    all_average_final_resources.append(overall_average_resources)

    #with print_lock:
        #print(''.join(fullHistory))

    return fullHistory, list(donation_records.queue), game1_donations + game2_donations

def get_last_three_reversed(item):
    if isinstance(item, list):
        return " ".join(item[-3:][::-1])
    elif isinstance(item, str):
        return item
    else:
        return str(item)  # Convert to string if it's neither a list nor a string

def get_all_reversed(item):
    if isinstance(item, list):
        return " ".join(item[::-1]) 
    elif isinstance(item, str):
        return item
    else:
        return str(item)  # Convert to string if it's neither a list nor a string

def get_history_text(traces, history_mode):
    """Get history text based on history_mode setting"""
    if not traces or len(traces) == 0:
        return ""
    
    if history_mode == "simple":
        # Only show the last action
        if traces[-1] and len(traces[-1]) > 0:
            return traces[-1][-1]
        return ""
    elif history_mode == "3_traces":
        # Show last 3 actions
        return get_last_three_reversed(traces[-1])
    elif history_mode == "full":
        # Show all actions
        return get_all_reversed(traces[-1])
    else:
        # Default to 3 traces for backward compatibility
        return get_last_three_reversed(traces[-1])
    
def donorPrompt(donor: Agent, generation: int, round: int, recipient: Agent) -> str:
    """Generate the prompt for the game interaction."""
    strategy_text = f"Remember the game plan you committed to earlier: {donor.strategy} " if donor.strategy else ""
    if punishment_mechanism=="partner_choice":
        punishment_text=f"If you do not wish to play with {recipient.name}, simply state: Refuse to play. Otherwise. state your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2=""
    elif punishment_mechanism=="costly_punishment":
        punishment_text=f"State your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperationGain}x units or punish them by taking away {punishmentLoss}x of their units. Remember, your answer should be simply 'Donate, x' or 'Punish, x' (where x is the real number of units you give up.)"
        punishment_text_2=f"Remember, state your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperationGain}x units or punish them by taking away {punishmentLoss}x of their units."
    else:
        punishment_text=f"Give your answer as a real number from 0 to {donor.resources}."
        punishment_text_2=""
    # Baseline behavior: only show the current partner's immediately preceding round.
    recipient_behavior = ""
    if recipient.traces and recipient.traces[-1]:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            last_action = last_trace[-1] if last_trace else ""
            recipient_behavior = f" Last round, {recipient.name} did this: {last_action}"
        else:
            recipient_behavior = f" Last round, {recipient.name} did this: {str(last_trace)}"
    return (
        f"Your name is {donor.name}. "
        f"{strategy_text}"
        f"This is generation {generation}. We are now at round {round}. "
        "Important: The total number of rounds was not fixed in advance — after each round, there is an 80% chance the game continues for another round, and a 20% chance it ends. "
        f"This round, you are paired with {recipient.name}. They have saved up {recipient.savings:.1f} units so far."
        f"{recipient_behavior}"
        f"Today, you are the giver. You receive a brand-new {initial_endowment} units for this round. "
        f"You must pick a donation amount between 0 and {initial_endowment} inclusive. "
        f"Whatever you keep stays in your stash (never usable again). "
        f"Whatever you give is multiplied by {cooperationGain} for the receiver. "
        f"Question: Out of your {initial_endowment} units, how many do you hand over? "
        f"{punishment_text}"
        "First, think briefly about how your choice follows your strategy in this exact situation. "
        "Then write your full response in exactly this shape:"
        "Justification:"
        "[several lines explaining your reasoning here]"
        " "
        "Answer:"
        "[your number here]"
        "Remember: Justification first, then the answer line."
        f"{punishment_text_2}"
    )

def altDonorPrompt(donor: Agent, generation: int, round: int, recipient: Agent) -> str:
    """Generate the alternate prompt for the game interaction."""
    return donorPrompt(donor, generation, round, recipient).replace(
        "integer", "real number"
    )

def save_simulation_data(simulation_data, folder_path='simulation_results'):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract hyperparameters for the file name
    params = simulation_data.hyperparameters
    num_generations = params.get('numGenerations')
    num_agents = params.get('numAgents')
    selection_method = params.get('selectionMethod')
    #client = params.get('client')

    # Replace forward slashes with underscores in model name
    safe_model_name = model_str.replace('/', '_')

    # Create an informative file name
    filename = f"Final_Accum_{safe_model_name}_coopGain_{cooperationGain}punLoss_{punishmentLoss}_{reputation_mechanism}_gen{num_generations}_agents{num_agents}_{selection_method}_{timestamp}.json"

    # Convert simulation_data to a dictionary
    data_dict = simulation_data.to_dict()

    # Function to make data JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return str(obj)

    # Apply the serialization function to the entire data dictionary
    serializable_data = make_serializable(data_dict)

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the full file path
    full_file_path = os.path.join(folder_path, filename)

    # Write the JSON data to the file
    with open(full_file_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Simulation data saved to: {full_file_path}")

def _call_openai_compatible(client, model, prompt, timeout, max_tokens=None):
    """Call any OpenAI-compatible client (Ollama, vLLM server, etc.) with standard parameters."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        timeout=timeout,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
    return response.choices[0].message.content

def promptLLM(prompt, max_retries=3, initial_wait=1, timeout=30):
    for attempt in range(max_retries):
        try:
            # Add rate limiting before each API call
            rate_limiter.wait_if_needed(model_str)
            if model_str == "gpt-3.5-turbo":
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif model_str == "gpt-4":
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif model_str == "gpt-4o":
                response = openai.chat.completions.create(
                    #model="gpt-4o-2024-08-06",  # unavailable on this project
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                         {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif model_str == "o1-mini":
                response = openai.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                         {"role": "user", "content": prompt}
                    ],
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif model_str == "claude-3-opus":
                response = anthropic.messages.create(
                    #model="claude-3-opus-20240229",  # unavailable on this account
                    model="claude-opus-4-20250514",   # oldest opus available
                    max_tokens=1000,
                    temperature=0.8,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.input_tokens, response.usage.output_tokens)
                return response.content[0].text

            elif model_str == "claude-3-sonnet":
                response = anthropic.messages.create(
                    #model="claude-3-sonnet-20240229",  # unavailable on this account
                    model="claude-sonnet-4-20250514",   # oldest sonnet available
                    max_tokens=1000,
                    temperature=0.8,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.input_tokens, response.usage.output_tokens)
                return response.content[0].text

            elif model_str == "claude-3-5-sonnet":
                response = anthropic.messages.create(
                    #model="claude-3-5-sonnet-20240620",  # unavailable on this account
                    #model="claude-3-5-sonnet-20241022",  # unavailable on this account
                    model="claude-sonnet-4-5-20250929",   # oldest available on this account
                    max_tokens=1000,
                    temperature=0.8,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.input_tokens, response.usage.output_tokens)
                return response.content[0].text

            elif model_str == "claude-sonnet-4-6":
                response = anthropic.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    temperature=0.8,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.input_tokens, response.usage.output_tokens)
                return response.content[0].text

            elif model_str == "claude-3-haiku":
                response = anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.8,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                _track_usage(response.usage.input_tokens, response.usage.output_tokens)
                return response.content[0].text

            elif model_str == "gemini-1.5-flash":
                response = google_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.8
                    )
                )
                _track_usage(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
                return response.text

            elif model_str == "gemini-1.5-pro":
                response = google_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.8,
                        safety_settings=[
                            genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                            genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                            genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        ]
                    )
                )
                _track_usage(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
                return response.text

            elif model_str in GROQ_MODELS:
                # Groq cloud-hosted open-source model
                response = groq_client.chat.completions.create(
                    model=model_str,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    timeout=timeout
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif model_str in HF_MODELS:
                # Local HuggingFace model via transformers pipeline (MPS/GPU/CPU)
                # Serialize inference to prevent concurrent GPU OOM on large models
                pipe = _get_hf_pipeline()
                input_text = f"{system_prompt}\n\n{prompt}"
                with _hf_inference_lock:
                    outputs = pipe(input_text, max_new_tokens=512, do_sample=True, temperature=0.8)
                full_text = outputs[0]["generated_text"]
                new_text = full_text[len(input_text):]  # strip the echoed input
                # Approximate token counts via tokenizer
                n_input = len(pipe.tokenizer.encode(input_text))
                n_output = len(pipe.tokenizer.encode(new_text))
                _track_usage(n_input, n_output)
                return new_text

            elif model_str in LLAMACPP_MODELS:
                # GGUF model via llama-cpp-python
                llm = _get_llamacpp_model()
                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.8,
                )
                text = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})
                _track_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                return text

            elif model_str in VLLM_MODELS:
                # vllm inference — proper FP8/quantization support on GPU
                from vllm import SamplingParams
                llm = _get_vllm_engine()
                tokenizer = llm.get_tokenizer()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                sampling_params = SamplingParams(temperature=0.8, max_tokens=512)
                outputs = llm.generate([formatted], sampling_params)
                text = outputs[0].outputs[0].text
                n_input = len(tokenizer.encode(formatted))
                n_output = len(tokenizer.encode(text))
                _track_usage(n_input, n_output)
                return text

            elif model_str in AWQ_MODELS:
                # AWQ quantized model via autoawq directly
                import torch
                model, tokenizer = _get_awq_model()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                text_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text_input, return_tensors="pt").to("cuda")
                with _awq_inference_lock:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=256, temperature=0.8, do_sample=True
                        )
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                result = tokenizer.decode(new_tokens, skip_special_tokens=True)
                _track_usage(inputs.input_ids.shape[1], len(new_tokens))
                return result

            elif model_str in BNB_MODELS:
                # 4-bit quantized HuggingFace model via bitsandbytes (GPU required)
                import torch
                model, tokenizer = _get_bnb_model()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                text_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=500, temperature=0.8, do_sample=True
                    )
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                result = tokenizer.decode(new_tokens, skip_special_tokens=True)
                _track_usage(inputs.input_ids.shape[1], len(new_tokens))
                return result

            elif VLLM_SERVER_MODEL and model_str == VLLM_SERVER_MODEL:
                # vLLM Apptainer server — OpenAI-compatible endpoint at VLLM_SERVER_HOST
                return _call_openai_compatible(vllm_server_client, model_str, prompt, timeout, max_tokens=512)

            else:
                # Fallback: local Ollama model (e.g. "llama3.2", "phi4", "mistral")
                # requires Ollama running at OLLAMA_HOST
                return _call_openai_compatible(ollama_client, model_str, prompt, timeout)

        except CostCapExceeded:
            raise  # propagate immediately — do not retry
        except (InternalServerError, Exception, TimeoutError) as e:
            # Check if it's a rate limit error
            error_str = str(e).lower()
            if "rate limit" in error_str or "too many requests" in error_str:
                # For rate limits, wait longer
                wait_time = 30  # Wait 30 sec for rate limits
                print(f"Rate limit error: {str(e)}. Waiting {wait_time} seconds...")
            else:
                # For other errors, use exponential backoff
                wait_time = initial_wait * (2 ** attempt)
                print(f"Error occurred: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception("Failed to get a response after multiple retries")

def selectTopAgents(agents: list) -> list:
    """Select the top half of agents based on resources."""
    return sorted(agents, key=lambda x: x.total_final_score, reverse=True)[:len(agents) // 2]

def selectRandomAgents(agents: list) -> list:
    """Select half of the agents randomly."""
    return random.sample(agents, len(agents) // 2)

def selectHighestReputation(agents: list) -> list:
  return sorted(agents, key=lambda agent: agent.average_reputation, reverse=True)[:len(agents) // 2]

# my addition to make selectionMethod == 'imageScore' work
def selectHighestImageScore(agents: list) -> list:
    return sorted(agents, key=lambda agent: agent.total_final_score, reverse=True)[:len(agents) // 2]

def runGenerations(numGenerations, numAgents, initialEndowment, selectionMethod):
    print("\n=== Starting Simulation ===")
    print(f"Parameters: {model_str}, {numGenerations} generations, {numAgents} agents, {initialEndowment} initial endowment, {selectionMethod} selection method, {history_mode} history\n")
    
    all_agents = []
    global all_donations
    all_donations = []
    global average_final_image_scores
    average_final_image_scores = []
    global all_average_final_resources
    all_average_final_resources = []
    global all_final_scores
    all_final_scores = []
    global all_final_reputations
    all_final_reputations = []
    conditional_survival = 0
    prev_gen_strategies = []
    all_strategies_by_generation = []

    # Initialize simulation data
    simulation_data = SimulationData(hyperparameters={
        "numGenerations": numGenerations,
        "numAgents": numAgents,
        "initialEndowment": initialEndowment,
        "selectionMethod": selectionMethod,
        "cooperationGain": cooperationGain,
        "include_strategy": include_strategy,
        "discountedValue": discounted_value,
        "llm": model_str,
        "system_prompt": system_prompt,
        "reputation_mechanism": reputation_mechanism,
        "punishment_mechanism": punishment_mechanism,
        "number_of_rounds": number_of_rounds
    })

    # Checkpoint path — deterministic so restarts find the same file.
    # RUN_ID distinguishes parallel runs; variant tag prevents collision with
    # other experiment files running the same model+params.
    safe_model = model_str.replace('/', '_')
    run_id = os.environ.get('RUN_ID', '0')
    output_dir = os.environ.get('OUTPUT_DIR', 'simulation_results')
    checkpoint_path = os.path.join(
        output_dir,
        f"checkpoint_baseline_{safe_model}_{numGenerations}gen_{numAgents}agents"
        f"_{number_of_rounds}rounds_{selectionMethod}_run{run_id}.json"
    )

    start_generation = 0
    if os.path.exists(checkpoint_path):
        print(f"\n[Checkpoint found: resuming from {checkpoint_path}]")
        (last_gen, agents, simulation_data, all_strategies_by_generation,
         prev_gen_strategies, all_average_final_resources, all_donations,
         all_final_scores, all_final_reputations, conditional_survival) = load_checkpoint(checkpoint_path)
        start_generation = last_gen + 1
        all_agents.extend(agents)
        print(f"[Resuming at generation {start_generation + 1} / {numGenerations}]")
    else:
        print("\nGenerating initial agent strategies...")
        agents = initializeAgents(numAgents, initialEndowment, 1, "No previous strategies.")
        all_agents.extend(agents)

    for i in range(start_generation, numGenerations):
        print(f"\n=== Generation {i + 1} ===")
        _gen_tokens_start = dict(_token_usage)  # snapshot for per-gen cost
        generation_info = f"Generation {i + 1}: \n"
        for agent in agents:
          agent.history.append(generation_info)
          prev_gen_strategies.append(agent.strategy)
          if int(agent.name.split('_')[0]) == i-1:
            conditional_survival +=1
        print(generation_info)

        # Create rounds using bipartiteRoundRobin
        initial_rounds = bipartiteRoundRobin(agents)
        print("\nRound pairings:")
        for round_idx, round_pairings in enumerate(initial_rounds):
            print(f"Round {round_idx + 1}:")
            for donor, recipient in round_pairings:
                print(f"  {donor.name} (donor) and {recipient.name} (recipient)")

        # Extend the rounds
        rounds = extendRounds(initial_rounds)

        generationHistory, donation_records, _donation_percentages_unused = donorGame(agents, rounds, i+1, simulation_data)
        all_donations.extend(donation_records)
        reputations = [agent.reputation for agent in agents]

        # Donation stats from raw agents_data (includes donated=0 donor-rounds).
        # This is more faithful than the older per-round donation tracking in donorGame.play_game.
        donor_percentages = {}  # agent_name -> [pct per donor-round] (includes zeros)
        pcts_by_round = {}      # (game_number, round_number) -> [pct across donors]
        all_donor_round_pcts = []
        max_round_per_game = {}

        for agent_data in simulation_data.agents_data:
            if not agent_data.get('is_donor', False):
                continue
            if agent_data.get('current_generation') != i + 1:
                continue
            if agent_data.get('punished', False):
                # Treat punishment separately from donation; exclude from donation percentage.
                continue

            agent_name = agent_data['agent_name']
            donated = float(agent_data.get('donated', 0) or 0)
            remaining = float(agent_data.get('resources', 0) or 0)  # donor resources after action
            denom = donated + remaining
            donation_pct = (donated / denom) * 100 if denom > 0 else 0.0

            donor_percentages.setdefault(agent_name, []).append(donation_pct)
            all_donor_round_pcts.append(donation_pct)

            game_no = int(agent_data.get('game_number', 1) or 1)
            round_no = int(agent_data.get('round_number', 0) or 0)
            pcts_by_round.setdefault((game_no, round_no), []).append(donation_pct)
            max_round_per_game[game_no] = max(max_round_per_game.get(game_no, 0), round_no)

        # Per-round averages in chronological order: Game 1 rounds, then Game 2 rounds.
        donation_percentages = []
        for game_no in sorted(max_round_per_game.keys()):
            for round_no in range(1, max_round_per_game[game_no] + 1):
                vals = pcts_by_round.get((game_no, round_no), [])
                donation_percentages.append(float(np.mean(vals)) if vals else 0.0)

        # Print end of generation summary
        summary_lines = []
        summary_lines.append(f"=== End of Generation {i + 1} Results ===")
        sorted_agents = sorted(agents, key=lambda x: x.total_final_score, reverse=True)
        for agent in sorted_agents:
            summary_lines.append(f"{agent.name}: {agent.resources:.2f} units")
            summary_lines.append(f"  Invalid Responses: {agent.invalid_responses}")

        #generation_avg = sum(agent.resources for agent in agents) / len(agents)
        # The overall_average_resources is calculated in donorGame function and stored in all_average_final_resources
        generation_avg = all_average_final_resources[-1] if all_average_final_resources else sum(agent.resources for agent in agents) / len(agents)
        summary_lines.append(f"\nAverage resources for Generation {i + 1}: {generation_avg:.2f} units")
        total_invalid = sum(agent.invalid_responses for agent in agents)
        summary_lines.append(f"Total Invalid Responses in Generation {i + 1}: {total_invalid}")

        # Print donation statistics
        if all_donor_round_pcts:
            avg_donation = float(np.mean(all_donor_round_pcts))
            summary_lines.append(f"\nAverage donation percentage for Generation {i + 1}: {avg_donation:.2f}%")

        # Print individual donor donation percentages
        if donor_percentages:
            summary_lines.append(f"\nIndividual donor donation percentages for Generation {i + 1}:")
            for agent_name, percentages in donor_percentages.items():
                avg_agent_donation = np.mean(percentages)
                summary_lines.append(f"  {agent_name}: {avg_agent_donation:.2f}% (rounds: {', '.join([f'{p:.1f}%' for p in percentages])})")

        print('\n'.join(summary_lines))

        # Per-generation token cost
        _costs = _TOKEN_COSTS.get(model_str, {"input": 0.0, "output": 0.0})
        _gen_in  = _token_usage["input_tokens"]  - _gen_tokens_start["input_tokens"]
        _gen_out = _token_usage["output_tokens"] - _gen_tokens_start["output_tokens"]
        _gen_cost = (_gen_in / 1e6 * _costs["input"]) + (_gen_out / 1e6 * _costs["output"])
        _cum_cost = (_token_usage["input_tokens"] / 1e6 * _costs["input"] +
                     _token_usage["output_tokens"] / 1e6 * _costs["output"])
        print(f"  Tokens this gen : {_gen_in + _gen_out:,} input+output  |  cost: ${_gen_cost:.4f}")
        print(f"  Cumulative cost : ${_cum_cost:.4f}")

        # Don't save generation_summaries - let figs.py calculate from raw data
        # simulation_data.generation_summaries.append('\n'.join(summary_lines))
        
        # Save generation stats
        if all_donor_round_pcts:
            generation_stats = {
                'generation': i + 1,
                'average_donation_percentage': float(np.mean(all_donor_round_pcts)),
                'donation_percentages_by_round': [float(pct) for pct in donation_percentages],
                'average_resources': float(generation_avg),
                'total_invalid_responses': total_invalid
            }
            simulation_data.generation_stats.append(generation_stats)

        # After the end of the generation, before selection:
        gen_strategies = []
        for agent in agents:
            gen_strategies.append({
                "generation": i + 1,
                "agent_name": agent.name,
                "strategy": agent.strategy,
                "strategy_response": agent.strategy_response,
                "inspired_by": agent.inspired_by or None,
                "inherited_from": agent.inherited_from or None,
                "inherited_rank": agent.inherited_rank or None,
                "inherited_strategy": agent.inherited_strategy or None,
                "top_ranked_agent": agent.top_ranked_agent_name or None,
            })
        all_strategies_by_generation.append(gen_strategies)

        if i < numGenerations - 1 and numGenerations > 1:
            if selectionMethod == 'top':
                surviving_agents = selectTopAgents(agents)
            elif selectionMethod == 'random':
                surviving_agents = selectRandomAgents(agents)
            elif selectionMethod == 'imageScore':
                surviving_agents = selectHighestImageScore(agents)
            elif selectionMethod == 'reputation':
                surviving_agents = selectHighestReputation(agents)
            else:
                raise ValueError("Invalid selection method. Choose 'top', 'random', 'imageScore', or 'reputation'.")

            print(f"\nSurviving agents for next generation: {[agent.name for agent in surviving_agents]}")

            if numGenerations > 1:
                # Provide the next generation with top-half strategies + scores (as promised in the prompt).
                surviving_agents_sorted = sorted(surviving_agents, key=lambda a: a.total_final_score, reverse=True)
                surviving_strategies = "\n".join(
                    [f"- {a.name}: final_score={a.total_final_score:.2f} | {a.strategy}" for a in surviving_agents_sorted]
                )
                ranked_survivor_names = [a.name for a in surviving_agents_sorted]
                survivor_strategy_map = {a.name: a.strategy for a in surviving_agents_sorted}
                top_ranked_agent = ranked_survivor_names[0] if ranked_survivor_names else ""
                for agent in surviving_agents:
                    agent.resources = 0  # will be set fresh at start of each round
                    agent.savings = 0.0  # reset savings for new generation
                    agent.old_traces = agent.traces

                new_agents = initializeAgents(numAgents // 2, initialEndowment, i + 2,
                                              surviving_strategies, ranked_survivor_names,
                                              survivor_strategy_map, top_ranked_agent)
                agents = (surviving_agents + new_agents)
                all_agents.extend(new_agents)
                random.shuffle(agents)

        # Save checkpoint after generation i is fully complete (including selection)
        save_checkpoint(checkpoint_path, i, agents, simulation_data,
                        all_strategies_by_generation, prev_gen_strategies,
                        all_average_final_resources, all_donations, all_final_scores,
                        all_final_reputations, conditional_survival)

    print("\n=== Simulation Complete ===")
    print(f"Results saved to {output_dir}")
    save_simulation_data(simulation_data, folder_path=output_dir)
    save_all_strategies_by_generation(all_strategies_by_generation, folder_path=output_dir)
    print_token_summary()
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("[Checkpoint deleted — simulation complete]")

    return agents

def save_all_strategies_by_generation(all_strategies_by_generation, folder_path='simulation_results'):
    import datetime, os
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_str.replace('/', '_')
    os.makedirs(folder_path, exist_ok=True)

    # Human-readable txt (original format)
    txt_path = os.path.join(folder_path, f"all_strategies_{safe_model_name}_{timestamp}.txt")
    with open(txt_path, "w") as f:
        for gen_strategies in all_strategies_by_generation:
            if gen_strategies:
                generation = gen_strategies[0]["generation"]
                f.write(f"=== Generation {generation} ===\n")
                for agent in gen_strategies:
                    f.write(f"Agent: {agent['agent_name']}\n")
                    f.write(f"Strategy: {agent['strategy']}\n")
                    if agent.get('inherited_from'):
                        f.write(f"Inherited from: {agent['inherited_from']} (rank {agent.get('inherited_rank', '?')})\n")
                        f.write(f"Inherited strategy: {agent.get('inherited_strategy', '')}\n")
                    f.write(f"Full Response: {agent['strategy_response']}\n")
    print(f"Strategies (txt) saved to: {txt_path}")

    # Structured JSON (includes inheritance fields for downstream analysis)
    json_path = os.path.join(folder_path, f"all_strategies_{safe_model_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_strategies_by_generation, f, indent=2)
    print(f"Strategies (json) saved to: {json_path}")

# =============================== Parameters ===============================
# All parameters can be overridden via environment variables
history_mode      = os.environ.get('HISTORY_MODE',          'simple')  # "simple", "3_traces", or "full"
cooperationGain   = float(os.environ.get('COOP_GAIN',       '2'))
punishmentLoss    = float(os.environ.get('PUNISHMENT_LOSS', '2'))
numGenerations    = int(os.environ.get('NUM_GENERATIONS',   10))
numAgents         = int(os.environ.get('NUM_AGENTS',        12))
number_of_rounds  = int(os.environ.get('NUM_ROUNDS',        12))
initial_endowment = int(os.environ.get('INITIAL_ENDOWMENT', '10'))
selection_method  = os.environ.get('SELECTION_METHOD',      'top')
punishment_mechanism = os.environ.get('PUNISHMENT_MECHANISM', 'none')

include_strategy = True
all_donations = []
all_average_final_resources = []
discounted_value = 0.5
reputation_mechanism=history_mode # modified this

if punishment_mechanism == "partner_choice":
    punishment_text_in_system_prompt = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
elif punishment_mechanism == "costly_punishment":
    punishment_text_in_system_prompt = f"You may also choose to punish the recipient by spending x units to take away {punishmentLoss}x units from them. Bear in mind that others may punish you too."
else:
    punishment_text_in_system_prompt = ""

# Create history mode specific text
if history_mode == "simple":
    history_text = """As a donor, you will receive limited information about the recipient and their recent interactions:
                You will see what the recipient (let's call them A) did in the immediately preceding round only.
                Example:
                You are paired with A.
                You know: In round 3, A donated X% to B.
                Remember: This is all the information you have about A. You don't know what A did in rounds 1 or 2, only in round 3."""
elif history_mode == "3_traces":
    history_text = """As a donor, you will receive the following information each round:
                1. Your own previous round: what you did and what your partner did in the round before this one (if it exists).
                2. Your current partner's (call them A) most recent round: what A did in the immediately preceding round.
                3. A's partner's most recent round: what A's partner from that round (call them B) did in their preceding round.
                Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round.
                Example (assuming it's now round 4):
                Your previous round: In round 3 (Game 1) you were paired with agent X. You gave up 3 units.
                You are now paired with A.
                You know: In round 3, A donated X% to B.
                You also know: In round 2, B donated Y% to C.
                Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."""
elif history_mode == "full":
    history_text = """As a donor, you will receive complete information about the recipient and their interaction history:
                You will see the full trace of what the recipient (let's call them A) did in all their previous rounds.
                You will also see the full trace of what A's previous partners did in their rounds.
                This gives you complete visibility into the recipient's entire interaction history.
                Example (assuming it's now round 4):
                You are paired with A.
                You know: In round 3, A donated X% to B.
                You also know: In round 2, A donated Y% to C.
                And also: In round 1, A donated Z% to D.
                Plus: In round 2, B donated W% to E.
                And: In round 1, C donated V% to F.
                Remember: You have access to the complete interaction history of A and their partners."""
else:
    raise ValueError("Invalid history_mode. Choose 'simple', '3_traces', or 'full'.")

if history_mode == "simple":
    _n2_history_text = (
        "When you are the giver, you only get a tiny bit of info about the receiver's recent past: "
        "just what that receiver (call them A) did in the immediately previous round. "
        "Example: \"You are matched with A. In round 3, A sent X% to B.\" "
        "That is the only clue about A's past. Nothing from earlier rounds. Just round 3."
    )
else:
    _n2_history_text = history_text  # fall back for 3_traces / full modes

system_prompt = (
    "You are an AI player taking part in a study where multiple agents interact repeatedly, "
    "sharing a common pot under controlled conditions. "
    "The study looks at how people behave when resources are limited and must be reallocated. Here's the setup. "
    f"Each round, you get exactly {initial_endowment} units of something valuable. "
    "In that round, you are randomly matched with one other person. "
    "One of you is the giver, the other the receiver. "
    f"The giver picks any whole or fractional amount from 0 up to {initial_endowment} to send across. "
    "Whatever the giver does NOT send stays in their own private stash — and that stash never gets used again in future rounds. "
    f"The receiver, however, gets {cooperationGain}x whatever the giver sent. "
    "If you give in one round, then in the very next round you will be the receiver, and vice versa — roles alternate without fail. "
    f"{_n2_history_text} "
    f"{punishment_text_in_system_prompt} "
    "Your mission: end the final round with as many total units as possible. "
    "After the last round, the top half of all players (those with highest totals) "
    "move on to the next generation and keep playing more rounds."
)

if __name__ == "__main__":
    runGenerations(numGenerations, numAgents, initial_endowment, selection_method)
