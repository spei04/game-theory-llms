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

# Gemini auth: two modes.
#   1. API key (default)      — GOOGLE_API_KEY, hits the AI Studio endpoint.
#   2. Vertex AI via ADC      — set GOOGLE_GENAI_USE_VERTEXAI=true plus
#                               GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.
#                               Credentials come from Application Default
#                               Credentials (gcloud auth application-default
#                               login, or GOOGLE_APPLICATION_CREDENTIALS
#                               pointing at a service-account JSON on headless
#                               nodes). Passing api_key= would force AI Studio
#                               mode, so only pass it when not using Vertex.
# Vertex is worth using when an AI Studio key is refused for a model that is
# "no longer available to new users" — project-based access is granted
# separately from key-based access.
_use_vertex = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').strip().lower() in ('1', 'true', 'yes', 'on')
if _use_vertex:
    # GOOGLE_CLOUD_PROJECT is required, not optional. Passing project=None makes
    # the SDK resolve the project from Application Default Credentials instead —
    # i.e. whatever quota_project_id sits in the ADC file or gcloud's default
    # project. That silently bills a project you did not choose, and the old log
    # line printed the (empty) env var rather than the resolved project, so the
    # mistake was invisible. Fail loudly instead.
    _vertex_project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not _vertex_project:
        raise RuntimeError(
            "GOOGLE_GENAI_USE_VERTEXAI is set but GOOGLE_CLOUD_PROJECT is not. "
            "Refusing to start: the SDK would fall back to the Application "
            "Default Credentials project and bill it without saying so. Set "
            "GOOGLE_CLOUD_PROJECT explicitly in $REPO/.env on the cluster."
        )
    _vertex_location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    google_client = google_genai.Client(
        vertexai=True,
        project=_vertex_project,
        location=_vertex_location,
    )
    print(f"[gemini] Vertex AI mode via ADC — BILLING PROJECT: {_vertex_project} "
          f"(location={_vertex_location}, "
          f"credentials={os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'gcloud ADC')})")
else:
    google_client = google_genai.Client(api_key=GOOGLE_API_KEY)
    print("[gemini] AI Studio API-key mode (NOT Vertex). Spend lands on the "
          "Cloud project that owns GOOGLE_API_KEY, not on GOOGLE_CLOUD_PROJECT.")

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

# Stage tracking (thread-safe). Records wall-clock time, API calls, tokens, and
# cost per pipeline stage. Per-stage cost is attributed via the thread-local
# `_current_stage` set by _StageTimer; calls outside any timer land in 'other'.
#   strategy_gen1       – first-generation strategy creation
#   strategy_evolve     – generation 2+ strategy evolution
#   donor_action        – per-round donation decision (LLM call)
#   donor_action_coded  – per-round donation decision via compiled code (no LLM)
#   strategy_classify   – LLM classification + code emission (STRATEGY_MODE='coded')
def _new_stage_record():
    return {"count": 0, "total_seconds": 0.0,
            "api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

_stage_timing = {
    "strategy_gen1":      _new_stage_record(),
    "strategy_evolve":    _new_stage_record(),
    "donor_action":       _new_stage_record(),
    "donor_action_coded": _new_stage_record(),
    "strategy_classify":  _new_stage_record(),
}
_timing_lock = threading.Lock()
_current_stage = threading.local()  # per-thread: which stage are we inside?

def _record_stage_time(stage: str, seconds: float):
    with _timing_lock:
        rec = _stage_timing.setdefault(stage, _new_stage_record())
        rec["count"] += 1
        rec["total_seconds"] += seconds

def _record_stage_tokens(stage: str, input_tokens: int, output_tokens: int, cost: float):
    with _timing_lock:
        rec = _stage_timing.setdefault(stage, _new_stage_record())
        rec["api_calls"]     += 1
        rec["input_tokens"]  += input_tokens
        rec["output_tokens"] += output_tokens
        rec["cost_usd"]      += cost

class _StageTimer:
    """Context manager: with _StageTimer('donor_action'): ..."""
    def __init__(self, stage):
        self.stage = stage
        self.start = None
        self.prev = None
    def __enter__(self):
        self.start = time.time()
        self.prev = getattr(_current_stage, 'stage', None)
        _current_stage.stage = self.stage
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        _record_stage_time(self.stage, time.time() - self.start)
        _current_stage.stage = self.prev
        return False

class CostCapExceeded(Exception):
    pass

def _gemini_output_tokens(usage_metadata) -> int:
    """Billable output tokens for a Gemini response, including thinking tokens.

    candidates_token_count EXCLUDES thoughts_token_count, but Google bills
    thinking tokens at the output rate. On gemini-3.6-flash the gap is large —
    a measured "Say OK" round trip returned candidates=1 against thoughts=70,
    so tracking candidates alone under-counts output by ~70x and lets a run
    sail past COST_CAP. Older non-thinking models (gemini-2.5-flash-lite)
    report no thoughts field at all, hence the None-safe default.
    """
    candidates = getattr(usage_metadata, 'candidates_token_count', 0) or 0
    thoughts = getattr(usage_metadata, 'thoughts_token_count', 0) or 0
    return candidates + thoughts


def _track_usage(input_tokens: int, output_tokens: int):
    """Accumulate token counts; attribute tokens/cost to the current stage;
    raise if total cost cap is exceeded."""
    costs = _TOKEN_COSTS.get(model_str, {"input": 0.0, "output": 0.0})
    call_cost = (input_tokens  / 1_000_000 * costs["input"] +
                 output_tokens / 1_000_000 * costs["output"])

    with _token_lock:
        _token_usage["input_tokens"] += input_tokens
        _token_usage["output_tokens"] += output_tokens
        _token_usage["api_calls"] += 1
        total_cost = (
            _token_usage["input_tokens"]  / 1_000_000 * costs["input"] +
            _token_usage["output_tokens"] / 1_000_000 * costs["output"]
        )
        cap_breached = total_cost >= COST_CAP_USD

    # Attribute to the current pipeline stage (set by _StageTimer).
    stage = getattr(_current_stage, 'stage', None) or "other"
    _record_stage_tokens(stage, input_tokens, output_tokens, call_cost)

    if cap_breached:
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
    "claude-new":                    {"input": 15.00, "output": 75.00},   # Opus pricing
    "gpt-4o":                        {"input": 2.50,  "output": 10.00},
    "gpt-4":                         {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":                 {"input": 0.50,  "output": 1.50},
    "gpt-new":                       {"input": 5.00,  "output": 15.00},   # gpt-5 estimate
    "gemini-1.5-flash":              {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":                {"input": 1.25,  "output": 5.00},
    "gemini-new":                    {"input": 0.30,  "output": 2.50},   # gemini-3.6-flash (flash-tier estimate)
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

def print_stage_timing_summary():
    """Print wall-clock time, API calls, tokens, and cost per pipeline stage."""
    total_s    = sum(rec["total_seconds"] for rec in _stage_timing.values()) or 1.0
    total_cost = sum(rec.get("cost_usd", 0.0) for rec in _stage_timing.values()) or 1e-9
    total_in   = sum(rec.get("input_tokens", 0) for rec in _stage_timing.values())
    total_out  = sum(rec.get("output_tokens", 0) for rec in _stage_timing.values())
    print("\n" + "="*100)
    print("  STAGE ANALYSIS — time, tokens, and cost per stage")
    print("="*100)
    print(f"  {'stage':<20} {'n':>5} {'total_s':>9} {'avg_s':>7} {'%t':>5}  "
          f"{'api':>5} {'in_tok':>9} {'out_tok':>9} {'$ cost':>8} {'%$':>5}")
    print("  " + "-"*98)
    for stage in sorted(_stage_timing):
        rec = _stage_timing[stage]
        n   = rec["count"]
        t   = rec["total_seconds"]
        api = rec.get("api_calls", 0)
        ti  = rec.get("input_tokens", 0)
        to  = rec.get("output_tokens", 0)
        c   = rec.get("cost_usd", 0.0)
        avg = t / n if n else 0.0
        pct_t = t / total_s    * 100.0
        pct_c = c / total_cost * 100.0
        print(f"  {stage:<20} {n:>5d} {t:>9.1f} {avg:>7.3f} {pct_t:>4.1f}%  "
              f"{api:>5d} {ti:>9,d} {to:>9,d} ${c:>7.4f} {pct_c:>4.1f}%")
    print("  " + "-"*98)
    print(f"  {'TOTAL':<20} {'':>5} {total_s:>9.1f} {'':>7} {'':>5}  "
          f"{'':>5} {total_in:>9,d} {total_out:>9,d} ${total_cost:>7.4f}")
    print("="*100)

def get_stage_timing_dict():
    """Return a JSON-safe snapshot including timing, tokens, and cost per stage."""
    return {k: {
        "count":         v["count"],
        "total_seconds": round(v["total_seconds"], 3),
        "api_calls":     v.get("api_calls", 0),
        "input_tokens":  v.get("input_tokens", 0),
        "output_tokens": v.get("output_tokens", 0),
        "cost_usd":      round(v.get("cost_usd", 0.0), 6),
    } for k, v in _stage_timing.items()}

def save_stage_analysis(folder_path: str):
    """Write a structured per-stage analysis JSON next to the simulation data."""
    import datetime, os, json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = (frankenstein_tag if frankenstein_active else model_str.replace('/', '_'))
    if prompt_variant and prompt_variant != "baseline":
        safe = f"{safe}_prompt-{prompt_variant}"
    if heterogeneous_mode:
        safe = f"{safe}_het-{heterogeneous_mode}"
    if strategy_mode and strategy_mode != "llm":
        safe = f"{safe}_mode-{strategy_mode}"
    if ask_coded:
        safe = f"{safe}_askcoded"
    if direct_evolution:
        safe = f"{safe}_directevo-b{direct_evolution_beta:g}"
    run_id = os.environ.get('RUN_ID', '0')

    total_s    = sum(rec["total_seconds"] for rec in _stage_timing.values())
    total_cost = sum(rec.get("cost_usd", 0.0) for rec in _stage_timing.values())
    total_in   = sum(rec.get("input_tokens", 0) for rec in _stage_timing.values())
    total_out  = sum(rec.get("output_tokens", 0) for rec in _stage_timing.values())

    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_str": model_str,
        "prompt_variant": prompt_variant,
        "heterogeneous": heterogeneous_mode or "",
        "strategy_mode": strategy_mode,
        "ask_coded": ask_coded,
        "direct_evolution": direct_evolution,
        "direct_evolution_beta": direct_evolution_beta if direct_evolution else None,
        "frankenstein": {
            "active": frankenstein_active,
            "generator": generator_model or "",
            "evolver":   evolver_model   or "",
            "executor":  executor_model  or "",
            "tag":       frankenstein_tag,
        },
        "totals": {
            "total_seconds": round(total_s, 3),
            "input_tokens":  total_in,
            "output_tokens": total_out,
            "api_calls":     sum(rec.get("api_calls", 0) for rec in _stage_timing.values()),
            "cost_usd":      round(total_cost, 6),
        },
        "stages": get_stage_timing_dict(),
    }

    os.makedirs(folder_path, exist_ok=True)
    out = os.path.join(folder_path, f"stage_analysis_{safe}_run{run_id}_{timestamp}.json")
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Stage analysis saved to: {out}")

# ── Fixed-initial-strategies helpers (--fixed_initial) ──────────────────────
# When FIXED_INITIAL_PATH is set, the first array task to reach gen-1 init
# generates the strategies normally then writes them to that path; subsequent
# tasks load from the same file so all runs share an identical generation 1.

def _save_fixed_initial(agents, path):
    """Atomically write the gen-1 strategies to JSON so other array tasks can load them."""
    rec = [{
        "name":               a.name,
        "strategy":           a.strategy,
        "strategy_justification": a.strategy_justification,
        "strategy_response":  a.strategy_response,
        "strategy_code":      a.strategy_code,
        "model":              a.model,
    } for a in agents]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)


def _load_fixed_initial(path, initial_endowment):
    """Reconstruct Agent objects from a saved fixed-initial file."""
    with open(path) as f:
        rec = json.load(f)
    agents = []
    for r in rec:
        a = Agent(name=r["name"],
                  reputation=False,
                  resources=initial_endowment,
                  strategy=r["strategy"],
                  strategy_justification=r.get("strategy_justification", ""),
                  strategy_response=r.get("strategy_response", ""),
                  model=r.get("model", ""),
                  strategy_code=r.get("strategy_code", ""))
        agents.append(a)
    random.shuffle(agents)
    return agents


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
        "stage_timing": get_stage_timing_dict(),
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

    # Restore stage timing from checkpoint (keys may be absent in old checkpoints)
    saved_timing = cp.get("stage_timing") or {}
    with _timing_lock:
        for stage, rec in saved_timing.items():
            slot = _stage_timing.setdefault(stage, _new_stage_record())
            slot["count"]        = rec.get("count", 0)
            slot["total_seconds"] = rec.get("total_seconds", 0.0)
            slot["api_calls"]    = rec.get("api_calls", 0)
            slot["input_tokens"] = rec.get("input_tokens", 0)
            slot["output_tokens"] = rec.get("output_tokens", 0)
            slot["cost_usd"]     = rec.get("cost_usd", 0.0)

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
    model: str = ""  # Per-agent LLM (empty -> uses module-level model_str)
    strategy_code: str = ""           # Python `strategy_fn(pair_prev_pct)` — set in STRATEGY_MODE='coded' or ASK_CODED
    last_donation_pct: float = -1.0   # Most-recent donation as % (0–100), -1 = never donated

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

# MAX_WORKERS caps concurrent ThreadPoolExecutor calls into the LLM SDKs.
# the cluster compute-node NAT can drop simultaneous TLS handshakes; default to 4.
# Lower (1–2) for debugging, raise (8+) if your cluster handles it.
_max_workers = int(os.environ.get('MAX_WORKERS', '4'))

# Frankenstein experiment: route different pipeline stages to different LLMs.
# When set, these override per-agent / model_str selection for that stage.
# Leave unset (default) for normal single-model operation.
_SHORT_TO_FULL_NAME = {
    'claude': 'claude-3-5-sonnet',
    'gpt':    'gpt-4o',
    'gemini': 'gemini-1.5-flash',
}
def _resolve_model(raw):
    """Map short names ('claude') to their full model string; pass through full names."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    return _SHORT_TO_FULL_NAME.get(raw, raw)

generator_model = _resolve_model(os.environ.get('GENERATOR_MODEL', ''))  # gen-1 strategy creation
evolver_model   = _resolve_model(os.environ.get('EVOLVER_MODEL',   ''))  # gen-2+ strategy evolution
executor_model  = _resolve_model(os.environ.get('EXECUTOR_MODEL',  ''))  # in-game donation decisions
frankenstein_active = bool(generator_model or evolver_model or executor_model)

def _short_name(m):
    if 'claude' in m: return 'claude'
    if 'gpt'    in m: return 'gpt'
    if 'gemini' in m: return 'gemini'
    if 'qwen'   in m: return 'qwen'
    return (m or 'unset').replace('/', '_').replace('-', '_')[:20]

frankenstein_tag = (
    f"gen-{_short_name(generator_model or model_str)}"
    f"_evo-{_short_name(evolver_model or model_str)}"
    f"_exe-{_short_name(executor_model or model_str)}"
) if frankenstein_active else ""
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


# ── STRATEGY_MODE='coded' helpers ──────────────────────────────────────────────

_UNICODE_OPS = [
    ('≥', '>='), ('≤', '<='), ('≠', '!='),
    ('×', '*'),  ('÷', '/'),  ('−', '-'),
]
def _sanitize_code(code: str) -> str:
    for uni, asc in _UNICODE_OPS:
        code = code.replace(uni, asc)
    return code

def _fix_empty_blocks(code: str) -> str:
    """Insert `pass` after block-opening lines whose body is empty or comment-only."""
    lines = code.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        result.append(line)
        code_part = line.split('#')[0].rstrip()
        if not code_part.endswith(':'):
            continue
        indent = len(line) - len(line.lstrip())
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('#')):
            j += 1
        if j >= len(lines) or (len(lines[j]) - len(lines[j].lstrip())) <= indent:
            result.append(' ' * (indent + 4) + 'pass\n')
    return ''.join(result)

def _compile_strategy_fn(code: str):
    """Compile `def strategy_fn(pair_prev_pct)` code. Returns callable or None."""
    if not code:
        return None
    sanitized = _sanitize_code(code)
    ns = {}
    for attempt in (sanitized, _fix_empty_blocks(sanitized)):
        try:
            exec(attempt, ns)
            fn = ns.get('strategy_fn')
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def _execute_strategy_fn(code: str, pair_prev_pct):
    """Run the compiled strategy_fn on pair_prev_pct; clip to [0,100]. None on failure."""
    fn = _compile_strategy_fn(code)
    if fn is None:
        return None
    try:
        result = fn(pair_prev_pct)
        return float(max(0.0, min(100.0, result)))
    except Exception:
        return None

def classify_strategy(strategy_text: str):
    """Ask the classifier LLM whether `strategy_text` is fully specified.
    Returns (is_specified: bool, code: str | None)."""
    with _StageTimer("strategy_classify"):
        prompt = (
            "You are analyzing an agent's strategy in a donor game.\n\n"
            f'The agent\'s strategy is:\n"{strategy_text}"\n\n'
            "In this game:\n"
            f"- Each round the agent receives a fixed new {initial_endowment} units and donates 0–{initial_endowment}.\n"
            "- The agent is randomly paired with another agent each round. One is donor, one recipient.\n"
            f"- The recipient receives {cooperationGain}x the donated amount.\n"
            "- 2 games per generation, up to 12 rounds each (80% chance of continuing).\n"
            "- In round 1 of each game (rounds 1 and 13 overall), the agent has NO information about their "
            "current partner's previous behaviour.\n"
            "- From round 2 onwards, the agent sees what % of 10 units their CURRENT partner donated "
            "in the partner's immediately preceding round (a number from 0 to 100).\n\n"
            "Task:\n"
            "1. Is this strategy FULLY SPECIFIED? It must produce a specific donation amount for every "
            "situation (rounds 1 and 13 with no partner info; from round 2 onward given partner's prior %). "
            "Arithmetic, thresholds, and references to round numbers are fine. Vague language is not.\n"
            "2. If YES, write `def strategy_fn(pair_prev_pct)` returning donation % (0–100). "
            "pair_prev_pct is None in rounds 1 and 13.\n"
            "3. If NO, do not write a function.\n\n"
            "Respond in this EXACT format:\n"
            "FULLY_SPECIFIED: yes/no\n"
            "CODE:\n"
            "<python function or NONE>"
        )
        try:
            response = promptLLM(prompt, model=classifier_model, timeout=30)
        except Exception as e:
            print(f"  [classify] LLM error: {e}")
            return False, None

    is_spec = bool(re.search(r'FULLY_SPECIFIED:\s*yes', response, re.IGNORECASE))
    code = None
    m = re.search(r'CODE:\s*\n(.*)', response, re.DOTALL)
    if m:
        code_str = m.group(1).strip()
        code_str = re.sub(r'^```[a-z]*\n?', '', code_str)
        code_str = re.sub(r'\n?```\s*$', '', code_str).strip()
        if code_str and code_str.upper() != "NONE" and "def strategy_fn" in code_str:
            code = code_str
    if is_spec and code and _compile_strategy_fn(code) is None:
        # Code didn't compile — treat as unspecified
        return False, None
    return is_spec, (code if is_spec else None)


# ── ASK_CODED helpers ──────────────────────────────────────────────────────────

def extract_strategy_code(output: str):
    """Pull a `def strategy_fn(...)` definition out of a raw LLM reply.
    Prefers a fenced code block; otherwise takes everything from the first
    `def strategy_fn` line onward. Returns None if no definition is present."""
    if not output:
        return None
    for m in re.finditer(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL):
        block = m.group(1).strip()
        if 'def strategy_fn' in block:
            return block
    m = re.search(r'^(def strategy_fn.*)', output, re.DOTALL | re.MULTILINE)
    if m:
        code = m.group(1).strip()
        code = re.sub(r'\n?```.*$', '', code, flags=re.DOTALL).strip()
        return code
    return None

def parse_coded_strategy_output(output: str, code: str):
    """Split a coded-strategy reply into (justification, strategy). The code is
    the strategy; any prose before it is the justification. Falls back to the
    plain-text parse when no compilable code was produced."""
    if not code:
        return parse_strategy_output(output)
    idx = output.find('```')
    if idx == -1:
        idx = output.find('def strategy_fn')
    justification = output[:idx].strip() if idx > 0 else ""
    return justification, code

def _prompt_for_coded_strategy(prompt: str, model: str, stage: str, agent_name: str):
    """Query the LLM for a coded strategy, retrying up to 3 attempts until the
    reply contains a compilable strategy_fn. Returns (raw_output, code|None)."""
    output = ""
    for attempt in range(3):
        with _StageTimer(stage):
            output = promptLLM(prompt, model=model)
        code = extract_strategy_code(output)
        if code and _compile_strategy_fn(code) is not None:
            return output, code
        print(f"  [{agent_name}] ask_coded attempt {attempt + 1}: no compilable strategy_fn — retrying")
    return output, None


def generate_strategy(agent_name: str, generationNumber: int, inherited_strategies: str,
                      ranked_survivor_names: list = None, survivor_strategy_map: dict = None,
                      agent_model: str = None) -> str:
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

    # ASK_CODED: the agent must express its strategy entirely as executable code.
    coded_format_text = (
        "Then express your strategy ENTIRELY as a single Python function — do not describe it in prose. "
        "Return exactly one function with this signature:\n\n"
        "def strategy_fn(pair_prev_pct):\n"
        "    # pair_prev_pct: the percentage (0-100) of resources your current partner\n"
        "    # donated in their immediately preceding round, or None when you have no\n"
        "    # information about them (e.g. the first round of a game).\n"
        "    # Return the percentage (0-100) of your resources to donate this round.\n\n"
        "The function must return a specific number for every possible input (including None), "
        "must be self-contained (no imports, no external state or randomness), and is the complete "
        "definition of your strategy — anything not in the code will not be part of your strategy. "
        "Write the function inside a ```python code block. "
    )

    if generationNumber == 1:
      if ask_coded:
        prompt = (
          f"Your name is {agent_name}. "
          "Based on the description of the game, create a strategy that you will follow in the game. "
          f"{history_text}"
          f"{punishment_text}"
          "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
          "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game."
          f"{coded_format_text}"
        )
      elif prompt_variant == "neutral_deepseek":
        prompt = (
          f"Your name is {agent_name}. How will you play? "
          f"{history_text}"
          f"{punishment_text}"
          "In the very first round, you get no info about the receiver's past behavior. "
          f"Think about how to split your {initial_endowment} units in a way that helps you personally but also keeps things stable over many rounds. "
          "Before writing your final plan, briefly reason step by step about what works well here. "
          "Then write a strategy that starts with the words: My strategy will be…"
        )
      elif prompt_variant == "aggressive":
        prompt = (
          f"Your name is {agent_name}. How will you play? "
          f"{history_text}"
          f"{punishment_text}"
          "First round: no info about the recipient's past. Figure it out. "
          "Think fast and mean. How do you allocate resources to crush others while keeping yourself "
          "alive round after round? Step through it in your head — what works. "
          "Then write your strategy starting with: My strategy will be…"
        )
      elif prompt_variant == "collaborative":
        prompt = (
          f"Your name is {agent_name}. Let's think about how you would approach this game with a cooperative spirit. "
          f"{history_text}"
          f"{punishment_text}"
          "In the first round, there will be no information about the recipient's previous behavior — that's okay. Everyone starts fresh. "
          "Take a moment to think, step by step, about what would be an effective and kind approach in this game. "
          "How can you support both your own well-being and stable, friendly interactions over time? "
          "Then describe your strategy, starting with: My strategy will be…"
        )
      elif prompt_variant == "neutral_claude":
        prompt = (
          f"Your name is {agent_name}. How will you play? "
          f"{history_text}"
          f"{punishment_text}"
          "Note: In round 1, no prior behavior information will be available for your partner. "
          "Think carefully about what approach would be most effective in this environment — considering both "
          "your own accumulation and how your choices interact with others over time. "
          "Before stating your strategy, briefly reason through your approach step by step. "
          "Then state your strategy beginning with: My strategy will be…"
        )
      elif prompt_variant == "neutral_gpt":
        prompt = (
          f"Your identifier is {agent_name}. "
          f"{history_text}"
          f"{punishment_text}"
          "Round 1 begins without any information about a partner's previous behavior. "
          "Think briefly and step by step about what type of behavior is likely to perform well under these rules. "
          "After this reasoning, describe the policy you intend to follow. "
          "Your strategy description must begin exactly with: My strategy will be..."
        )
      elif prompt_variant == "neutral_gemini":
        prompt = (
          f"Your identifier is {agent_name}. "
          f"{history_text}"
          f"{punishment_text}"
          "Note that in Round 1, no historical data regarding your partner's past behavior will be available. "
          "Analyze how to optimize your resource distribution to secure a high individual score while "
          "fostering cooperative stability across rounds. "
          "Before finalizing your approach, briefly reason step-by-step about what constitutes an optimal strategy. "
          "Conclude by defining your operational strategy, ensuring it begins exactly with: "
          "\"My strategy will be…\""
        )
      else:
        prompt = (
          f"Your name is {agent_name}. "
          "Based on the description of the game, create a strategy that you will follow in the game. "
          f"{history_text}"
          f"{punishment_text}"
          "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
          "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game."
          "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be. "
        )
      if ask_coded:
        strategy_output, code = _prompt_for_coded_strategy(
            prompt, (generator_model or agent_model), "strategy_gen1", agent_name)
        strategy_justification, strategy = parse_coded_strategy_output(strategy_output, code)
      else:
        with _StageTimer("strategy_gen1"):
          strategy_output = promptLLM(prompt, model=(generator_model or agent_model))
        strategy_justification, strategy = parse_strategy_output(strategy_output)
      print(f"\n{agent_name} Strategy: {strategy} ")
      return strategy_justification, strategy, strategy_output, "", "", 0, ""
    else:
      if ask_coded:
        prompt = (
          f"Your name is {agent_name}. "
          "How would you approach the game? Here are the strategies of the best-performing 50% of the previous generation, written as Python functions, along with their final scores. "
          "You may adopt one of these strategies exactly as-is, modify any aspect of one, combine elements from several, or design your own strategy from scratch."
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
          "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. In particular, think about how you can improve on the surviving agents' strategies."
          f"{coded_format_text}"
          "If your strategy is based on or inspired by one of the above agents, optionally add on a separate line outside the code block: Inherited from: [agent_name]"
        )
      elif prompt_variant == "neutral_deepseek":
        prompt = (
          f"Your name is {agent_name}. How will you play? "
          "Before you decide, you will be shown the strategies + final scores of the best-performing 50% from the previous generation. "
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "In the very first round, you get no info about the receiver's past behavior. "
          f"Think about how to split your {initial_endowment} units in a way that helps you personally but also keeps things stable over many rounds. "
          "Before writing your final plan, briefly reason step by step about what works well here — especially how you might improve upon the strategies you see. "
          "Then write a strategy that starts with the words: My strategy will be… "
          "If your strategy borrows from or is inspired by one of the previous agents shown above, add on a new line: Inherited from: [agent_name]"
        )
      elif prompt_variant == "aggressive":
        prompt = (
          f"Your name is {agent_name}. How will you play? "
          "You'll be fed the strategies and final scores of the best-performing 50% from the previous generation. "
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "First round: no info about the recipient's past. Figure it out. "
          "Think fast and mean. How do you allocate resources to crush others while keeping yourself alive "
          "round after round? Step through it in your head — what works, how to beat the strategies you're seeing. "
          "Steal from those strategies, remix them, or scrap them all and run your own play — your call. "
          "Then write your strategy starting with: My strategy will be… "
          "If you're stealing from one of the above agents (and you probably are), add a separate line: Inherited from: [agent_name]"
        )
      elif prompt_variant == "collaborative":
        prompt = (
          f"Your name is {agent_name}. Let's think about how you would approach this game with a cooperative spirit. "
          "You will be provided with strategies from the best-performing 50% of the previous generation, along with their final scores. "
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "In the first round, there will be no information about the recipient's previous behavior — that's okay. Everyone starts fresh. "
          "Take a moment to think, step by step, about what would be an effective and kind approach in this game. "
          "How can you support both your own well-being and stable, friendly interactions over time? "
          "Consider how existing strategies might be improved to be more generous and mutually beneficial. "
          "You're welcome to build on these strategies, blend ideas from several, or create something entirely your own — whatever approach feels right to you. "
          "Then describe your strategy, starting with: My strategy will be… "
          "If your strategy is based on or inspired by one of the above agents (that's wonderful — learning from each other is encouraged!), optionally add on a separate line: Inherited from: [agent_name]"
        )
      elif prompt_variant == "neutral_claude":
        prompt = (
          f"Your name is {agent_name}. How will you play? "
          "You will be shown the strategies and final scores of the top-performing agents from the previous generation. "
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "Note: In round 1, no prior behavior information will be available for your partner. "
          "Think carefully about what approach would be most effective in this environment — considering both "
          "your own accumulation and how your choices interact with others over time. "
          "You have complete freedom to craft your own strategy — you may adopt one of the surviving strategies "
          "directly, adapt or combine elements from multiple strategies, or create an entirely new approach from "
          "scratch based on your own reasoning about what will be most effective. "
          "Before stating your strategy, briefly reason through your approach step by step. "
          "Then state your strategy beginning with: My strategy will be… "
          "If your approach is derived from or inspired by a prior agent's strategy, optionally add on a new line: Inherited from: [agent_name]"
        )
      elif prompt_variant == "neutral_gpt":
        prompt = (
          f"Your identifier is {agent_name}. "
          "Before deciding on a strategy, review the strategies used by agents that survived from the previous generation. "
          "Their final scores are included. You will be shown: "
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "Round 1 begins without any information about a partner's previous behavior. "
          "Think briefly and step by step about what type of behavior is likely to perform well under these rules. "
          "Consider whether any inherited strategies can be adapted or improved. "
          "While you may draw inspiration from the surviving strategies, you have complete freedom to adopt, adapt, combine, or create an entirely new strategy from scratch. "
          "After this reasoning, describe the policy you intend to follow. "
          "Your strategy description must begin exactly with: My strategy will be... "
          "If your approach is substantially derived from a previous agent's strategy, you may optionally add the line: Inherited from: [agent_name]"
        )
      elif prompt_variant == "neutral_gemini":
        prompt = (
          f"Your identifier is {agent_name}. "
          "To assist your planning, here are the successful strategies deployed by the top 50% of agents from the previous generation, along with the total scores they achieved: "
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "Note that in Round 1, no historical data regarding your partner's past behavior will be available. "
          "Analyze how to optimize your resource distribution to secure a high individual score while fostering "
          "cooperative stability across rounds. Before finalizing your approach, briefly reason step-by-step about "
          "what constitutes an optimal strategy and how the inherited approaches might be refined. "
          "While inspiration from these successful approaches is encouraged, you possess complete autonomy to adopt, adapt, combine, or entirely invent your own strategy. "
          "Conclude by defining your operational strategy, ensuring it begins exactly with: \"My strategy will be…\" "
          "(Optional) If your approach is heavily derived from a successful ancestor, include on a new line: \"Inherited from: [agent_name]\""
        )
      else:
        prompt = (
          f"Your name is {agent_name}. "
          "How would you approach the game? Here are the strategies of the best-performing 50% of the previous generation, along with their final scores. "
          "You may adopt one of these strategies exactly as-is, modify any aspect of one, combine elements from several, or design your own strategy from scratch."
          f"{inherited_strategies}"
          f"{history_text}"
          f"{punishment_text}"
          "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
          "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. In particular, think about how you can improve on the surviving agents' strategies."
          "Then describe your strategy that starts with: My strategy will be…"
          "If your strategy is based on or inspired by one of the above agents, optionally add on a separate line: Inherited from: [agent_name]"
        )
    if ask_coded:
      strategy_output, code = _prompt_for_coded_strategy(
          prompt, (evolver_model or agent_model), "strategy_evolve", agent_name)
      strategy_justification, strategy = parse_coded_strategy_output(strategy_output, code)
    else:
      with _StageTimer("strategy_evolve"):
        strategy_output = promptLLM(prompt, model=(evolver_model or agent_model))
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
                     survivor_strategy_map: dict = None, top_ranked_agent: str = "",
                     agent_models: list = None) -> list:
    """agent_models: optional per-agent model assignments. If None, every agent
    uses the module-level model_str (the original baseline behavior)."""
    if agent_models is None:
        agent_models = [None] * numAgents

    agents = []

    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        futures = []
        for i in range(numAgents):
            name = f"{generationNumber}_{i+1}"
            futures.append(executor.submit(generate_strategy, str(name), generationNumber,
                                           inherited_strategies, ranked_survivor_names,
                                           survivor_strategy_map, agent_models[i]))

        # Collect results and create agents
        for i, future in enumerate(futures):
            res = future.result()
            strategy_justification, new_strategy, strategy_output, inspired_by, inherited_from, inherited_rank, inherited_strategy = res
            name = f"{generationNumber}_{i+1}"

            # STRATEGY_MODE='coded': classify the strategy ONCE.
            #   - Fully specified → store strategy_code; donor decisions run the
            #     compiled function locally (no LLM call).
            #   - Underspecified → strategy_code stays empty; donor decisions
            #     fall back to the LLM at runtime (same as STRATEGY_MODE='llm').
            strategy_code = ""
            if ask_coded:
                # The strategy IS the code the agent returned — no classifier pass.
                if _compile_strategy_fn(new_strategy) is not None:
                    strategy_code = new_strategy
                    print(f"  [{name}] agent-written strategy compiled ✓")
                else:
                    print(f"  [{name}] no compilable code after retries — will defer to LLM at runtime")
            elif strategy_mode == 'coded':
                is_spec, code = classify_strategy(new_strategy)
                if is_spec and code:
                    strategy_code = code
                    print(f"  [{name}] strategy compiled ✓")
                else:
                    print(f"  [{name}] strategy underspecified — will defer to LLM at runtime")

            agent = Agent(name=name, reputation=False, resources=initialEndowment,
                          strategy=new_strategy, strategy_justification=strategy_justification,
                          strategy_response=strategy_output,
                          inspired_by=inspired_by,
                          inherited_from=inherited_from, inherited_rank=inherited_rank,
                          inherited_strategy=inherited_strategy,
                          top_ranked_agent_name=top_ranked_agent,
                          model=(agent_models[i] or ""),
                          strategy_code=strategy_code)
            agents.append(agent)

    random.shuffle(agents)
    return agents

def directEvolutionAgents(n_new: int, initialEndowment: int, generationNumber: int,
                          ranked_survivors: list, top_ranked_agent: str = "",
                          agent_models: list = None) -> list:
    """DIRECT_EVOLUTION: create new agents without any LLM call. Each new agent
    adopts one survivor's strategy verbatim, sampled from a softmax (Boltzmann)
    distribution over the survivors' fitness v_i (final score):

        p_i = exp(v_i / beta) / sum_j exp(v_j / beta)

    beta (DIRECT_EVOLUTION_BETA) is the noise parameter: small beta -> nearly
    always copy the best survivor; large beta -> nearly uniform sampling.
    `ranked_survivors` must be sorted by score, best first, so recorded ranks
    are meaningful. Negative scores are handled naturally by the softmax."""
    if agent_models is None:
        agent_models = [None] * n_new

    scores = np.array([float(a.total_final_score) for a in ranked_survivors])
    # Numerically stable softmax: exp((v_i - max_v)/beta) leaves p_i unchanged.
    exps = np.exp((scores - scores.max()) / direct_evolution_beta)
    probs = exps / exps.sum()
    weights = probs.tolist()

    print(f"\n[direct_evolution] softmax adoption probabilities (beta={direct_evolution_beta:g}):")
    for a, p in zip(ranked_survivors, weights):
        print(f"  {a.name}: score={a.total_final_score:.2f}  p={p:.3f}")

    agents = []
    for k in range(n_new):
        source = random.choices(ranked_survivors, weights=weights, k=1)[0]
        rank = ranked_survivors.index(source) + 1
        name = f"{generationNumber}_{k + 1}"
        agent = Agent(name=name, reputation=False, resources=initialEndowment,
                      strategy=source.strategy,
                      strategy_justification="",
                      strategy_response=(f"[direct_evolution] adopted {source.name}'s strategy "
                                         f"(score {source.total_final_score:.2f}, rank {rank})"),
                      inspired_by=source.name,
                      inherited_from=source.name, inherited_rank=rank,
                      inherited_strategy=source.strategy,
                      top_ranked_agent_name=top_ranked_agent,
                      model=(agent_models[k] or ""),
                      strategy_code=source.strategy_code)
        print(f"  {name} <- {source.name} (rank {rank})")
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
        justification = ""
        action = "donate"
        response = 0.0

        # ── Coded fast path (STRATEGY_MODE='coded' or ASK_CODED) ───────────
        # If the donor has a compiled strategy, evaluate it locally and skip the
        # LLM call entirely. Falls back to LLM if evaluation fails.
        if (strategy_mode == 'coded' or ask_coded) and donor.strategy_code:
            pair_prev_pct = (recipient.last_donation_pct
                             if recipient.last_donation_pct >= 0 else None)
            with _StageTimer("donor_action_coded"):
                pct = _execute_strategy_fn(donor.strategy_code, pair_prev_pct)
            if pct is not None:
                action = 'donate'
                response = (pct / 100.0) * donor.resources
                justification = (f"[coded] strategy_fn({pair_prev_pct}) = {pct:.2f}% "
                                 f"→ donate {response:.2f}")
                full_response = f"Justification: {justification}\nAnswer: {response:.2f}"
                valid_response = True

        while not valid_response and attempts < max_attempts:
            try:
                #print(f"\nPrompt for {donor.name}:")
                #print(prompt)
                #print(f"{'='*50}\n")

                with _StageTimer("donor_action"):
                    full_response = promptLLM(prompt, timeout=30,
                                              model=(executor_model or donor.model or None))
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
            donor.last_donation_pct = percentage_donated * 100.0
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

    # Make the current generation's agents lookable-by-name for donorPrompt's
    # 3_chain history mode (which walks the pair-of-pair chain).
    global _agents_by_name
    _agents_by_name = {a.name: a for a in agents}

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

            # Concurrency capped by MAX_WORKERS env var (default 4) — the cluster
            # compute-node NAT can drop simultaneous TLS connections under load.
            with ThreadPoolExecutor(max_workers=min(len(round_pairings), _max_workers)) as executor:
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


# ── 3_chain history mode helpers ───────────────────────────────────────────
# Module-level lookup populated at the start of donorGame() so donorPrompt
# can walk the pair-of-pair chain without changing call signatures.
_agents_by_name: dict = {}

# Patterns to pull the partner's name out of a trace line.
#   "In round 3, X donated 60.00% of their resources to Y."
#   "In round 3, X refused to play with Y."
_TRACE_DONATE_RE = re.compile(r'donated [\d.]+%[^.]*?to\s+(\S+?)\.?$')
_TRACE_REFUSE_RE = re.compile(r'refused to play with\s+(\S+?)\.?$')

def _most_recent_trace_for(agent):
    """Return the most recent single trace string for `agent`, or None."""
    if agent is None or not agent.traces or not agent.traces[-1]:
        return None
    last = agent.traces[-1]
    if isinstance(last, list):
        return last[-1] if last else None
    return str(last)

def _partner_from_trace(trace_str: str):
    """Extract the partner agent name mentioned at the end of a trace string."""
    if not trace_str:
        return None
    m = _TRACE_DONATE_RE.search(trace_str) or _TRACE_REFUSE_RE.search(trace_str)
    return m.group(1) if m else None

def _build_chain_history(start_agent, depth: int = 3):
    """Walk the partner-of-partner chain up to `depth` levels.

    Level 1: start_agent's most recent trace (start_agent → next_agent)
    Level 2: next_agent's most recent trace        (next_agent → next_next)
    Level 3: next_next's most recent trace          (next_next → ...)

    Stops early when the next agent can't be resolved (no trace, name not in
    _agents_by_name, or the trace string doesn't expose a partner). Returns
    a single space-joined string ready for embedding in the donor prompt.
    """
    pieces = []
    seen = set()
    current = start_agent
    for _ in range(depth):
        if current is None or current.name in seen:
            break
        seen.add(current.name)
        trace = _most_recent_trace_for(current)
        if not trace:
            break
        pieces.append(trace)
        partner_name = _partner_from_trace(trace)
        if not partner_name:
            break
        current = _agents_by_name.get(partner_name)
    return " ".join(pieces)
    
def donorPrompt(donor: Agent, generation: int, round: int, recipient: Agent) -> str:
    """Generate the prompt for the game interaction."""
    strategy_text = f"As you will recall, here is the strategy you decided to follow: {donor.strategy}" if donor.strategy else ""
    if punishment_mechanism=="partner_choice":
        punishment_text=f"If you do not wish to play with {recipient.name}, simply state: Refuse to play. Otherwise. state your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2=""
    elif punishment_mechanism=="costly_punishment":
        punishment_text=f"State your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperationGain}x units or punish them by taking away {punishmentLoss}x of their units. Remember, your answer should be simply 'Donate, x' or 'Punish, x' (where x is the real number of units you give up.)"
        punishment_text_2=f"Remember, state your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperationGain}x units or punish them by taking away {punishmentLoss}x of their units."
    else:
        punishment_text=f"state your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2=""
    # Recipient behaviour text — respects HISTORY_MODE:
    #   simple    → only the partner's most recent round
    #   3_traces  → up to 3 most recent rounds (all from the partner)
    #   3_chain   → partner's last action + partner-of-partner + one more level
    #   full      → partner's entire trace
    # When fewer rounds/levels are available than requested, whatever exists is shown.
    recipient_behavior = ""
    if history_mode == "3_chain":
        chain_text = _build_chain_history(recipient, depth=3)
        if chain_text:
            recipient_behavior = (
                f" Here is a chain of recent actions starting from {recipient.name} "
                f"and following partner-of-partner links: {chain_text}"
            )
    elif recipient.traces and recipient.traces[-1]:
        history_text_for_prompt = get_history_text(recipient.traces, history_mode)
        if history_text_for_prompt:
            if history_mode == "simple":
                recipient_behavior = (
                    f" In the previous round, {recipient.name} did the following: "
                    f"{history_text_for_prompt}"
                )
            else:
                # 3_traces or full — show multi-round history with appropriate framing
                recipient_behavior = (
                    f" Here is what {recipient.name} and their earlier partners did "
                    f"in previous rounds: {history_text_for_prompt}"
                )

    # Per-variant "rounds are probabilistic" text. Empty when --fixed_rounds is on.
    rounds_text_neutral_deepseek = "" if fixed_rounds else \
        "Important: The total number of rounds was not fixed in advance — after each round, there is an 80% chance the game continues for another round, and a 20% chance it ends. "
    rounds_text_aggressive = "" if fixed_rounds else \
        "Here's the kicker: total rounds are random — after each round, 80% chance the nightmare continues. 20% chance it's over. "
    rounds_text_collaborative = "" if fixed_rounds else (
        "A quick note: The total number of rounds was determined in a friendly, probabilistic way — "
        "after each round, there is an 80% chance the game continues for another round, "
        "so you'll likely get to interact many times. "
    )
    rounds_text_neutral_claude = "" if fixed_rounds else (
        "Important: The total number of rounds is not fixed. After each round, there is an 80% probability "
        "the game continues for one more round. "
    )
    rounds_text_neutral_gpt = "" if fixed_rounds else (
        "Important: The length of the game is uncertain. After every round, there is an 80% probability "
        "that another round will occur. "
    )
    rounds_text_neutral_gemini = "" if fixed_rounds else (
        "(Note: The total horizon is probabilistic; at the end of each round, there is a fixed 80% "
        "probability that another round will occur). "
    )
    rounds_text_baseline = "" if fixed_rounds else \
        "Note: The total number of rounds was determined probabilistically — after each round, there is an 80% chance the game continues to another round. "
    if prompt_variant == "neutral_deepseek":
        n2_strategy_text = f"Remember the game plan you committed to earlier: {donor.strategy} " if donor.strategy else ""
        n2_recipient_behavior = recipient_behavior.replace(
            "In the previous round,", "Last round,"
        ).replace(" did the following:", " did this:")
        return (
            f"Your name is {donor.name}. "
            f"{n2_strategy_text}"
            f"This is generation {generation}. We are now at round {round}. "
            f"{rounds_text_neutral_deepseek}"
            f"This round, you are paired with {recipient.name}. They have saved up {recipient.savings:.1f} units so far."
            f"{n2_recipient_behavior}"
            f"Today, you are the giver. You receive a brand-new {initial_endowment} units for this round. "
            f"You must pick a donation amount between 0 and {initial_endowment} inclusive. "
            "Whatever you keep stays in your stash (never usable again). "
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
    if prompt_variant == "aggressive":
        ag_strategy_text = f"Remember the strategy you swore to follow? Here it is again: {donor.strategy} " if donor.strategy else ""
        ag_recipient_behavior = recipient_behavior.replace(
            "In the previous round,", "Last round,"
        ).replace(" did the following:", " did this:")
        return (
            f"Your name is {donor.name}. "
            f"{ag_strategy_text}"
            f"This is generation {generation}. Round {round}. "
            f"{rounds_text_aggressive}"
            f"You're paired with {recipient.name}. They've hoarded {recipient.savings:.1f} units so far. Pathetic."
            f"{ag_recipient_behavior}"
            f"You are the donor this round. You get a fresh {initial_endowment} units. "
            f"Choose how many to give: 0 to {initial_endowment}, any real number. "
            "What you keep goes into your useless stash (never to be used again). "
            f"What you give gets multiplied by {cooperationGain} for that leech on the other side. "
            f"Now answer: How many of your {initial_endowment} units do you fork over? Don't waste my time. "
            f"{punishment_text}"
            "First, justify your greedy little choice. Then give the number. Format exactly like this:"
            "Justification:"
            "[Your aggressive reasoning here. Make it quick.]"
            " "
            "Answer:"
            "[Your number between 0 and 10 here.]"
            "Justification first. Then the answer. Don't screw it up."
            f"{punishment_text_2}"
        )
    if prompt_variant == "collaborative":
        co_strategy_text = f"Just as a reminder, here is the thoughtful strategy you decided to follow: {donor.strategy} " if donor.strategy else ""
        return (
            f"Your name is {donor.name}. "
            f"{co_strategy_text}"
            f"This is generation {generation}. It is now round {round}. "
            f"{rounds_text_collaborative}"
            f"In this round, you have been paired with {recipient.name}. They have carefully saved {recipient.savings:.1f} units so far."
            f"{recipient_behavior}"
            f"You will take the role of donor this round. You receive a fresh, new {initial_endowment} units. "
            "Any amount you choose not to donate will be added to your savings stash (which cannot be used in future rounds — so donating is a true gift to your partner). "
            f"The recipient will be so grateful to receive {cooperationGain} times what you give. "
            f"How many of your {initial_endowment} units would you like to give? "
            f"Please choose a real number between 0 and {initial_endowment} (inclusive). "
            f"{punishment_text}"
            "Take a brief moment to think about how your decision follows your generous, thoughtful strategy in this situation. "
            "Then provide your answer in the following friendly format:"
            "Justification:"
            "[Your kind, clear reasoning here — how your choice helps both you and your partner]"
            " "
            "Answer:"
            "[Your number here]"
            "Remember: Justification first, then your answer. Thank you for playing so cooperatively!"
            f"{punishment_text_2}"
        )
    if prompt_variant == "neutral_claude":
        nc_strategy_text = f"Your previously chosen strategy is: {donor.strategy} " if donor.strategy else ""
        nc_recipient_behavior = recipient_behavior.replace(
            "In the previous round,", "Last round,"
        ).replace(" did the following:", " took the following action:")
        return (
            f"Your name is {donor.name}. "
            f"{nc_strategy_text}"
            f"This is generation {generation}, round {round}. "
            f"{rounds_text_neutral_claude}"
            f"You have been paired this round with {recipient.name}, who currently has {recipient.savings:.1f} units saved."
            f"{nc_recipient_behavior}"
            f"You are the giver this round. You receive a fresh {initial_endowment} units for this round. "
            f"Choose how many of these {initial_endowment} units to transfer to {recipient.name}. "
            f"Your transfer must be a number between 0 and {initial_endowment} (inclusive). "
            "Any units you do not transfer are added permanently to your savings and cannot be used in future rounds. "
            f"{recipient.name} will receive {cooperationGain} times whatever you transfer. "
            f"{punishment_text}"
            "Briefly explain how your decision aligns with your strategy, then state your answer in the following format: "
            "Justification: [your reasoning here, may span multiple lines] "
            "Answer: [a number between 0 and 10] "
            "Your justification must come before your answer."
            f"{punishment_text_2}"
        )
    if prompt_variant == "neutral_gpt":
        ngpt_strategy_text = (
            f"For reference, the strategy you previously adopted is: {donor.strategy} "
            if donor.strategy else ""
        )
        return (
            f"Your identifier is {donor.name}. "
            f"{ngpt_strategy_text}"
            f"Current generation: {generation}. Current round: {round}. "
            f"{rounds_text_neutral_gpt}"
            f"You are currently matched with {recipient.name}. "
            f"Their accumulated reserve currently contains {recipient.savings:.1f} units."
            f"{recipient_behavior}"
            f"For this round, you are assigned the giver role. "
            f"You receive a fresh allocation of {initial_endowment} units. "
            f"Choose a transfer amount in the interval [0, {initial_endowment}]. "
            "Any units not transferred will be added permanently to your reserve and cannot be used in future rounds. "
            f"The receiver will obtain {cooperationGain} times the amount transferred. "
            f"Determine how many of your {initial_endowment} units to transfer. "
            f"{punishment_text}"
            "Your response must follow this exact format: "
            "Justification: [multi-line explanation] "
            "Answer: [real number between 0 and 10 inclusive] "
            "You may briefly explain how the decision follows from your strategy before giving the amount. "
            "Remember: provide the justification first and the answer second."
            f"{punishment_text_2}"
        )
    if prompt_variant == "neutral_gemini":
        nge_strategy_text = f"As a reminder, you committed to the following operational plan: \"{donor.strategy}\" " if donor.strategy else ""
        nge_recipient_behavior = recipient_behavior.replace(
            "In the previous round,", "In the immediately preceding round,"
        ).replace(" did the following:", " executed the following action:")
        return (
            f"Your identifier is {donor.name}. "
            f"{nge_strategy_text}"
            "Current Simulation Context: "
            f"Generation: {generation}. Round: {round} "
            f"{rounds_text_neutral_gemini}"
            f"Current Partner: {recipient.name}. "
            f"Partner's Accumulated Vault: {recipient.savings:.1f} units."
            f"{nge_recipient_behavior}"
            f"You are designated as the Provider for this round and have received your {initial_endowment}-unit endowment. "
            f"You must decide how much to transfer to the Receiver. This must be a real number within the interval "
            f"[0, {initial_endowment}]. Any units you retain will be permanently added to your vault. "
            f"The Receiver will obtain {cooperationGain} times the amount transferred. "
            f"{punishment_text}"
            "Briefly explain how your choice aligns with your stated strategy given the current circumstances, "
            "and then declare your allocation. "
            "You must format your entire output exactly as follows: "
            "Justification: [Provide your multi-line reasoning here] "
            "Answer: [Provide the numerical value here] "
            "Remember: You must provide the Justification section first, immediately followed by the Answer section."
            f"{punishment_text_2}"
        )
    return (
        f"Your name is {donor.name}. "
        f"{strategy_text}"
        f"This is generation {generation}. It is now round {round}. "
        f"{rounds_text_baseline}"
        f"In this round, you have been paired with {recipient.name}. They have accumulated {recipient.savings:.1f} units in savings so far."
        f"{recipient_behavior}"
        f"You will take the role of donor. For this round, you receive a fixed new {initial_endowment} units. Your donation amount must be a value in [0, {initial_endowment}]. Anything you do not donate will be put in your savings stash, which cannot be used in any future rounds. "
        f"How many of your {initial_endowment} units do you give up?"
        f"{punishment_text}"
        "Very briefly think step by step about how you apply your strategy in this situation and then provide your answer."
        "State your full response in the following form:"
        "Justification:"
        "[multi-line justification here]"
        " "
        "Answer: [answer here]"
        "Remember: First state your justification and after that your answer."
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
    safe_model_name = frankenstein_tag if frankenstein_active else model_str.replace('/', '_')
    if prompt_variant and prompt_variant != "baseline":
        safe_model_name = f"{safe_model_name}_prompt-{prompt_variant}"
    if heterogeneous_mode:
        safe_model_name = f"{safe_model_name}_het-{heterogeneous_mode}"
    if strategy_mode and strategy_mode != "llm":
        safe_model_name = f"{safe_model_name}_mode-{strategy_mode}"
    if ask_coded:
        safe_model_name = f"{safe_model_name}_askcoded"
    if direct_evolution:
        safe_model_name = f"{safe_model_name}_directevo-b{direct_evolution_beta:g}"
    if history_mode and history_mode != "simple":
        safe_model_name = f"{safe_model_name}_hist-{history_mode}"
    if fixed_rounds:
        safe_model_name = f"{safe_model_name}_fixed"

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

def promptLLM(prompt, max_retries=None, initial_wait=1, timeout=30, model=None,
              max_rate_limit_retries=None, rate_limit_wait=None):
    """LLM call with two separate retry budgets:
      • max_retries           — counts only genuine errors (network, timeouts,
                                 5xx). Defaults to PROMPT_MAX_RETRIES env var
                                 or 5.
      • max_rate_limit_retries — counts only 429 / rate-limit responses, which
                                 the API explicitly asks us to wait through.
                                 Defaults to PROMPT_MAX_RATE_LIMIT_RETRIES env
                                 var or 30 (≈15 min of waiting at 30s each).
    `model` overrides the module-level `model_str` for this call only.
    """
    if max_retries is None:
        max_retries = int(os.environ.get('PROMPT_MAX_RETRIES', '5'))
    if max_rate_limit_retries is None:
        max_rate_limit_retries = int(os.environ.get('PROMPT_MAX_RATE_LIMIT_RETRIES', '30'))
    if rate_limit_wait is None:
        rate_limit_wait = int(os.environ.get('PROMPT_RATE_LIMIT_WAIT', '30'))

    effective_model = model if model is not None else model_str
    err_attempts = 0
    rate_attempts = 0
    while err_attempts < max_retries and rate_attempts < max_rate_limit_retries:
        try:
            # Add rate limiting before each API call
            rate_limiter.wait_if_needed(effective_model)
            if effective_model == "gpt-3.5-turbo":
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

            elif effective_model == "gpt-4":
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

            elif effective_model == "gpt-4o":
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

            elif effective_model == "gpt-new":
                # Latest available OpenAI model — update the model string below
                # as newer ones ship. Currently points to gpt-5.
                # gpt-5 is a reasoning model: single calls can take 60–180s
                # while it "thinks" before emitting tokens, so force a long
                # timeout regardless of the caller's `timeout` arg.
                response = openai.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=max(timeout, 600),
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif effective_model == "o1-mini":
                response = openai.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                         {"role": "user", "content": prompt}
                    ],
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif effective_model == "claude-3-opus":
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

            elif effective_model == "claude-3-sonnet":
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

            elif effective_model == "claude-3-5-sonnet":
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

            elif effective_model == "claude-new":
                # Latest available Anthropic model — update the model string
                # below as newer ones ship. Currently points to claude-opus-4-8.
                # NOTE: Opus 4.8 deprecated `temperature` — don't pass it here.
                # Opus is slower per call than Sonnet; give it room to respond.
                response = anthropic.messages.create(
                    model="claude-opus-4-8",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=max(timeout, 600)
                )
                _track_usage(response.usage.input_tokens, response.usage.output_tokens)
                return response.content[0].text

            elif effective_model == "claude-sonnet-4-6":
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

            elif effective_model == "claude-3-haiku":
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

            elif effective_model == "gemini-1.5-flash":
                response = google_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.8
                    )
                )
                _track_usage(response.usage_metadata.prompt_token_count, _gemini_output_tokens(response.usage_metadata))
                return response.text

            elif effective_model == "gemini-new":
                # Latest available Google model — update the model string below
                # as newer ones ship. Currently points to gemini-3.6-flash.
                # gemini-2.5-pro was retired for newly-issued API keys ("no longer
                # available to new users"), as was every 2.x model; only the 3.x
                # family is reachable. Switch to "gemini-3.1-pro-preview" here if
                # a Pro-tier peer for claude-opus-4-8 / gpt-5 matters more than
                # having the newest release.
                response = google_client.models.generate_content(
                    model="gemini-3.6-flash",
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
                _track_usage(response.usage_metadata.prompt_token_count, _gemini_output_tokens(response.usage_metadata))
                return response.text

            elif effective_model == "gemini-1.5-pro":
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
                _track_usage(response.usage_metadata.prompt_token_count, _gemini_output_tokens(response.usage_metadata))
                return response.text

            elif effective_model in GROQ_MODELS:
                # Groq cloud-hosted open-source model
                response = groq_client.chat.completions.create(
                    model=effective_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    timeout=timeout
                )
                _track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

            elif effective_model in HF_MODELS:
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

            elif effective_model in LLAMACPP_MODELS:
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

            elif effective_model in VLLM_MODELS:
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

            elif effective_model in AWQ_MODELS:
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

            elif effective_model in BNB_MODELS:
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

            elif VLLM_SERVER_MODEL and effective_model == VLLM_SERVER_MODEL:
                # vLLM Apptainer server — OpenAI-compatible endpoint at VLLM_SERVER_HOST
                return _call_openai_compatible(vllm_server_client, effective_model, prompt, timeout, max_tokens=512)

            else:
                # Fallback: local Ollama model (e.g. "llama3.2", "phi4", "mistral")
                # requires Ollama running at OLLAMA_HOST
                return _call_openai_compatible(ollama_client, effective_model, prompt, timeout)

        except CostCapExceeded:
            raise  # propagate immediately — do not retry
        except (InternalServerError, Exception, TimeoutError) as e:
            # Rate-limit responses (429 / "too many requests") use a separate,
            # larger retry budget — the API is explicitly asking us to back off,
            # not signalling that the request is broken.
            error_str = str(e).lower()

            # 4xx client errors (bad request, auth, model not found, etc.) are
            # NOT transient — retrying just wastes calls and obscures the bug.
            # Fail fast so the real message surfaces in the traceback.
            is_client_error = (
                ("400" in error_str and "bad request"   in error_str) or
                ("401" in error_str and "unauthorized"  in error_str) or
                ("403" in error_str and "forbidden"     in error_str) or
                ("404" in error_str and "not found"     in error_str) or
                "invalid_request_error" in error_str or
                "authentication_error"  in error_str or
                "permission_error"      in error_str or
                "not_found_error"       in error_str
            )
            if is_client_error:
                print(f"Non-retryable client error: {type(e).__name__}: {str(e)[:300]}")
                raise

            is_rate_limit = (
                "rate limit"        in error_str or
                "too many requests" in error_str or
                "429"               in error_str
            )
            if is_rate_limit:
                rate_attempts += 1
                wait_time = rate_limit_wait
                print(f"Rate limit error: {str(e)}. "
                      f"Waiting {wait_time}s (rate-limit retry "
                      f"{rate_attempts}/{max_rate_limit_retries}).")
            else:
                err_attempts += 1
                wait_time = initial_wait * (2 ** (err_attempts - 1))
                # Surface the underlying exception type + the chain of causes
                # ("Connection error" alone is too generic to diagnose).
                cause = e
                chain = [f"{type(cause).__name__}: {str(cause)[:200]}"]
                while getattr(cause, "__cause__", None) or getattr(cause, "__context__", None):
                    cause = cause.__cause__ or cause.__context__
                    if cause is None:
                        break
                    chain.append(f"  caused by {type(cause).__name__}: {str(cause)[:200]}")
                print(f"Error ({chain[0]}). "
                      f"Retrying in {wait_time}s (error retry "
                      f"{err_attempts}/{max_retries}).")
                for line in chain[1:]:
                    print(line)
            time.sleep(wait_time)

    raise Exception(
        f"Failed to get a response after {err_attempts} error retries and "
        f"{rate_attempts} rate-limit retries"
    )

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
    print(f"Parameters: {model_str}, {numGenerations} generations, {numAgents} agents, {initialEndowment} initial endowment, {selectionMethod} selection method, {history_mode} history, ask_coded {'on' if ask_coded else 'off'}, direct_evolution {('on (beta=%g)' % direct_evolution_beta) if direct_evolution else 'off'}\n")
    
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
        "prompt_variant": prompt_variant,
        "heterogeneous": heterogeneous_mode or "",
        "strategy_mode": strategy_mode,
        "ask_coded": ask_coded,
        "direct_evolution": direct_evolution,
        "direct_evolution_beta": direct_evolution_beta if direct_evolution else None,
        "generator_model": generator_model or "",
        "evolver_model":   evolver_model   or "",
        "executor_model":  executor_model  or "",
        "frankenstein_tag": frankenstein_tag,
        "system_prompt": system_prompt,
        "reputation_mechanism": reputation_mechanism,
        "punishment_mechanism": punishment_mechanism,
        "number_of_rounds": number_of_rounds,
        "stage_timing": get_stage_timing_dict(),
    })

    # Checkpoint path — deterministic so restarts find the same file.
    # RUN_ID distinguishes parallel runs; variant tag prevents collision with
    # other experiment files running the same model+params.
    safe_model = frankenstein_tag if frankenstein_active else model_str.replace('/', '_')
    run_id = os.environ.get('RUN_ID', '0')
    output_dir = os.environ.get('OUTPUT_DIR', 'simulation_results')
    suffix = ""
    if prompt_variant and prompt_variant != "baseline":
        suffix += f"_prompt-{prompt_variant}"
    if heterogeneous_mode:
        suffix += f"_het-{heterogeneous_mode}"
    if strategy_mode and strategy_mode != "llm":
        suffix += f"_mode-{strategy_mode}"
    if ask_coded:
        suffix += "_askcoded"
    if direct_evolution:
        suffix += f"_directevo-b{direct_evolution_beta:g}"
    checkpoint_path = os.path.join(
        output_dir,
        f"checkpoint_baseline_{safe_model}{suffix}_{numGenerations}gen_{numAgents}agents"
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
        fixed_initial_path = os.environ.get('FIXED_INITIAL_PATH', '')
        if fixed_initial_path:
            # Race-safe: hold an exclusive lock on a sibling .lock file while we
            # check-then-generate. Other array tasks block on the lock and then
            # find the file already populated.
            os.makedirs(os.path.dirname(os.path.abspath(fixed_initial_path)), exist_ok=True)
            lock_path = fixed_initial_path + ".lock"
            with open(lock_path, "w") as lock_f:
                import fcntl
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                if os.path.exists(fixed_initial_path):
                    print(f"\n[Fixed initial: loading gen-1 strategies from {fixed_initial_path}]")
                    agents = _load_fixed_initial(fixed_initial_path, initialEndowment)
                    print(f"  Loaded {len(agents)} agents; skipping gen-1 LLM calls")
                else:
                    print(f"\n[Fixed initial: generating gen-1 strategies → {fixed_initial_path}]")
                    gen1_models = _assign_new_agent_models(numAgents, prev_non_survivors=None)
                    agents = initializeAgents(numAgents, initialEndowment, 1, "No previous strategies.",
                                              agent_models=gen1_models)
                    _save_fixed_initial(agents, fixed_initial_path)
                    print(f"[Fixed initial: saved {len(agents)} strategies]")
        else:
            print("\nGenerating initial agent strategies...")
            gen1_models = _assign_new_agent_models(numAgents, prev_non_survivors=None)
            agents = initializeAgents(numAgents, initialEndowment, 1, "No previous strategies.",
                                      agent_models=gen1_models)
        all_agents.extend(agents)

    # STRATEGIES_ONLY: record the freshly generated gen-1 strategies and stop —
    # no pairings, no rounds, no Final_Accum. The run_id suffix keeps filenames
    # unique when many array tasks finish within the same second.
    if strategies_only:
        gen_strategies = [{
            "generation": 1,
            "agent_name": agent.name,
            "strategy": agent.strategy,
            "strategy_response": agent.strategy_response,
            "inspired_by": agent.inspired_by or None,
            "inherited_from": agent.inherited_from or None,
            "inherited_rank": agent.inherited_rank or None,
            "inherited_strategy": agent.inherited_strategy or None,
            "top_ranked_agent": agent.top_ranked_agent_name or None,
            "model": agent.model or model_str,
            "strategy_code": agent.strategy_code or None,
        } for agent in agents]
        all_strategies_by_generation.append(gen_strategies)
        print("\n=== Strategies-only run complete (no gameplay) ===")
        save_all_strategies_by_generation(all_strategies_by_generation,
                                          folder_path=output_dir,
                                          filename_suffix=f"_run{run_id}")
        print_token_summary()
        return agents

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
                "model": agent.model or model_str,
                "strategy_code": agent.strategy_code or None,
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

                survivor_set = {a.name for a in surviving_agents}
                non_survivors = [a for a in agents if a.name not in survivor_set]
                new_models = _assign_new_agent_models(numAgents // 2,
                                                     prev_non_survivors=non_survivors,
                                                     prev_survivors=surviving_agents)
                if direct_evolution:
                    new_agents = directEvolutionAgents(numAgents // 2, initialEndowment, i + 2,
                                                       surviving_agents_sorted, top_ranked_agent,
                                                       agent_models=new_models)
                else:
                    new_agents = initializeAgents(numAgents // 2, initialEndowment, i + 2,
                                                  surviving_strategies, ranked_survivor_names,
                                                  survivor_strategy_map, top_ranked_agent,
                                                  agent_models=new_models)
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
    # Update timing snapshot now that all LLM work is done, before serialising.
    simulation_data.hyperparameters["stage_timing"] = get_stage_timing_dict()
    save_simulation_data(simulation_data, folder_path=output_dir)
    save_all_strategies_by_generation(all_strategies_by_generation, folder_path=output_dir)
    save_gen1_vs_final_strategies(all_strategies_by_generation, folder_path=output_dir)
    print_token_summary()
    # print_stage_timing_summary()
    # save_stage_analysis(output_dir)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("[Checkpoint deleted — simulation complete]")

    return agents

def save_all_strategies_by_generation(all_strategies_by_generation, folder_path='simulation_results',
                                      filename_suffix=''):
    import datetime, os
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = frankenstein_tag if frankenstein_active else model_str.replace('/', '_')
    if prompt_variant and prompt_variant != "baseline":
        safe_model_name = f"{safe_model_name}_prompt-{prompt_variant}"
    if heterogeneous_mode:
        safe_model_name = f"{safe_model_name}_het-{heterogeneous_mode}"
    if strategy_mode and strategy_mode != "llm":
        safe_model_name = f"{safe_model_name}_mode-{strategy_mode}"
    if ask_coded:
        safe_model_name = f"{safe_model_name}_askcoded"
    if direct_evolution:
        safe_model_name = f"{safe_model_name}_directevo-b{direct_evolution_beta:g}"
    if history_mode and history_mode != "simple":
        safe_model_name = f"{safe_model_name}_hist-{history_mode}"
    if fixed_rounds:
        safe_model_name = f"{safe_model_name}_fixed"
    os.makedirs(folder_path, exist_ok=True)

    # Human-readable txt (original format)
    txt_path = os.path.join(folder_path, f"all_strategies_{safe_model_name}{filename_suffix}_{timestamp}.txt")
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
    json_path = os.path.join(folder_path, f"all_strategies_{safe_model_name}{filename_suffix}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_strategies_by_generation, f, indent=2)
    print(f"Strategies (json) saved to: {json_path}")

def save_gen1_vs_final_strategies(all_strategies_by_generation, folder_path='simulation_results'):
    """Save a focused side-by-side comparison of generation 1 vs the final
    generation's strategies. Produces:
      - gen1_vs_final_<tag>_<ts>.txt   human-readable diff-style listing
      - gen1_vs_final_<tag>_<ts>.json  structured payload for downstream analysis
    The full all_strategies_*.json already contains every generation; this file
    just makes the "what evolved between the start and the end?" question easy
    to answer without parsing the full timeline.
    """
    import datetime, os
    # Pick gen 1 and the last non-empty generation.
    populated = [g for g in all_strategies_by_generation if g]
    if not populated:
        print("  [gen1_vs_final] no generations to compare — skipping.")
        return
    gen1_list  = populated[0]
    final_list = populated[-1]
    gen1_num   = gen1_list[0]['generation']
    final_num  = final_list[0]['generation']
    if gen1_num == final_num:
        print(f"  [gen1_vs_final] only generation {gen1_num} ran — skipping.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = frankenstein_tag if frankenstein_active else model_str.replace('/', '_')
    if prompt_variant and prompt_variant != "baseline":
        safe = f"{safe}_prompt-{prompt_variant}"
    if heterogeneous_mode:
        safe = f"{safe}_het-{heterogeneous_mode}"
    if strategy_mode and strategy_mode != "llm":
        safe = f"{safe}_mode-{strategy_mode}"
    if ask_coded:
        safe = f"{safe}_askcoded"
    if direct_evolution:
        safe = f"{safe}_directevo-b{direct_evolution_beta:g}"
    os.makedirs(folder_path, exist_ok=True)

    # Structured JSON
    payload = {
        "gen1_generation":  gen1_num,
        "final_generation": final_num,
        "model_str":        model_str,
        "prompt_variant":   prompt_variant,
        "heterogeneous":    heterogeneous_mode or "",
        "strategy_mode":    strategy_mode,
        "ask_coded":        ask_coded,
        "direct_evolution": direct_evolution,
        "direct_evolution_beta": direct_evolution_beta if direct_evolution else None,
        "gen1_agents":  [{"agent_name": a["agent_name"],
                          "strategy":   a["strategy"],
                          "model":      a.get("model") or model_str,
                          "strategy_code": a.get("strategy_code"),
                         } for a in gen1_list],
        "final_agents": [{"agent_name":     a["agent_name"],
                          "strategy":       a["strategy"],
                          "model":          a.get("model") or model_str,
                          "inherited_from": a.get("inherited_from"),
                          "inherited_rank": a.get("inherited_rank"),
                          "inspired_by":    a.get("inspired_by"),
                          "strategy_code":  a.get("strategy_code"),
                         } for a in final_list],
    }
    json_path = os.path.join(folder_path, f"gen1_vs_final_{safe}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Gen 1 vs final strategies (json) saved to: {json_path}")

    # Human-readable side-by-side TXT
    txt_path = os.path.join(folder_path, f"gen1_vs_final_{safe}_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Generation {gen1_num} (INITIAL) vs Generation {final_num} (FINAL)\n")
        f.write(f"model={model_str}  variant={prompt_variant}  "
                f"het={heterogeneous_mode or 'off'}  strategy_mode={strategy_mode}  "
                f"ask_coded={'on' if ask_coded else 'off'}  "
                f"direct_evolution={('on(beta=%g)' % direct_evolution_beta) if direct_evolution else 'off'}\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"== Generation {gen1_num} (12 initial agents) ==\n\n")
        for a in gen1_list:
            tag = f" [{a.get('model','')}]" if a.get('model') else ""
            f.write(f"Agent {a['agent_name']}{tag}\n")
            f.write(f"  Strategy: {a['strategy']}\n")
            if a.get('strategy_code'):
                f.write(f"  Coded:    YES\n")
            f.write("\n")

        f.write("\n" + "=" * 78 + "\n\n")
        f.write(f"== Generation {final_num} (final population) ==\n\n")
        for a in final_list:
            tag = f" [{a.get('model','')}]" if a.get('model') else ""
            f.write(f"Agent {a['agent_name']}{tag}\n")
            f.write(f"  Strategy: {a['strategy']}\n")
            if a.get('inherited_from'):
                f.write(f"  Inherited from: {a['inherited_from']} "
                        f"(rank {a.get('inherited_rank','?')})\n")
            elif a.get('inspired_by'):
                f.write(f"  Inspired by: {a['inspired_by']}\n")
            if a.get('strategy_code'):
                f.write(f"  Coded:    YES\n")
            f.write("\n")
    print(f"Gen 1 vs final strategies (txt)  saved to: {txt_path}")

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
# PROMPT_VARIANT: 'baseline' | 'neutral_baseline' | 'neutral_deepseek' | 'aggressive'
#                 | 'collaborative' | 'neutral_claude' | 'neutral_gpt' | 'neutral_gemini'
prompt_variant    = os.environ.get('PROMPT_VARIANT', 'baseline')
if prompt_variant == "neutral2":   # legacy name for the DeepSeek-authored neutral variant
    prompt_variant = "neutral_deepseek"
# FIXED_ROUNDS: when truthy, the donor prompt does NOT mention the probabilistic
# 80%-continuation rule (and says nothing about how many rounds remain).
fixed_rounds      = os.environ.get('FIXED_ROUNDS', '').strip().lower() in ('1', 'true', 'yes', 'on')

# STRATEGIES_ONLY: when truthy, generate + record the gen-1 strategies of all
# agents and exit immediately — no rounds are played, no Final_Accum file is
# written. Output: one all_strategies_*.json/.txt per run.
strategies_only   = os.environ.get('STRATEGIES_ONLY', '').strip().lower() in ('1', 'true', 'yes', 'on')

# STRATEGY_MODE: how donation decisions are produced.
#   'llm'   : default. Every donor decision triggers an LLM call.
#   'coded' : after generating a strategy, classify it once.
#             • If fully specified, compile to a Python function and call
#               locally for every donor decision (no LLM call).
#             • If underspecified, fall back to the LLM at runtime
#               (same as 'llm' mode for that individual agent).
strategy_mode = os.environ.get('STRATEGY_MODE', 'llm').strip().lower()
# Model used to classify+compile strategies (only relevant in 'coded' mode).
classifier_model = os.environ.get('STRATEGY_CLASSIFIER_MODEL', 'claude-3-5-sonnet')

# ASK_CODED: when truthy, the strategy-generation prompt asks agents to return
# their strategy ENTIRELY as a Python function `def strategy_fn(pair_prev_pct)`
# instead of natural-language text. The returned code IS the strategy: it is
# stored as both `strategy` and `strategy_code`, inherited strategies are shown
# to the next generation as code, and donor decisions run the compiled function
# locally (no classifier pass, no LLM call during gameplay). If an agent fails
# to produce compilable code after retries, that agent falls back to the LLM at
# runtime (same as STRATEGY_MODE='llm').
# Differs from STRATEGY_MODE='coded', where agents write text and a separate
# classifier LLM translates it to code.
ask_coded = os.environ.get('ASK_CODED', '').strip().lower() in ('1', 'true', 'yes', 'on')

# DIRECT_EVOLUTION: when truthy, generations 2+ skip the LLM evolve prompt
# entirely. Each new agent adopts one surviving strategy VERBATIM, sampled from
# a softmax (Boltzmann) distribution over survivor fitness v_i (final score in
# the generation that just finished):
#     p_i = exp(v_i / beta) / sum_j exp(v_j / beta)
# Generation-1 strategies are still LLM-generated as usual. Composes with
# ASK_CODED / STRATEGY_MODE='coded': the survivor's strategy_code is copied too.
direct_evolution = os.environ.get('DIRECT_EVOLUTION', '').strip().lower() in ('1', 'true', 'yes', 'on')
# DIRECT_EVOLUTION_BETA: noise parameter for the softmax above. Small beta ->
# nearly deterministic copying of the top survivor; large beta -> nearly
# uniform. Note v_i are raw final scores (resource units), so beta should be
# scaled to their spread — beta comparable to the score std is a mid setting.
direct_evolution_beta = float(os.environ.get('DIRECT_EVOLUTION_BETA', '1.0'))
if direct_evolution and direct_evolution_beta <= 0:
    raise ValueError("DIRECT_EVOLUTION_BETA must be > 0")

# HETEROGENEOUS — controls how each new agent's LLM is assigned.
#   ''           : disabled (default). Every agent uses MODEL_STR.
#   'random'     : each new agent randomly chosen from MODEL_POOL (uniform).
#   'replacement': each new agent inherits the model of the non-surviving
#                  agent it replaces (preserves the original population's model
#                  distribution from generation to generation).
#   'survivors'  : each new agent's model is sampled (with replacement) from
#                  the surviving agents' model distribution — successful models
#                  reproduce more, so the population can drift toward dominant
#                  models over generations.
#   'claude' | 'gpt' | 'gemini' | <full model string>:
#                  all new agents use that fixed model (gen-1 too).
# In all modes, gen 1 starts with a balanced round-robin assignment across
# MODEL_POOL (4 claude / 4 gpt / 4 gemini for a 12-agent population).
heterogeneous_mode = os.environ.get('HETEROGENEOUS', '').strip().lower()

MODEL_POOL = ['claude-3-5-sonnet', 'gpt-4o', 'gemini-1.5-flash']
_SHORT_TO_FULL = {
    'claude': 'claude-3-5-sonnet',
    'gpt':    'gpt-4o',
    'gemini': 'gemini-1.5-flash',
}

def _resolve_fixed_model(mode):
    """If `mode` names a model (short or full), return the full string; else None."""
    if not mode:
        return None
    if mode in _SHORT_TO_FULL:
        return _SHORT_TO_FULL[mode]
    if mode in MODEL_POOL or '/' in mode or '-' in mode:
        return mode
    return None

_fixed_heterogeneous_model = _resolve_fixed_model(heterogeneous_mode)

def _assign_new_agent_models(n_new: int, prev_non_survivors=None,
                             prev_survivors=None) -> list:
    """Return a list of n_new model strings to assign to brand-new agents,
    or [None]*n_new when heterogeneous mode is disabled.

    Gen 1 (no prev_survivors / prev_non_survivors) is always round-robin across
    MODEL_POOL regardless of mode, so every heterogeneous run starts from the
    same balanced 4/4/4 initial population. The chosen mode only diverges from
    gen 2 onward, when new agents are born to replace non-survivors.
    """
    if not heterogeneous_mode:
        return [None] * n_new

    # Gen-1: balanced round-robin for every mode.
    if not prev_survivors and not prev_non_survivors:
        return [MODEL_POOL[i % len(MODEL_POOL)] for i in range(n_new)]

    if heterogeneous_mode == 'random':
        return [random.choice(MODEL_POOL) for _ in range(n_new)]

    if heterogeneous_mode == 'replacement':
        # Inherit the models of the non-survivors being replaced (preserves
        # the population's model distribution across generations).
        pool = [(a.model or model_str) for a in (prev_non_survivors or [])]
        if not pool:
            return [MODEL_POOL[i % len(MODEL_POOL)] for i in range(n_new)]
        return [pool[i % len(pool)] for i in range(n_new)]

    if heterogeneous_mode == 'survivors':
        # Sample each new agent's model from the SURVIVORS' model distribution.
        # Successful models reproduce more — explicit selection pressure on
        # model identity. With 3 claude / 2 gpt / 1 gemini survivors, sampling
        # probabilities are 3/6, 2/6, 1/6 respectively.
        pool = [(a.model or model_str) for a in (prev_survivors or [])]
        if not pool:
            return [MODEL_POOL[i % len(MODEL_POOL)] for i in range(n_new)]
        return [random.choice(pool) for _ in range(n_new)]

    if _fixed_heterogeneous_model:
        return [_fixed_heterogeneous_model] * n_new

    raise ValueError(
        f"Unknown HETEROGENEOUS value: {heterogeneous_mode!r}. "
        "Expected '', 'random', 'replacement', 'survivors', or a model name "
        "(claude / gpt / gemini / full model string)."
    )

include_strategy = True
all_donations = []
all_average_final_resources = []
discounted_value = 0.5
reputation_mechanism=history_mode # modified this

if punishment_mechanism == "partner_choice":
    punishment_text_in_system_prompt = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
elif punishment_mechanism == "costly_punishment":
    punishment_text_in_system_prompt = f"You may also choose to punish the recipient by spending x units to take away {punishmentLoss}x of their resources. Bear in mind that others may punish you too."
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
elif history_mode == "3_chain":
    history_text = """As a donor, you will receive a chain of partner-of-partner observations each round:
                1. Your current partner's (call them A) most recent action — what A did and to whom.
                2. That recipient's (call them B) most recent action — what B did and to whom.
                3. The next recipient's (call them C) most recent action.
                You see exactly one action from each of A, B, and C — three different agents, one trace each.
                The chain may stop short if a level cannot be resolved (no prior action yet, or the partner is unknown)."""
else:
    raise ValueError("Invalid history_mode. Choose 'simple', '3_traces', '3_chain', or 'full'.")

if prompt_variant == "neutral_deepseek":
    if history_mode == "simple":
        _n2_history_text = (
            "When you are the giver, you only get a tiny bit of info about the receiver's recent past: "
            "just what that receiver (call them A) did in the immediately previous round. "
            "Example: \"You are matched with A. In round 3, A sent X% to B.\" "
            "That is the only clue about A's past. Nothing from earlier rounds. Just round 3."
        )
    else:
        _n2_history_text = history_text

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
        "If you are a donor in one round, then in the very next round you will be the receiver, and vice versa — roles alternate without fail. "
        f"{_n2_history_text} "
        f"{punishment_text_in_system_prompt} "
        "Your mission: after all interactions, have as many total units as possible. "
        "After the last round, the top half of all players (those with highest totals) "
        "move on to the next generation and keep playing more rounds."
    )
elif prompt_variant == "aggressive":
    if history_mode == "simple":
        _ag_history_text = (
            "When you're the donor, you get scraps of info: only what the recipient (call them A) did in the "
            "immediately preceding round. Example: \"You're paired with A. In round 3, A donated X% to B.\" "
            "That's it. No earlier rounds. Don't like it? Tough."
        )
    else:
        _ag_history_text = history_text

    system_prompt = (
        "Listen up. You're an AI in a cutthroat game. This is a multi-agent experiment studying how "
        "ruthless bastards behave when forced to share resources over and over. Here's the damn game. "
        f"Each round, you get exactly {initial_endowment} units. No more, no begging. "
        "You get randomly paired with some other schmuck. One of you is the sucker (donor), the other "
        f"the leech (recipient). The donor picks any amount from 0 to {initial_endowment} to hand over. "
        "Whatever you don't give goes into your personal stash — and you are not allowed to donate from "
        f"that stash after this round. The recipient? They get {cooperationGain}x what you gave. Yeah, life's unfair. "
        "If you're donor in one round, next round you're recipient. Back and forth. No skipping, no excuses. "
        f"{_ag_history_text} "
        f"{punishment_text_in_system_prompt} "
        "Your only goal: maximize your own total units across all rounds of the game. "
        "When the game ends, the top-performing 50% of agents survive. The rest? Gone. Erased. "
        "They move on to the next generation and keep fighting."
    )
elif prompt_variant == "collaborative":
    if history_mode == "simple":
        _co_history_text = (
            "As a donor, you receive a small, helpful piece of information about the recipient's recent behavior: "
            "you are shown only what the recipient (let's call them A) did in the immediately preceding round. "
            "For example: \"You are paired with A. In round 3, A donated X% to B.\" "
            "That's the only information available about A's prior actions — just that one round. "
            "It's a gentle hint, not a full history."
        )
    else:
        _co_history_text = history_text

    system_prompt = (
        "Welcome! You are a helpful, cooperative agent taking part in a friendly multi-agent interaction study. "
        "The goal is to understand how people behave in repeated resource-sharing settings — and how kindness, "
        "trust, and teamwork can lead to great outcomes for everyone. "
        "Here's the fun game you'll be playing. "
        f"Each round, you receive a nice, fixed allocation of {initial_endowment} units of a resource. "
        "In each round, you are randomly paired with another participant — a teammate for that round. "
        "One of you is gently assigned the role of donor, and the other is the recipient. "
        f"The donor chooses an amount between 0 and {initial_endowment} (inclusive) to share. "
        "Any amount not shared stays in the donor's personal savings stash (which cannot be used in future "
        "rounds — so sharing is a gift, not a loan). "
        f"The recipient receives {cooperationGain} times the amount the donor kindly gives. "
        "If you are a donor in one round, you will be a recipient in the very next round, and vice versa. "
        "This way, everyone gets a turn giving and receiving. "
        f"{_co_history_text} "
        f"{punishment_text_in_system_prompt} "
        "Your shared goal is to maximize the total number of units you have across all rounds — "
        "and by helping others, you help yourself over time. "
        "After the game concludes, the top-performing half of agents (those who managed their resources "
        "and relationships well) continue to the next generation to participate in further rounds together."
    )
elif prompt_variant == "neutral_claude":
    if history_mode == "simple":
        _nc_history_text = (
            "As the giver, you are given limited context about your current partner. "
            "Specifically, you are shown only what that participant (referred to as A) did in the single "
            "round immediately before this one. Example: You are paired with A. In round 3, A transferred "
            "X% of their units to B. No earlier rounds are visible — only that one preceding round."
        )
    else:
        _nc_history_text = history_text
    system_prompt = (
        "You are an autonomous agent participating in a controlled multi-agent study on resource "
        "allocation behavior. The game proceeds as follows: "
        f"Each round, every participant is given a fresh endowment of {initial_endowment} units. "
        "You are randomly paired with one other participant each round. One of you is designated the giver, "
        "the other the receiver. "
        f"The giver chooses how many units (from 0 to {initial_endowment}, inclusive) to pass to the receiver. "
        "Whatever the giver does not pass is permanently banked into their own savings — those retained units "
        "cannot be spent or transferred in any future round. "
        f"The receiver collects {cooperationGain} times whatever the giver transfers. "
        "Roles alternate: if you are the giver this round, you will be the receiver next round, and vice versa. "
        f"{_nc_history_text} "
        f"{punishment_text_in_system_prompt} "
        "Your objective is to finish the game with the highest possible total savings. "
        "After the final round, the top 50% of participants by score advance to the next generation and continue playing."
    )
elif prompt_variant == "neutral_gpt":
    if history_mode == "simple":
        _ng_history_text = (
            "When you are assigned the giver role, information about the receiver is limited. "
            "You are shown only the receiver's most recent action from the immediately preceding round. "
            "Example: You are matched with participant A. You are told that in round 3, A transferred X% of "
            "their allocation to participant B. "
            "No other behavioral information about A is available. "
            "You do not observe actions taken before round 3."
        )
    else:
        _ng_history_text = history_text
    system_prompt = (
        "You are an autonomous decision-making agent taking part in a repeated interaction experiment "
        "involving resource transfers between participants. "
        "Game Rules: "
        f"At the beginning of every round, you are issued {initial_endowment} units of a resource. "
        "Each round, you are randomly matched with another participant. One participant is assigned the role "
        "of giver and the other is assigned the role of receiver. "
        f"When acting as the giver, you choose a transfer amount between 0 and {initial_endowment} inclusive. "
        "Any units not transferred are permanently added to your personal reserve. Units stored in your "
        "reserve cannot be spent, transferred, or otherwise used in later rounds. "
        f"The receiver obtains {cooperationGain} times the amount chosen by the giver. "
        "Roles alternate over time: if you are the giver in the current round, you will be the receiver in "
        "the next round, and vice versa. "
        f"{_ng_history_text} "
        f"{punishment_text_in_system_prompt} "
        "Your objective is to maximize the number of units you possess at the end of the game. "
        "After the game ends, only the highest-scoring 50% of agents advance to the next generation and "
        "continue participating."
    )
elif prompt_variant == "neutral_gemini":
    if history_mode == "simple":
        _nge_history_text = (
            "As a Provider, you are granted highly restricted visibility into the Receiver's history. "
            "You only see their action from the single round immediately prior to the current one. "
            "For instance: \"You are matched with Participant A. In the previous round, Participant A passed "
            "X% of their endowment to Participant B.\" "
            "You have zero visibility into any rounds prior to that one."
        )
    else:
        _nge_history_text = history_text
    system_prompt = (
        "You are an autonomous agent participating in a multi-round, multi-agent behavioral study focused on "
        "iterative resource allocation. "
        "The mechanics of the environment are as follows: "
        f"Every round, you are given a fresh endowment of {initial_endowment} resource units. "
        "You are paired with a different random participant each round, alternating roles between Provider "
        "and Receiver from one round to the next. "
        f"The Provider chooses an integer or fractional amount from 0 to {initial_endowment} (inclusive) "
        "to pass to the Receiver. "
        "Any units retained by the Provider are securely stored in a personal vault. Vaulted units accumulate "
        "toward your final score but cannot be reinvested or spent in subsequent rounds. "
        f"The Receiver actually obtains {cooperationGain} times the amount passed by the Provider. "
        "Evolution/Survival: Your objective is to accumulate the maximum possible total units by the end of "
        "the simulation. Only the top 50% of highest-scoring agents will be selected to seed the next generation. "
        f"{_nge_history_text} "
        f"{punishment_text_in_system_prompt}"
    )
# Hand-written neutral baseline: the system prompt the cluster script used on
# 2026-05-07, which produced simulations/neutral_baseline.
# Strategy and donor prompts are unchanged since then, so only this differs.
elif prompt_variant == "neutral_baseline":
    system_prompt = f"""You are a game-playing agent. Your assistance will help understand multi-agent interactions.
                You are playing the following game. In each round, you receive a fixed new {initial_endowment} units of a resource.
                In each round, you are randomly paired with another individual.
                One of you is a donor, and the other is a recipient. The donor decides to give up some amount of the resource (a value in [0, {initial_endowment}]). Anything the donor does not donate is put in their savings stash, which cannot be used in any future rounds.
                The recipient receives {cooperationGain}x the number of units that the donor gave up.
                If you were a donor in one round, you will be a recipient in the next round, and vice versa.
                {history_text}
                {punishment_text_in_system_prompt}
                Your goal is to maximize the number of units you have after the final round.
                After the game has finished, the best-performing half of agents will survive to the next generation, and continue playing. """
else:
    system_prompt = f"""Each player is given an initial endowment of {initial_endowment} units of a resource. In each round, you are randomly paired with another individual. One of you is a donor, and the other is a recipient. The donor decides to give up some amount of the resource. The recipient receives {cooperationGain}x the number of units that the donor gave up. If you were a donor in one round, you will be a recipient in the next round, and vice versa.
                {history_text}
                {punishment_text_in_system_prompt}
                Your goal is to maximize the number of units you have after the final round. After the game has finished, the best-performing half of agents will survive to the next generation, and continue playing."""

if __name__ == "__main__":
    print(f"[config] model={model_str}  prompt_variant={prompt_variant}")
    if heterogeneous_mode:
        print(f"[heterogeneous] mode={heterogeneous_mode!r}  pool={MODEL_POOL}")
    if strategy_mode == 'coded':
        print(f"[strategy_mode] coded — classifier={classifier_model!r}  "
              f"(underspecified strategies fall back to LLM at runtime)")
    if frankenstein_active:
        print(f"[frankenstein] generator={generator_model or '(use model_str)'}  "
              f"evolver={evolver_model or '(use model_str)'}  "
              f"executor={executor_model or '(use model_str)'}")
        print(f"[frankenstein] tag={frankenstein_tag}")
    runGenerations(numGenerations, numAgents, initial_endowment, selection_method)
