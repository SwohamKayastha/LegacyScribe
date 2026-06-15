import os
import modal

# ──────────────────────────────────────────────────────────────────────────────
# 1. FIXED IMAGE WITH EXPLICIT CUDA LIBRARY PATHS
# ──────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("huggingface_hub", "fastapi[standard]")
    # Tell both llama-cpp and the Linux linker exactly where the CUDA runtime lives
    .env({
        "GGML_CUDA": "on",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
    })
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu122",
    )
)
app = modal.App("legacystribe-backend")

# Create a persistent storage cache volume to save your model binary file permanently
volume = modal.Volume.from_name("gguf-model-cache", create_if_missing=True)
CACHE_DIR = "/model_cache"

# Config constants based on your HF script
MODEL_REPO = "build-small-hackathon/legacystribe-Qwen3.5-9B.Q4_K_M"
MODEL_FILE = "Qwen3.5-9B.Q4_K_M.gguf"
N_CTX = 4096

# ──────────────────────────────────────────────────────────────────────────────
# 2. DEFINE SYSTEM PROMPTS AND AGENTS
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "questioner": (
        "/no_think "
        "You are a gentle memory guide helping an elderly person tell their life story. "
        "Ask exactly one warm, open follow-up question. Never ask more than one question. "
        "Be patient, kind, and culturally sensitive to Nepali and South Asian contexts."
    ),
    "extractor": (
        "/no_think "
        "You are an extractor agent. Given a memory fragment, extract structured information "
        "as JSON with keys relevant to the content (who, when, where, what, emotion). "
        "Output only valid JSON, nothing else."
    ),
    "arcdetector": (
        "/no_think "
        "You are an arc detector agent. Given a memory fragment, identify the narrative stage. "
        "Output one word only: setup, tension, turn, or meaning."
    ),
    "publisher": (
        "You are a publisher agent. Given memory notes, synthesize them into a single warm, "
        "narrative paragraph suitable for a family memory book. Write in first person. "
        "Use natural, unhurried language. Output only the paragraph, nothing else."
    ),
}

AGENT_DEFAULTS = {
    "questioner":  {"max_tokens": 2048, "temp": 0.7},
    "extractor":   {"max_tokens": 2048, "temp": 0.1},
    "arcdetector": {"max_tokens": 1024, "temp": 0.1},
    "publisher":   {"max_tokens": 2048, "temp": 0.4},
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCT SERVER CONTAINER CLASS
# ──────────────────────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    # gpu="A10G",                   # Fast, cost-efficient, handles 4096 context with ease
    gpu="T4",                   # Fast, cost-efficient, handles 4096 context with ease
    volumes={CACHE_DIR: volume},
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")]
)
class LegacyScribeServer:
    @modal.enter()
    def initialize_backend(self):
        """Runs automatically once when the server container spins up."""
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        # 1. Absolute path inside the volume mount directory
        local_model_path = os.path.join(CACHE_DIR, MODEL_FILE)

        # 2. Fetch from Hugging Face if it isn't in cache volume yet
        if not os.path.exists(local_model_path):
            print(f"📥 Downloading model {MODEL_FILE} from HF Hub to cache...")
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=CACHE_DIR
            )
            volume.commit() # Flush writes down to persistent network storage
            print("📦 Model downloaded successfully!")
        else:
            print("✅ Model already located in volume cache.")

        # 3. Instantiate model architecture inside active GPU VRAM
        print("🤖 Instantiating llama.cpp backend pipeline...")
        self.llm = Llama(
            model_path=local_model_path,
            n_ctx=N_CTX,
            n_gpu_layers=-1, # Shift 100% of the layers to the GPU
            verbose=False,
        )
        print("🚀 LegacyScribe model loaded and active!")

    @modal.fastapi_endpoint(method="POST")
    def predict(self, data: dict):
        """
        Accepts standard web requests.
        Expected JSON body: 
        {"agent": "questioner", "user_text": "text content", "max_tokens": -1, "temp": -1.0}
        """
        agent = data.get("agent")
        user_text = data.get("user_text", "")
        max_tokens = data.get("max_tokens", -1)
        temp = data.get("temp", -1.0)

        if agent not in SYSTEM_PROMPTS:
            return {"error": f"unknown agent: {agent}"}

        defaults = AGENT_DEFAULTS.get(agent, {"max_tokens": 256, "temp": 0.4})
        _max_tokens = defaults["max_tokens"] if max_tokens < 0 else int(max_tokens)
        _temp = defaults["temp"] if temp < 0 else float(temp)

        try:
            # Generate the response
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[agent]},
                    {"role": "user",   "content": f"/no_think\n{user_text}"},
                ],
                max_tokens=_max_tokens,
                temperature=_temp,
            )
            
            raw_text = response["choices"][0]["message"]["content"].strip()
            print(f"[RAW:{agent}] {repr(raw_text)}")
            return {"response": raw_text}

        except Exception as e:
            return {"error": str(e)}

    # Change here too!
    @modal.fastapi_endpoint(method="GET")
    def health(self):
        """Fast HTTP GET request endpoint for health checks."""
        return {"status": "ok"}