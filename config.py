import os

# ====== ENV ======
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
HF_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN", "")
HF_TEXT_MODEL = os.getenv("HUGGING_FACE_TEXT_MODEL", "google/flan-t5-large")
HF_IMAGE_MODEL = os.getenv("HUGGING_FACE_IMAGE_MODEL", "stabilityai/stable-diffusion-2")

# Mode: "webhook" untuk Render, "polling" untuk lokal
MODE = os.getenv("MODE", "webhook").lower()
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = os.getenv("BASE_URL", os.getenv("RENDER_EXTERNAL_URL", ""))  # https://yourservice.onrender.com

# ====== PATHS ======
DATA_DIR = os.getenv("DATA_DIR", "./campaigns")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
ABILITIES_PATH = os.getenv("ABILITIES_PATH", "./abilities.json")

# ====== OTHER ======
DELETE_CONFIRM_SECONDS = 120
MODEL_WAIT = True  # tunggu model HF warmup
