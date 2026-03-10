import os
from dotenv import load_dotenv

# Load from .env file at project root
load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Other options:
# BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# If you finetune the model, set MY_MODEL to "your-username/your-model-name"
MY_MODEL = None

HF_TOKEN = os.getenv("HF_TOKEN")

# PROJECT_ROOT is parent of src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")