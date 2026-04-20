from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Agent limits
MAX_ITERATIONS = 5          # max fix attempts before giving up
MAX_COST_PER_RUN = 0.10     # USD — hard ceiling per run
EXECUTION_TIMEOUT = 10      # seconds — hard timeout on subprocess

# Cost tracking
COST_PER_1M_INPUT_TOKENS = 0.27
COST_PER_1M_OUTPUT_TOKENS = 1.10

# Scoring
MIN_PASS_SCORE = 7          # evaluator score threshold to declare success (1-10)

PROMPT_VERSION = "v1.0.0"   # increment when system prompt changes