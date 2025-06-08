"""
Configuration manager for the joke generation pipeline.
Loads environment variables and provides configuration across modules.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:1234/v1/")

# LLM Model Selection
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma-3-4b-it-qat")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek/deepseek-chat:free")

# Pipeline Configuration
DEFAULT_THEME = os.getenv("DEFAULT_THEME", "penguins")
DEFAULT_NUM_IDEAS = int(os.getenv("DEFAULT_NUM_IDEAS", "3"))
DEFAULT_RUBRICS_PER_IDEA = int(os.getenv("DEFAULT_RUBRICS_PER_IDEA", "2"))
DEFAULT_CRITIQUES_PER_RUBRIC = int(os.getenv("DEFAULT_CRITIQUES_PER_RUBRIC", "1"))
DEFAULT_OUTPUT_FILE = os.getenv("DEFAULT_OUTPUT_FILE", "results.json")

# Baseline Configuration
BASELINE_OUTPUT_FILE = os.getenv("BASELINE_OUTPUT_FILE", "baseline.json")

def initialize_config():
    """
    Initialize and validate configuration.
    Returns True if valid, False otherwise.
    """
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Set it in your .env file or as an environment variable.")
        return False
    
    print(f"Configuration loaded:")
    print(f"- API Base URL: {LLM_API_BASE_URL}")
    print(f"- Default model: {DEFAULT_MODEL}")
    print(f"- Judge model: {JUDGE_MODEL}")
    return True

def get_api_base_url():
    return LLM_API_BASE_URL

def get_openai_key():
    return OPENAI_API_KEY

def get_openrouter_key():
    return OPENROUTER_API_KEY
