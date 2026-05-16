import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Find the folder where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Load the .env file from that folder
load_dotenv(BASE_DIR / ".env")

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found. Check that .env is in the same folder as this script.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

print("OpenAI client initialized successfully.")
print("API key loaded securely from environment variable.")