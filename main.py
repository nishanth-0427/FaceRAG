import os
from dotenv import load_dotenv

# Load environment variables (like your Gemini API key)
load_dotenv()

def chunk_text(filepath):
    """Reads a text file and returns a list of non-empty lines (chunks)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        # Strip whitespace and keep only lines that aren't empty
        return [line.strip() for line in f if line.strip()]

# Let's test it to make sure it works!
if __name__ == "__main__":
    chunks = chunk_text("faceprep.txt")
    print(f"Successfully loaded {len(chunks)} chunks!")