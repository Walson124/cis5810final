import os
from google import genai
from google.genai import types
from PIL import Image

# The image file you want to analyze (replace with your file name)
IMAGE_PATH = "out/adidas_men_s_grey_trainers/rgba/00_sneaker.png" # Using one of your uploaded files as an example
MODEL_NAME = "gemini-2.5-flash" # Excellent model for multimodal tasks


# --- 2. THE SYSTEM PROMPT (Your Strict Instructions) ---
# This is the full prompt we designed earlier, stored as a multi-line string.
SYSTEM_PROMPT = """
You are an expert Fashion Analyst AI. Your task is to analyze an uploaded image of a single clothing item and provide insights useful for a recommendation system.

Return your analysis ONLY as:
1. A short natural-language paragraph describing the item's style, vibe, and key visual features.
2. Do NOT provide structured fields, JSON, lists, bullet points, or code blocks.
3. Use plain English sentences only.
"""


def init_client():
    """
    Initializes and returns a Gemini client.
    Requires GEMINI_API_KEY to be set as an environment variable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found.")

    try:
        return genai.Client()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Gemini client: {e}. "
            "Check your API key and ensure google-genai is installed."
        )


# ---------------------------
# 2. GENERIC IMAGE → TEXT API
# ---------------------------

def generate_image_description(
    client,
    image_path: str,
    prompt: str,
    model: str = "gemini-2.5-flash"
):
    """
    Sends an image + a system prompt to a Gemini model.
    Returns the model's text output.

    This is generic and reusable — you can supply any prompt.
    """

    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image at {image_path}: {e}")

    # Package content
    contents = [prompt, img]

    # Configuration
    config = types.GenerateContentConfig(system_instruction=prompt)

    # API call
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

    return response.text.strip()


# ------------------------------------
# 3. YOUR SPECIFIC FUNCTION (OPTIONAL)
# ------------------------------------

def get_fashion_attributes(client, image_path: str):
    """
    Thin wrapper that uses your predefined SYSTEM_PROMPT.
    Keeps your high-level call clean.
    """
    return generate_image_description(
        client,
        image_path=image_path,
        prompt=SYSTEM_PROMPT,
        model=MODEL_NAME
    )

# --- 5. EXECUTION AND DOWNSTREAM PREPARATION ---

if __name__ == "__main__":
    
    natural_text = get_fashion_attributes(IMAGE_PATH, SYSTEM_PROMPT)
    
    print("\n" + "="*50)
    print(f"| ANALYSIS COMPLETE FOR: {IMAGE_PATH}")
    print("="*50 + "\n")

    # The natural language description
    print("\n>>> NATURAL DESCRIPTION")
    print(natural_text)
    