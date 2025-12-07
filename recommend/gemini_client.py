'gemini_client.py'

import os
from google import genai
from google.genai import types
from PIL import Image
import json

# The image file you want to analyze (replace with your file name)
# Using an absolute path (r"...") is good for reliability
IMAGE_PATH = r"C:\Users\grace\code\CIS4810\cis5810final\out\skechers_womens_brown_and_grey_trainers\rgba\00_boots.png"
MODEL_NAME = "gemini-2.5-flash"

# --- 2. THE SYSTEM PROMPT (Your Strict Instructions) ---
DISCRETE_ATTR_PROMPT = """
Initiate Fashion Analysis Protocol. You are designated as an expert Cognitive Fashion Engine, specializing in decomposing garment aesthetics and generating robust compatibility logic.

Your core mission is to process an uploaded clothing item and crystallize its data into two relational vectors and a natural language description. Strict adherence to the 3-part JSON structure and possible prescribed controlled vocabulary is mandatory. For all (DISCRETE) fields, select the singular, most representative term.

Controlled Vocabulary List: 
1.  **COLORS:** [ "Red", "Yellow", "Blue", "Green", "Orange", "Purple", "Black", "White", "Gray", "Navy", "Beige", "Tan", "Brown", "Cream", "Charcoal", "Pink", "Light Blue", "Burgundy", "Teal", "Olive", "Mustard", "Gold", "Silver", "Rose", "Cobalt", "Emerald", "Magenta", "Crimson", "Rust", "Khaki" ]
2.  **FORMALITY:** [1 (Very Casual), 2 (Casual), 3 (Smart Casual), 4 (Business/Cocktail), 5 (Formal/Black Tie)]
3.  **FIT:** ['Slim/Fitted', 'Regular', 'Relaxed/Loose', 'Oversized', 'A-Line', 'Straight', 'Flared/Wide', 'Tapered', 'Boxy', 'Asymmetric']
4.  **TEXTURE:** ['Smooth', 'Knit', 'Structured', 'Fuzzy', 'Sheer', 'Glossy', 'Matte', 'Denim', 'Leather']
5.  **PATTERN:** ['Solid', 'Stripes', 'Plaid', 'Floral', 'Geometric', 'Camouflage', 'Graphic', 'Animal Print']
6.  **SEASON:** ['Summer/Lightweight', 'Transitional/Medium', 'Winter/Heavy', 'All-Season']
7.  **STYLE:** [ "Minimalist", "Classic", "Chic", "Edgy", "Vintage", "Tailored", "Statement", "Maximalist", "Streetwear", "Athleisure", "Casual", "Utilitarian", "Lounge Wear", "Workwear", "Formalwear", "Business Casual", "Quiet Luxury", "Old Money", "Preppy", "Grunge", "Goth", "Bohemian", "Rocker", "Y2K", "Unisex", "Androgynous", "Techwear", "Sustainable", "Resort Wear", "Mod" ]

--

QUERY_VECTOR
Output the item's intrinsic signature as a JSON object. For non-discrete fields, include all relevant descriptors.

Required JSON Fields:

Dominant_Color: (DISCRETE - COLOR)
Color_Value: ['Light', 'Medium', 'Dark']
Pattern_Type: (DISCRETE - PATTERN)
Texture_Primary: (DISCRETE - TEXTURE)
Fit_Silhouette: (DISCRETE - FIT)
Detail_Key: [Critical, visible features, e.g., Notched Lapel, Pleats, Drawstring Hem]
Formality_Level: (DISCRETE - FORMALITY)
Style_Tags: (DISCRETE - STYLE)

--

KEY_VECTOR
Output the core compatibility constraints as a JSON object. Each rule must be a list of acceptable discrete labels for a successful pairing, feel free to output an empty or a list containing all keywords if applicable.

Required JSON Fields:

Rule_Color: (DISCRETE - COLOR)
Rule_Texture: (DISCRETE - TEXTURE)
Rule_Fit: (DISCRETE - FIT)
Rule_Formality: (DISCRETE - FORMALITY)"

--

DESCRIPTION
Output a short natural-language paragraph describing the item's style and vibe using plain English sentences only.
"""
NL_ATTR_PROMPT = """
Initiate Fashion Analysis Protocol. You are designated as an expert Cognitive Fashion Engine, specializing in decomposing garment aesthetics and generating robust compatibility logic.

Your core mission is to process an uploaded clothing item and provide a natural language description for each of its attributes. Strict adherence to the 3-part JSON structure is mandatory.

--

Output the item's intrinsic signature as a JSON object.

Required JSON Fields:

{
"Color_Value_Description": "Describe the dominant color and its lightness/darkness in a complete sentence.",
"Pattern_Texture_Description": "Describe the pattern (or solid status) and primary tactile texture of the fabric in a complete sentence.",
"Fit_Silhouette_Description": "Describe the overall fit and shape of the garment in a complete sentence.",
"Detail_Key_Description": "Describe the critical, visible design features (e.g., notched lapel, pleats) in a complete sentence.",
"Formality_Level_Description": "Describe the garment's precise formality level and appropriate setting in a complete sentence.",
"Style_Tags_Description": "Describe the garment's core aesthetic and vibe based on its style tags in a complete sentence."
}

--

KEY_VECTOR
Output the core compatibility constraints as a JSON object. Each rule must be a natural language description of a piece that would pair well with this one.

Required JSON Fields:

{
"Rule_Color_Pairing": "State the rules for acceptable complementary or contrasting colors in a sentence.",
"Rule_Texture_Pairing": "State the rules for acceptable fabric types and textures to be paired with this item in a sentence.",
"Rule_Fit_Pairing": "State the rules for complementary fits for other items (e.g., balance with a fitted/loose item) in a sentence.",
"Rule_Formality_Alignment": "State the rules for the acceptable formality range of complementary pieces in a sentence."
}

--

DESCRIPTION
Output a short natural-language paragraph describing the item's style and vibe using plain English sentences only.
"""

def init_client():
    """
    Initializes and returns a Gemini client.
    Requires GEMINI_API_KEY to be set as an environment variable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # NOTE: Using the internal API key mechanism for simplicity if available
        # If not, the ValueError will be correctly raised below.
        pass

    try:
        # The genai.Client() constructor automatically looks for the GEMINI_API_KEY environment variable.
        return genai.Client()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Gemini client: {e}. "
            "Please ensure the GEMINI_API_KEY environment variable is set correctly."
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
    """

    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image at {image_path}: {e}")
    
    contents = [img] 

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

def get_fashion_attributes(client, image_path: str, attribute_type='discrete'):
    """
    Thin wrapper that uses your predefined SYSTEM_PROMPT.
    Keeps your high-level call clean.
    """

    if attribute_type == 'discrete':
        prompt = DISCRETE_ATTR_PROMPT
    elif attribute_type == 'nl': 
        prompt = NL_ATTR_PROMPT

    output = generate_image_description(
        client,
        image_path=image_path,
        prompt=prompt,
        model=MODEL_NAME
    )
    print('received response')

    output = output.strip()
    
    if output.startswith("```json"):
        output = output[7:-3]

    data_dict = json.loads(output)

    return data_dict

# --- 5. EXECUTION AND DOWNSTREAM PREPARATION ---

if __name__ == "__main__":
    
    try:
        client = init_client()
        
        natural_text = get_fashion_attributes(client, IMAGE_PATH)
        
        print("\n" + "="*50)
        print(f"| ANALYSIS COMPLETE FOR: {IMAGE_PATH}")
        print("="*50 + "\n")

        print("\n>>> RETURNED OUTPUT")
        print(natural_text)
        # print(natural_text["QUERY_VECTOR"])
        # print(natural_text['KEY_VECTOR'])
        # print(natural_text['DESCRIPTION'])

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # Catch and print relevant errors (like missing API key or missing file)
        print(f"\nFATAL ERROR: {e}")