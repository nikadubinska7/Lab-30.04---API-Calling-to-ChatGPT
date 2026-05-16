import os
import json
import base64
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Setup
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

client = OpenAI(api_key=api_key)


# -----------------------------
# Image encoding function
# -----------------------------

def encode_image_to_base64(image_path):
    """
    Convert an image file into a base64 string.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_image


# -----------------------------
# Prompt creation function
# -----------------------------

def create_product_listing_prompt(
    product_name,
    price,
    category,
    subcategory=None,
    article_type=None,
    base_colour=None,
    gender=None,
    season=None,
    usage=None,
    additional_info=None
):
    """
    Create a detailed prompt for generating an e-commerce product listing.
    """

    prompt = f"""
You are an expert e-commerce copywriter.

Analyze the product image and create a professional product listing using both the image and the metadata below.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
- Subcategory: {subcategory}
- Article Type: {article_type}
- Base Colour: {base_colour}
- Gender: {gender}
- Season: {season}
- Usage: {usage}
{f"- Additional Info: {additional_info}" if additional_info else ""}

Create a product listing with:

1. Product Title
- Catchy and SEO-friendly
- Maximum 60 characters

2. Product Description
- 150 to 200 words
- Mention visible details from the image
- Highlight benefits, style, design, use case, and value
- Use clear persuasive language

3. Key Features
- 5 to 7 bullet points
- Specific and useful for an online shopper

4. SEO Keywords
- 10 to 15 relevant keywords
- Comma-separated

Return only valid JSON in this exact structure:

{{
    "title": "Product title here",
    "description": "Full product description here",
    "features": [
        "Feature 1",
        "Feature 2",
        "Feature 3"
    ],
    "keywords": "keyword1, keyword2, keyword3"
}}

Do not include markdown.
Do not include explanations outside the JSON.
Be specific about what is visible in the image, but do not invent details that cannot be reasonably inferred.
"""

    return prompt.strip()


# -----------------------------
# API call function
# -----------------------------

def generate_product_listing(product):
    """
    Send product metadata and image to OpenAI and return a parsed JSON listing.
    """

    image_path = product["image_path"]
    encoded_image = encode_image_to_base64(image_path)

    prompt = create_product_listing_prompt(
        product_name=product["product_name"],
        price=float(product["price"]),
        category=product["category"],
        subcategory=product["subcategory"],
        article_type=product["article_type"],
        base_colour=product["base_colour"],
        gender=product["gender"],
        season=product["season"],
        usage=product["usage"],
        additional_info="Use a polished but concise e-commerce tone."
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                ]
            }
        ],
        temperature=0.7
    )

    response_text = response.output_text

    try:
        listing = json.loads(response_text)
        return listing

    except json.JSONDecodeError:
        print("The API response was not valid JSON.")
        print("Raw response:")
        print(response_text)
        return None


# -----------------------------
# Test with one product
# -----------------------------

print("=" * 50)
print("TESTING OPENAI VISION API CALL")
print("=" * 50)

products_df = pd.read_csv("products_metadata.csv")

product = products_df.iloc[0]

print(f"Processing product: {product['product_name']}")
print(f"Image path: {product['image_path']}")

try:
    listing = generate_product_listing(product)

    if listing:
        print("\nAPI call successful.")
        print("\nGenerated Product Listing:")
        print(json.dumps(listing, indent=4))

except Exception as e:
    print("\nAPI call failed.")
    print(f"Error: {e}")