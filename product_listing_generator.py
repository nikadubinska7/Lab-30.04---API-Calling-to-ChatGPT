import os
import json
import base64
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

client = OpenAI(api_key=api_key)


# ==================================================
# IMAGE ENCODING
# ==================================================

def encode_image_to_base64(image_path):
    """
    Convert image file into base64 string for API transmission.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ==================================================
# PROMPT CREATION
# ==================================================

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
    Create a structured prompt for product listing generation.
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

Return only raw valid JSON in this exact structure:

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


# ==================================================
# API CALL
# ==================================================

def generate_product_listing(product):
    """
    Send product image and metadata to OpenAI API.
    Returns parsed JSON product listing.
    """
    encoded_image = encode_image_to_base64(product["image_path"])

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
        model="gpt-5.4-mini",
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
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise ValueError(f"API response was not valid JSON: {response_text}")


# ==================================================
# BATCH PROCESSING
# ==================================================

def process_multiple_products(limit=3):
    """
    Process multiple products and save generated listings.
    """
    products_df = pd.read_csv("products_metadata.csv")

    results = []
    errors = []

    products_to_process = products_df.head(limit)

    print("=" * 60)
    print("BATCH PRODUCT LISTING GENERATION")
    print("=" * 60)
    print(f"Products selected: {len(products_to_process)}")

    for index, product in products_to_process.iterrows():
        product_id = product["id"]
        product_name = product["product_name"]

        print("\n" + "-" * 60)
        print(f"Processing product {index + 1}: {product_name}")
        print("-" * 60)

        try:
            listing = generate_product_listing(product)

            result = {
                "product_id": int(product_id),
                "original_name": product_name,
                "category": product["category"],
                "price": float(product["price"]),
                "image_path": product["image_path"],
                "generated_listing": listing
            }

            results.append(result)

            print("Success.")
            print(f"Generated title: {listing.get('title', 'No title')}")

        except Exception as e:
            error_record = {
                "product_id": int(product_id),
                "product_name": product_name,
                "error": str(e)
            }

            errors.append(error_record)

            print("Failed.")
            print(f"Error: {e}")

        # Small pause to avoid sending requests too quickly
        time.sleep(1)

    output_data = {
        "total_processed": len(products_to_process),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

    with open("generated_product_listings-15.json", "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successful listings: {len(results)}")
    print(f"Failed listings: {len(errors)}")
    print("Results saved to: generated_product_listings-15.json")

    return output_data


# ==================================================
# RUN SCRIPT
# ==================================================

if __name__ == "__main__":
    process_multiple_products(limit=15)