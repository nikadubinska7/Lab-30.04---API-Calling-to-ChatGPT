import pandas as pd


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

    Parameters:
    - product_name: product name from metadata
    - price: product price
    - category: main product category
    - subcategory: optional product subcategory
    - article_type: optional article/product type
    - base_colour: optional visible product color
    - gender: optional target gender
    - season: optional season
    - usage: optional use case
    - additional_info: optional extra notes

    Returns:
    - formatted prompt string
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


print("=" * 50)
print("TESTING PROMPT CREATION")
print("=" * 50)

# Load product metadata
products_df = pd.read_csv("products_metadata.csv")

# Select first product
product = products_df.iloc[0]

# Create prompt for first product
test_prompt = create_product_listing_prompt(
    product_name=product["product_name"],
    price=product["price"],
    category=product["category"],
    subcategory=product["subcategory"],
    article_type=product["article_type"],
    base_colour=product["base_colour"],
    gender=product["gender"],
    season=product["season"],
    usage=product["usage"],
    additional_info="Use a polished but concise e-commerce tone."
)

print("\nPrompt created successfully.")
print("\nPrompt preview:")
print("-" * 50)
print(test_prompt[:1200])
print("-" * 50)
print(f"\nPrompt length: {len(test_prompt)} characters")