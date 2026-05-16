import base64
from pathlib import Path
import pandas as pd


def encode_image_to_base64(image_path):
    """
    Convert an image file into a base64 string.

    Parameters:
    - image_path: path to the image file

    Returns:
    - base64 encoded image string
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_image


print("=" * 50)
print("TESTING IMAGE ENCODING")
print("=" * 50)

# Load product metadata
products_df = pd.read_csv("products_metadata.csv")

# Select the first product
first_product = products_df.iloc[0]
image_path = first_product["image_path"]

print(f"Product name: {first_product['product_name']}")
print(f"Image path: {image_path}")

try:
    encoded_image = encode_image_to_base64(image_path)

    print("\nImage encoded successfully.")
    print(f"Encoded image type: {type(encoded_image)}")
    print(f"Encoded image length: {len(encoded_image)} characters")
    print(f"Encoded preview: {encoded_image[:100]}...")

except Exception as e:
    print("Image encoding failed.")
    print(f"Error: {e}")