from pathlib import Path
from datasets import load_dataset
import pandas as pd

# Create folder for product images
images_dir = Path("product_images")
images_dir.mkdir(exist_ok=True)

print("=" * 50)
print("LOADING PRODUCT DATASET")
print("=" * 50)

try:
    # Load first 20 products from HuggingFace dataset
    dataset = load_dataset(
        "ashraq/fashion-product-images-small",
        split="train[:20]"
    )

    print(f"Dataset loaded successfully.")
    print(f"Total products loaded: {len(dataset)}")

    # Convert dataset to pandas DataFrame
    products_df = pd.DataFrame(dataset)

    print("\nDataset columns:")
    print(products_df.columns.tolist())

    # Save product images locally
    product_records = []

    for index, product in enumerate(dataset):
        image = product["image"]

        image_path = images_dir / f"product_{index + 1}.jpg"
        image.save(image_path)

        product_records.append({
            "id": index + 1,
            "product_name": product.get("productDisplayName", f"Product {index + 1}"),
            "category": product.get("masterCategory", "Unknown"),
            "subcategory": product.get("subCategory", "Unknown"),
            "article_type": product.get("articleType", "Unknown"),
            "base_colour": product.get("baseColour", "Unknown"),
            "gender": product.get("gender", "Unknown"),
            "season": product.get("season", "Unknown"),
            "usage": product.get("usage", "Unknown"),
            "price": 49.99,
            "image_path": str(image_path)
        })

    # Create clean DataFrame with only useful fields
    clean_products_df = pd.DataFrame(product_records)

    # Save metadata to CSV
    clean_products_df.to_csv("products_metadata.csv", index=False)

    print("\nDataset prepared successfully.")
    print(f"Images saved to: {images_dir}")
    print("Metadata saved to: products_metadata.csv")

    print("\nSample products:")
    print(clean_products_df.head())

except Exception as e:
    print("Could not load the HuggingFace dataset.")
    print(f"Error: {e}")