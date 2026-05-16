import os
import json
import base64
import time
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


# ==================================================
# CONSTANTS
# ==================================================

MODEL_NAME = "gpt-5.4-mini"
DEFAULT_INPUT_CSV = "products_metadata.csv"
DEFAULT_OUTPUT_JSON = "generated_product_listings_refactored.json"
DEFAULT_REQUEST_DELAY = 1
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2

REQUIRED_PRODUCT_FIELDS = [
    "id",
    "product_name",
    "price",
    "category",
    "subcategory",
    "article_type",
    "base_colour",
    "gender",
    "season",
    "usage",
    "image_path",
]


# ==================================================
# LOGGING SETUP
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("product_generator.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ==================================================
# SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")


# ==================================================
# API WRAPPER
# ==================================================

class OpenAIWrapper:
    """
    Wrapper for OpenAI API calls with error handling and retry logic.
    """

    def __init__(
        self,
        api_key,
        model_name=MODEL_NAME,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_delay=DEFAULT_RETRY_DELAY,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate_vision_response(self, prompt, encoded_image):
        """
        Send prompt and encoded image to OpenAI API with retry logic.
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Calling OpenAI API with model {self.model_name}, "
                    f"attempt {attempt} of {self.max_retries}"
                )

                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                                },
                            ],
                        }
                    ],
                    temperature=0.7,
                )

                logger.info("OpenAI API call completed successfully.")
                return response.output_text

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                wait_time = self.retry_delay * attempt

                logger.warning(
                    f"Retryable API error on attempt {attempt}: "
                    f"{type(e).__name__} - {e}"
                )

                print(
                    f"WARNING in OpenAIWrapper.generate_vision_response(): {type(e).__name__}\n"
                    f"  Location: API attempt {attempt} of {self.max_retries}\n"
                    f"  Message: {e}\n"
                    f"  Action: Retrying after {wait_time} seconds."
                )

                time.sleep(wait_time)

            except APIError as e:
                logger.error(f"OpenAI API error: {type(e).__name__} - {e}")

                print(
                    f"ERROR in OpenAIWrapper.generate_vision_response(): APIError\n"
                    f"  Location: OpenAI API request\n"
                    f"  Message: {e}\n"
                    f"  Suggestion: Check model name, request format, and OpenAI account status."
                )
                raise

            except Exception as e:
                logger.error(f"OpenAI API error: {type(e).__name__} - {e}")

                print(
                    f"ERROR in OpenAIWrapper.generate_vision_response(): {type(e).__name__}\n"
                    f"  Location: OpenAI API request\n"
                    f"  Message: {e}\n"
                    f"  Suggestion: Check API key, model name, internet connection, and request payload."
                )
                raise

        raise RuntimeError(
            f"OpenAI API request failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )


api_wrapper = OpenAIWrapper(api_key=api_key)


# ==================================================
# FILE LOADING HELPERS
# ==================================================

def load_products_csv(csv_path):
    """
    Load product metadata from a CSV file.
    """
    try:
        csv_path = Path(csv_path)
        logger.info(f"Loading products CSV from: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        products_df = pd.read_csv(csv_path)

        if products_df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")

        logger.info(f"Loaded {len(products_df)} products from CSV.")
        return products_df

    except FileNotFoundError as e:
        logger.error(f"Failed to load products CSV: {e}")

        print(
            f"ERROR in load_products_csv(): FileNotFoundError\n"
            f"  Location: {csv_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check that the CSV file exists in the project folder."
        )
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"Failed to load products CSV: {e}")

        print(
            f"ERROR in load_products_csv(): EmptyDataError\n"
            f"  Location: {csv_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check that the CSV file contains product data."
        )
        raise

    except Exception as e:
        logger.error(f"Failed to load products CSV: {e}")

        print(
            f"ERROR in load_products_csv(): {type(e).__name__}\n"
            f"  Location: {csv_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check the CSV file format and path."
        )
        raise


def save_results_to_json(output_data, output_path):
    """
    Save generated product listing results to a JSON file.
    """
    try:
        logger.info(f"Saving results to JSON file: {output_path}")

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4, ensure_ascii=False)

        logger.info(f"Results saved successfully to: {output_path}")

    except PermissionError as e:
        logger.error(f"Failed to save results to JSON: {e}")

        print(
            f"ERROR in save_results_to_json(): PermissionError\n"
            f"  Location: {output_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check file permissions or close the file if it is open elsewhere."
        )
        raise

    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}")

        print(
            f"ERROR in save_results_to_json(): {type(e).__name__}\n"
            f"  Location: {output_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check that the output path is valid."
        )
        raise


# ==================================================
# VALIDATION HELPERS
# ==================================================

def validate_product_row(product):
    """
    Validate that a product row contains all required fields.
    """
    logger.info(
        f"Validating product row: {product.get('product_name', 'Unknown product')}"
    )

    missing_fields = []

    for field in REQUIRED_PRODUCT_FIELDS:
        if field not in product or pd.isna(product[field]):
            missing_fields.append(field)

    if missing_fields:
        logger.error(
            f"Validation failed for product ID {product.get('id', 'Unknown')}: "
            f"missing fields {missing_fields}"
        )

        raise ValueError(
            f"Product data is missing required fields: {missing_fields}. "
            f"Product ID: {product.get('id', 'Unknown')}"
        )

    try:
        float(product["price"])
    except ValueError:
        logger.error(
            f"Validation failed for product ID {product.get('id', 'Unknown')}: "
            f"invalid price {product['price']}"
        )

        raise ValueError(
            f"Invalid price value for product ID {product.get('id', 'Unknown')}: "
            f"{product['price']}"
        )

    logger.info(f"Validation passed for product ID: {product.get('id', 'Unknown')}")
    return True


# ==================================================
# IMAGE ENCODING
# ==================================================

def encode_image_to_base64(image_path):
    """
    Convert image file into base64 string for API transmission.
    """
    try:
        image_path = Path(image_path)
        logger.info(f"Encoding image to base64: {image_path}")

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        logger.info(f"Image encoded successfully: {image_path}")
        return encoded_image

    except FileNotFoundError as e:
        logger.error(f"Failed to encode image: {e}")

        print(
            f"ERROR in encode_image_to_base64(): FileNotFoundError\n"
            f"  Location: {image_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check that the image path in the CSV is correct."
        )
        raise

    except Exception as e:
        logger.error(f"Failed to encode image: {e}")

        print(
            f"ERROR in encode_image_to_base64(): {type(e).__name__}\n"
            f"  Location: {image_path}\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check that the image file can be opened and read."
        )
        raise


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
    additional_info=None,
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


def create_prompt_from_product(product):
    """
    Create a product listing prompt from one product row.
    """
    logger.info(
        f"Creating prompt for product: {product.get('product_name', 'Unknown product')}"
    )

    return create_product_listing_prompt(
        product_name=product["product_name"],
        price=float(product["price"]),
        category=product["category"],
        subcategory=product["subcategory"],
        article_type=product["article_type"],
        base_colour=product["base_colour"],
        gender=product["gender"],
        season=product["season"],
        usage=product["usage"],
        additional_info="Use a polished but concise e-commerce tone.",
    )


# ==================================================
# API HELPERS
# ==================================================

def call_openai_vision_api(prompt, encoded_image):
    """
    Send prompt and encoded image to OpenAI API through the API wrapper.
    """
    return api_wrapper.generate_vision_response(prompt, encoded_image)


def parse_api_json_response(response_text):
    """
    Parse the API response text into JSON.
    """
    try:
        logger.info("Parsing API response as JSON.")
        parsed_response = json.loads(response_text)
        logger.info("API response parsed successfully.")
        return parsed_response

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse API response as JSON: {e}")

        print(
            f"ERROR in parse_api_json_response(): JSONDecodeError\n"
            f"  Location: API response parsing\n"
            f"  Message: {e}\n"
            f"  Suggestion: Check whether the model returned raw valid JSON only."
        )
        raise ValueError(f"API response was not valid JSON: {response_text}")


# ==================================================
# RESULT HELPERS
# ==================================================

def build_success_record(product, listing):
    """
    Build a structured success record for one product.
    """
    return {
        "product_id": int(product["id"]),
        "original_name": product["product_name"],
        "category": product["category"],
        "price": float(product["price"]),
        "image_path": product["image_path"],
        "generated_listing": listing,
    }


def build_error_record(product, error):
    """
    Build a structured error record for one product.
    """
    return {
        "product_id": int(product.get("id", -1)),
        "product_name": product.get("product_name", "Unknown product"),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def build_output_data(products_to_process, results, errors):
    """
    Build the final output dictionary.
    """
    return {
        "total_processed": len(products_to_process),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


# ==================================================
# DISPLAY HELPERS
# ==================================================

def print_batch_start(products_to_process):
    """
    Print batch processing start message.
    """
    print("=" * 60)
    print("BATCH PRODUCT LISTING GENERATION")
    print("=" * 60)
    print(f"Products selected: {len(products_to_process)}")


def print_product_start(index, product_name):
    """
    Print product processing start message.
    """
    print("\n" + "-" * 60)
    print(f"Processing product {index + 1}: {product_name}")
    print("-" * 60)


def print_batch_summary(results, errors, output_json):
    """
    Print batch processing summary.
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successful listings: {len(results)}")
    print(f"Failed listings: {len(errors)}")
    print(f"Results saved to: {output_json}")


# ==================================================
# PRODUCT LISTING GENERATION
# ==================================================

def generate_product_listing(product):
    """
    Generate a product listing for one product.
    """
    product_name = product.get("product_name", "Unknown product")
    logger.info(f"Generating listing for product: {product_name}")

    validate_product_row(product)

    encoded_image = encode_image_to_base64(product["image_path"])
    prompt = create_prompt_from_product(product)
    response_text = call_openai_vision_api(prompt, encoded_image)
    listing = parse_api_json_response(response_text)

    logger.info(f"Listing generated for product: {product_name}")
    return listing


# ==================================================
# BATCH PROCESSING
# ==================================================

def process_multiple_products(
    input_csv=DEFAULT_INPUT_CSV,
    output_json=DEFAULT_OUTPUT_JSON,
    limit=3,
    request_delay=DEFAULT_REQUEST_DELAY,
):
    """
    Process multiple products and save generated listings.
    """
    products_df = load_products_csv(input_csv)
    logger.info(f"Starting batch processing. Limit: {limit}")

    results = []
    errors = []

    products_to_process = products_df.head(limit)

    print_batch_start(products_to_process)

    for index, product in products_to_process.iterrows():
        product_name = product.get("product_name", "Unknown product")

        print_product_start(index, product_name)
        logger.info(f"Processing product row index {index}: {product_name}")

        try:
            listing = generate_product_listing(product)
            result = build_success_record(product, listing)

            results.append(result)

            logger.info(
                f"Successfully processed product row index {index}: {product_name}"
            )

            print("Success.")
            print(f"Generated title: {listing.get('title', 'No title')}")

        except Exception as e:
            error_record = build_error_record(product, e)
            errors.append(error_record)

            logger.error(
                f"Failed to process product row index {index}: "
                f"{product_name}. Error: {type(e).__name__} - {e}"
            )

            print("Failed.")
            print(
                f"ERROR in process_multiple_products(): {type(e).__name__}\n"
                f"  Location: Product row index {index}, product name: {product_name}\n"
                f"  Message: {e}\n"
                f"  Suggestion: Check this product row, image path, API response, or validation rules."
            )

        time.sleep(request_delay)

    output_data = build_output_data(products_to_process, results, errors)
    save_results_to_json(output_data, output_json)

    logger.info(
        f"Batch processing complete. Successful: {len(results)}, Failed: {len(errors)}"
    )

    print_batch_summary(results, errors, output_json)

    return output_data


# ==================================================
# RUN SCRIPT
# ==================================================

if __name__ == "__main__":
    process_multiple_products(limit=15)