"""
Gemini/Google Merchant Center Feed Transformer
Transforms product feeds to Google Shopping/Gemini format
"""

import csv
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.transformer import (
    parse_json_field,
    extract_description,
    map_availability,
    extract_price,
    get_additional_images,
    detect_product_type,
    build_q_and_a,
    COMPANY_CONFIG
)


# Google Merchant Center product categories
GOOGLE_CATEGORY_MAP = {
    "new_watch": "Apparel & Accessories > Jewelry > Watches",
    "rolex_cpo": "Apparel & Accessories > Jewelry > Watches",
    "preowned_watch": "Apparel & Accessories > Jewelry > Watches",
    "jewelry": "Apparel & Accessories > Jewelry",
    "handbag": "Apparel & Accessories > Handbags, Wallets & Cases > Handbags",
}


def map_google_availability(status: str) -> str:
    """Map availability to Google's enum values."""
    status_map = {
        "IN_STOCK": "in_stock",
        "OUT_OF_STOCK": "out_of_stock",
        "PRE_ORDER": "preorder",
        "PREORDER": "preorder",
        "BACKORDER": "backorder",
    }
    return status_map.get(status.upper(), "out_of_stock")


def map_google_condition(is_preowned: bool) -> str:
    """Map condition to Google's values: new, refurbished, used."""
    return "used" if is_preowned else "new"


def transform_row_to_google(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transform a single row to Google Merchant Center format.

    Google Shopping Required Fields:
    - id, title, description, link, image_link, price, availability, brand

    For products without GTIN:
    - identifier_exists: false
    - brand + mpn required
    """

    # Parse nested JSON fields
    description_data = parse_json_field(row.get("description", ""))
    specs = parse_json_field(row.get("specifications", ""))

    # Skip products that shouldn't be in feed
    if row.get("call_for_price") == "true" or row.get("online") != "1":
        return None

    # Detect product type
    product_type = detect_product_type(row, specs)
    is_preowned = specs.get("isPreOwned", "false").lower() == "true"

    # Extract price
    price, currency = extract_price(row.get("price", ""), row.get("book_price", ""))
    if price <= 0:
        return None

    # Get MPN (required when no GTIN)
    mpn = (specs.get("baseRefNum") or specs.get("referenceNumber") or
           specs.get("reference") or specs.get("modelNumber") or specs.get("sku"))

    # Build Google product object
    product = {
        # Required fields
        "id": row.get("id", ""),
        "title": row.get("title", row.get("name", ""))[:150],
        "description": extract_description(description_data)[:5000],
        "link": row.get("link", ""),
        "image_link": row.get("image_link", ""),
        "availability": map_google_availability(row.get("availability_status", "OUT_OF_STOCK")),
        "price": f"{price:.2f} {currency}",
        "brand": row.get("brand", "")[:70],

        # Condition
        "condition": map_google_condition(is_preowned),

        # Identifier handling - luxury watches typically don't have GTINs
        "identifier_exists": "false",  # No GTIN available
        "mpn": str(mpn)[:70] if mpn else row.get("id", ""),

        # Categories
        "google_product_category": GOOGLE_CATEGORY_MAP.get(product_type, "Apparel & Accessories > Jewelry > Watches"),
        "product_type": row.get("category", ""),

        # Item grouping for variants
        "item_group_id": row.get("item_group_id", "")[:70] if row.get("item_group_id") else "",
    }

    # Additional images (up to 10)
    additional_images = get_additional_images(row.get("additional_image_link", ""))
    if additional_images:
        for i, img_url in enumerate(additional_images[:10]):
            product[f"additional_image_link_{i+1}" if i > 0 else "additional_image_link"] = img_url

    # Optional attributes

    # Color (dial color for watches)
    dial_color = specs.get("dialColor") or specs.get("dial_color") or specs.get("color")
    if dial_color:
        product["color"] = str(dial_color)[:100]

    # Material
    material = specs.get("material") or specs.get("caseMaterial")
    if material:
        product["material"] = str(material)[:200]

    # Size (case size for watches)
    case_size = specs.get("caseSize") or specs.get("caseDiameter") or specs.get("diameter")
    if case_size:
        product["size"] = str(case_size)[:100]

    # Gender
    gender = specs.get("gender") or row.get("gender")
    if gender:
        gender_lower = gender.lower()
        if gender_lower in ["male", "men", "mens"]:
            product["gender"] = "male"
        elif gender_lower in ["female", "women", "womens"]:
            product["gender"] = "female"
        else:
            product["gender"] = "unisex"

    # Age group (default to adult for luxury items)
    product["age_group"] = "adult"

    # Shipping info
    product["shipping_weight"] = "1 lb"  # Default for watches/jewelry

    # Custom labels for segmentation
    product["custom_label_0"] = product_type  # Product type
    product["custom_label_1"] = "luxury"  # Price tier
    if is_preowned:
        product["custom_label_2"] = "pre-owned"
    else:
        product["custom_label_2"] = "new"

    # =================================================================
    # UCP (Universal Commerce Protocol) Attributes
    # For AI-powered checkout via Google AI Mode & Gemini
    # =================================================================

    # Enable UCP checkout
    product["native_commerce"] = "TRUE"

    # Map to checkout API - use same product ID
    product["merchant_item_id"] = row.get("id", "")

    # Consumer notice for luxury items
    if is_preowned:
        product["consumer_notice"] = "Pre-owned luxury item. Authenticity guaranteed. All items inspected and certified by expert watchmakers."
    else:
        product["consumer_notice"] = "Authorized retailer. Factory warranty included. Authenticity guaranteed."

    # Product highlights for AI discovery (up to 10 bullet points)
    highlights = []
    brand = row.get("brand", "")
    if brand:
        highlights.append(f"Authentic {brand} product")
    if is_preowned:
        highlights.append("Certified pre-owned with warranty")
    else:
        highlights.append("Brand new with manufacturer warranty")
    if material:
        highlights.append(f"Crafted in {material}")
    if dial_color:
        highlights.append(f"{dial_color} dial")
    if case_size:
        highlights.append(f"{case_size} case size")

    water_resistance = specs.get("waterResistance") or specs.get("water_resistance")
    if water_resistance:
        highlights.append(f"Water resistant to {water_resistance}")

    movement = specs.get("movement") or specs.get("caliber")
    if movement:
        highlights.append(f"{movement} movement")

    if product_type in ["jewelry"]:
        gemstones = specs.get("gemstones") or specs.get("stones")
        if gemstones:
            highlights.append(f"Features {gemstones}")

    highlights.append("Free shipping available")
    highlights.append("Expert customer service")

    # Join highlights as pipe-separated for Google feed
    if highlights:
        product["product_highlight"] = "|".join(highlights[:10])

    # Structured product details for AI understanding
    product_details = []
    if brand:
        product_details.append(f"Brand:{brand}")
    if mpn:
        product_details.append(f"Reference:{mpn}")
    if material:
        product_details.append(f"Material:{material}")
    if case_size:
        product_details.append(f"Case Size:{case_size}")
    if dial_color:
        product_details.append(f"Dial Color:{dial_color}")
    if water_resistance:
        product_details.append(f"Water Resistance:{water_resistance}")
    if movement:
        product_details.append(f"Movement:{movement}")

    if product_details:
        product["product_detail"] = "|".join(product_details)

    # AI discovery content - reuse Q&A builder from OpenAI transformer
    q_and_a = build_q_and_a(row, specs, product_type)
    if q_and_a:
        product["structured_description"] = q_and_a[:5000]

    # Ads redirect (optional - for tracking)
    # product["ads_redirect"] = row.get("link", "") + "?utm_source=google&utm_medium=shopping"

    # Remove empty values
    product = {k: v for k, v in product.items() if v is not None and v != ""}

    # Store product type for stats
    product["_product_type"] = product_type

    return product


def transform_feed_to_google(input_path: str, output_path: str, output_format: str = "tsv") -> Dict[str, Any]:
    """
    Transform entire feed to Google Merchant Center format.

    Args:
        input_path: Path to input CSV file
        output_path: Path for output file
        output_format: 'tsv' (recommended), 'csv', or 'jsonl'

    Returns:
        Statistics about the transformation
    """
    import os
    print(f"[Gemini] transform_feed called with input_path={input_path}")
    print(f"[Gemini] Output format: {output_format}")

    stats = {
        "total_rows": 0,
        "transformed": 0,
        "skipped": 0,
        "by_product_type": {},
        "errors": [],
    }

    products = []

    # Read and transform all products
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            stats["total_rows"] += 1

            try:
                product = transform_row_to_google(row)

                if product:
                    ptype = product.pop("_product_type", "unknown")
                    stats["by_product_type"][ptype] = stats["by_product_type"].get(ptype, 0) + 1
                    products.append(product)
                    stats["transformed"] += 1
                else:
                    stats["skipped"] += 1

            except Exception as e:
                stats["errors"].append(f"Row {stats['total_rows']}: {str(e)}")
                stats["skipped"] += 1

    # Write output file
    if output_format == "jsonl":
        output_file = output_path + ".jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for product in products:
                f.write(json.dumps(product) + "\n")

    elif output_format in ["tsv", "csv"]:
        delimiter = "\t" if output_format == "tsv" else ","
        output_file = output_path + f".{output_format}"

        if products:
            # Get all unique fields
            all_fields = set()
            for p in products:
                all_fields.update(p.keys())

            # Order fields with required ones first
            required_fields = ["id", "title", "description", "link", "image_link",
                             "availability", "price", "brand", "condition",
                             "identifier_exists", "mpn", "google_product_category"]

            ordered_fields = [f for f in required_fields if f in all_fields]
            ordered_fields += sorted([f for f in all_fields if f not in required_fields])

            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ordered_fields, delimiter=delimiter,
                                       extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                writer.writerows(products)

    stats["output_file"] = output_file

    # Return frontend-compatible format
    return {
        "total_products": stats["total_rows"],
        "successful": stats["transformed"],
        "skipped": stats["skipped"],
        "errors": len(stats["errors"]),
        "error_details": stats["errors"][:10],
        "product_types": stats["by_product_type"],
        "output_file": stats["output_file"],
    }


# Google Merchant Center Feed Specification Reference
GOOGLE_FEED_SPEC = """
GOOGLE MERCHANT CENTER / GEMINI FEED SPECIFICATION
===================================================

REQUIRED FIELDS:
- id: Unique product identifier (max 50 chars)
- title: Product title (max 150 chars)
- description: Product description (max 5000 chars)
- link: Product landing page URL
- image_link: Main product image URL
- availability: in_stock, out_of_stock, preorder, backorder
- price: Price with currency (e.g., "29.99 USD")
- brand: Product brand (max 70 chars)

REQUIRED FOR PRODUCTS WITHOUT GTIN:
- identifier_exists: "false"
- mpn: Manufacturer Part Number (required when no GTIN)

RECOMMENDED FIELDS:
- google_product_category: Google's product taxonomy
- product_type: Your category hierarchy
- condition: new, refurbished, used
- gtin: Global Trade Item Number (UPC/EAN/ISBN)
- item_group_id: Group variants together
- additional_image_link: Up to 10 additional images
- color, size, material, gender, age_group
- shipping, tax
- custom_label_0 through custom_label_4

FILE FORMATS:
- TSV (Tab-separated, recommended)
- CSV (Comma-separated)
- XML
- JSON/JSONL

SUBMISSION:
1. Google Merchant Center: https://merchants.google.com
2. Upload via SFTP, scheduled fetch, or API
3. Feed refreshes: At least every 30 days (daily recommended)
"""


if __name__ == "__main__":
    print(GOOGLE_FEED_SPEC)
