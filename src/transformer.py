"""
LLM Feed Optimizer - Product Feed Transformer
Transforms 1916 Company product feeds to OpenAI Commerce format
"""

import csv
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Company Configuration
COMPANY_CONFIG = {
    "store_name": "The 1916 Company",
    "seller_url": "https://www.the1916company.com",
    "store_country": "US",
    "target_countries": ["US"],
    "seller_privacy_policy": "https://www.the1916company.com/privacy/",
    "seller_tos": "https://www.the1916company.com/terms-conditions/",
    "return_policy_url": "https://www.the1916company.com/returns-exchanges/",
    "return_policy_text": "To be eligible for a return, your item(s) must be unused, in their original packaging, and in the same condition that you received it.",
}

# Return window rules by product type
RETURN_WINDOWS = {
    "new_watch": 14,
    "rolex_cpo": 14,
    "jewelry": 14,
    "handbag": 14,
    "preowned_watch": 7,  # Non-Rolex pre-owned watches
}


def detect_product_type(row: Dict[str, Any], specs: Dict[str, Any]) -> str:
    """
    Detect product type from row data and specifications.
    Returns: new_watch, rolex_cpo, preowned_watch, jewelry, handbag
    """
    product_id = row.get("id", "").lower()
    category = row.get("category", "").lower()
    brand = row.get("brand", "").lower()
    is_preowned = specs.get("isPreOwned", "false").lower() == "true"

    # Check for jewelry
    if "jewelry" in category or product_id.startswith("ns-j-"):
        return "jewelry"

    # Check for handbags
    if "handbag" in category or "bag" in category:
        return "handbag"

    # Check for Rolex CPO
    if brand == "rolex" and is_preowned:
        return "rolex_cpo"

    # Check for pre-owned watches (non-Rolex)
    if is_preowned:
        return "preowned_watch"

    # Default to new watch
    return "new_watch"


def parse_json_field(value: str) -> Any:
    """Safely parse a JSON field, returning empty dict/list on failure."""
    if not value or value == "":
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_description(description_json: Dict) -> str:
    """Extract plain text description from JSON structure."""
    if isinstance(description_json, dict):
        # Prefer long_description, fall back to short_description
        desc = description_json.get("long_description") or description_json.get("short_description") or ""
    elif isinstance(description_json, str):
        desc = description_json
    else:
        desc = ""

    # Strip HTML tags if present (basic)
    import re
    desc = re.sub(r'<[^>]+>', '', desc)

    # Limit to 5000 chars per OpenAI spec
    return desc[:5000].strip()


def map_availability(status: str) -> str:
    """Map availability status to OpenAI enum values."""
    status_map = {
        "IN_STOCK": "in_stock",
        "OUT_OF_STOCK": "out_of_stock",
        "PRE_ORDER": "pre_order",
        "BACKORDER": "backorder",
        "PREORDER": "pre_order",
    }
    return status_map.get(status.upper(), "unknown")


def extract_price(price_value: str, book_price: str) -> tuple[float, str]:
    """Extract price and currency. Returns (price, currency_code)."""
    # Try to get USD price from book_price JSON first
    if book_price:
        try:
            prices = json.loads(book_price)
            for price_obj in prices:
                if "ns-company-list-usd" in price_obj:
                    usd_price = price_obj["ns-company-list-usd"]
                    if usd_price and usd_price > 0:
                        return (float(usd_price), "USD")
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to main price field
    try:
        return (float(price_value), "USD")
    except (ValueError, TypeError):
        return (0.0, "USD")


def get_additional_images(additional_images_json: str) -> List[str]:
    """Extract additional image URLs from JSON."""
    images = parse_json_field(additional_images_json)
    if isinstance(images, list):
        return [img.get("url") for img in images if img.get("url")]
    return []


def build_q_and_a(row: Dict[str, Any], specs: Dict[str, Any], product_type: str) -> str:
    """
    Generate Q&A content to help LLMs understand and recommend the product.
    This is one of the most valuable fields for LLM reasoning.
    """
    qa_pairs = []
    brand = row.get("brand", "").strip()
    is_preowned = specs.get("isPreOwned", "false").lower() == "true"

    # Condition questions
    if is_preowned:
        if brand.lower() == "rolex":
            qa_pairs.append("Q: Is this watch certified?\nA: Yes, this is a Rolex Certified Pre-Owned watch with a 2-year international guarantee from Rolex.")
        else:
            qa_pairs.append("Q: What is the condition?\nA: This is a pre-owned watch that has been inspected and authenticated by our expert watchmakers.")

    # Water resistance
    water_resistance = specs.get("waterResistance") or specs.get("water_resistance")
    if water_resistance:
        qa_pairs.append(f"Q: Is this watch waterproof?\nA: This watch has a water resistance rating of {water_resistance}.")

    # Movement/power reserve
    movement = specs.get("movementType") or specs.get("movement") or specs.get("caliber")
    if movement:
        power_reserve = specs.get("powerReserve") or specs.get("power_reserve")
        if power_reserve:
            qa_pairs.append(f"Q: What type of movement does it have?\nA: {movement} with approximately {power_reserve} power reserve.")
        else:
            qa_pairs.append(f"Q: What type of movement does it have?\nA: {movement}.")

    # Box and papers
    has_box = specs.get("hasBox", "").lower() == "true" or specs.get("box", "").lower() == "true"
    has_papers = specs.get("hasPapers", "").lower() == "true" or specs.get("papers", "").lower() == "true"
    if has_box or has_papers:
        if has_box and has_papers:
            qa_pairs.append("Q: Does it come with box and papers?\nA: Yes, includes original box and papers/documentation.")
        elif has_box:
            qa_pairs.append("Q: Does it come with the original box?\nA: Yes, includes original box.")
        elif has_papers:
            qa_pairs.append("Q: Does it come with papers?\nA: Yes, includes original papers/documentation.")

    # Warranty
    warranty = specs.get("warranty") or specs.get("warrantyYears")
    if warranty:
        qa_pairs.append(f"Q: Is there a warranty?\nA: Yes, this watch comes with a {warranty} warranty.")
    elif product_type == "rolex_cpo":
        qa_pairs.append("Q: Is there a warranty?\nA: Yes, Rolex Certified Pre-Owned watches come with a 2-year international guarantee.")

    # Case size
    case_size = specs.get("caseSize") or specs.get("caseDiameter") or specs.get("diameter")
    if case_size:
        qa_pairs.append(f"Q: What size is the watch?\nA: The case diameter is {case_size}.")

    return "\n\n".join(qa_pairs) if qa_pairs else ""


def build_product_category(row: Dict[str, Any], specs: Dict[str, Any], product_type: str) -> str:
    """Build a proper category taxonomy for LLM understanding."""
    brand = row.get("brand", "").strip()
    category = row.get("category", "")

    # Build hierarchy based on product type
    if product_type in ["new_watch", "rolex_cpo", "preowned_watch"]:
        base = "Watches"
        if product_type == "rolex_cpo":
            return f"{base} > Certified Pre-Owned > Rolex"
        elif product_type == "preowned_watch":
            return f"{base} > Pre-Owned > {brand}" if brand else f"{base} > Pre-Owned"
        else:
            return f"{base} > Luxury Watches > {brand}" if brand else f"{base} > Luxury Watches"
    elif product_type == "jewelry":
        return f"Jewelry > {category}" if category else "Jewelry"
    elif product_type == "handbag":
        return f"Handbags > Designer > {brand}" if brand else "Handbags > Designer"

    return category


def extract_material(specs: Dict[str, Any]) -> str:
    """Extract material information from specs."""
    materials = []

    case_material = specs.get("caseMaterial") or specs.get("case_material")
    if case_material:
        materials.append(case_material)

    bezel_material = specs.get("bezelMaterial") or specs.get("bezel_material")
    if bezel_material and bezel_material != case_material:
        materials.append(f"{bezel_material} bezel")

    band_material = specs.get("bandMaterial") or specs.get("braceletMaterial") or specs.get("strap_material")
    if band_material and band_material not in materials:
        materials.append(f"{band_material} bracelet")

    return ", ".join(materials)[:100] if materials else ""


def transform_row_to_openai(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transform a single row to OpenAI format with enhanced LLM attributes."""

    # Parse nested JSON fields
    description_data = parse_json_field(row.get("description", ""))
    specs = parse_json_field(row.get("specifications", ""))

    # Skip products that shouldn't be in feed
    if row.get("call_for_price") == "true" or row.get("online") != "1":
        return None

    # Detect product type for return policy
    product_type = detect_product_type(row, specs)
    return_window = RETURN_WINDOWS.get(product_type, 14)

    # Extract price
    price, currency = extract_price(row.get("price", ""), row.get("book_price", ""))
    if price <= 0:
        return None  # Skip products without valid price

    # Build OpenAI product object
    product = {
        # Required fields
        "item_id": row.get("id", ""),
        "title": row.get("title", row.get("name", ""))[:150],  # Max 150 chars
        "description": extract_description(description_data),
        "brand": row.get("brand", "")[:70],  # Max 70 chars
        "url": row.get("link", ""),
        "image_url": row.get("image_link", ""),
        "price": price,
        "currency": currency,
        "availability": map_availability(row.get("availability_status", "unknown")),

        # OpenAI flags
        "is_eligible_search": True,
        "is_eligible_checkout": row.get("allow_buy_now", "").lower() == "true",

        # Variant grouping
        "group_id": row.get("item_group_id", "")[:70] if row.get("item_group_id") else None,
        "listing_has_variations": bool(row.get("item_group_id")),

        # Store info
        "store_name": COMPANY_CONFIG["store_name"],
        "seller_url": COMPANY_CONFIG["seller_url"],
        "store_country": COMPANY_CONFIG["store_country"],
        "target_countries": COMPANY_CONFIG["target_countries"],

        # Policies (required for checkout)
        "seller_privacy_policy": COMPANY_CONFIG["seller_privacy_policy"],
        "seller_tos": COMPANY_CONFIG["seller_tos"],
        "return_policy": COMPANY_CONFIG["return_policy_text"],
        "return_window": return_window,

        # Recommended fields
        "condition": "used" if specs.get("isPreOwned", "false").lower() == "true" else "new",
        "product_category": build_product_category(row, specs, product_type),
    }

    # Add additional images if available
    additional_images = get_additional_images(row.get("additional_image_link", ""))
    if additional_images:
        product["additional_image_urls"] = ",".join(additional_images[:10])

    # === LLM Enhancement Fields ===

    # Q&A content - HIGHLY valuable for LLM reasoning
    q_and_a = build_q_and_a(row, specs, product_type)
    if q_and_a:
        product["q_and_a"] = q_and_a

    # Material extraction
    material = extract_material(specs)
    if material:
        product["material"] = material

    # MPN (Manufacturer Part Number) - use baseRefNum or reference number for watches
    mpn = specs.get("baseRefNum") or specs.get("referenceNumber") or specs.get("reference") or specs.get("modelNumber") or specs.get("sku")
    if mpn:
        product["mpn"] = str(mpn)[:70]

    # Dimensions for watches
    case_size = specs.get("caseSize") or specs.get("caseDiameter") or specs.get("diameter")
    if case_size:
        product["dimensions"] = f"{case_size} diameter"
        product["size"] = str(case_size)[:20]

    case_thickness = specs.get("caseThickness") or specs.get("thickness")
    if case_thickness and case_size:
        product["dimensions"] = f"{case_size} x {case_thickness}"

    # Water resistance as a feature
    water_resistance = specs.get("waterResistance") or specs.get("water_resistance")
    if water_resistance:
        if "material" in product:
            product["material"] += f", {water_resistance} water resistant"
        else:
            product["material"] = f"{water_resistance} water resistant"

    # Color (dial color for watches)
    dial_color = specs.get("dialColor") or specs.get("dial_color") or specs.get("color")
    if dial_color:
        product["color"] = str(dial_color)[:40]

    # Gender targeting
    gender = specs.get("gender") or row.get("gender")
    if gender:
        product["gender"] = gender.lower()

    # Year of production (useful context)
    year = specs.get("year") or specs.get("productionYear")
    if year:
        # Add to description or as custom field
        if product.get("description"):
            product["description"] = f"{product['description']}\n\nYear: {year}"

    # GTIN if available (UPC/EAN)
    gtin = row.get("gtin") or row.get("upc") or row.get("ean")
    if gtin:
        product["gtin"] = str(gtin)

    # Add product type as custom attribute (useful for filtering)
    product["_product_type"] = product_type

    # Remove None values and empty strings
    product = {k: v for k, v in product.items() if v is not None and v != ""}

    return product


def transform_feed(input_path: str, output_path: str, compress: bool = True) -> Dict[str, Any]:
    """
    Transform entire feed from CSV to OpenAI JSONL format.

    Args:
        input_path: Path to input CSV file
        output_path: Path for output JSONL file
        compress: Whether to gzip the output (required by OpenAI)

    Returns:
        Statistics about the transformation
    """
    import os
    print(f"[DEBUG] transform_feed called with input_path={input_path}")
    print(f"[DEBUG] File exists: {os.path.exists(input_path)}")
    if os.path.exists(input_path):
        print(f"[DEBUG] File size: {os.path.getsize(input_path)} bytes")

    stats = {
        "total_rows": 0,
        "transformed": 0,
        "skipped": 0,
        "by_product_type": {},
        "errors": [],
    }

    output_file = output_path + (".gz" if compress else "")

    # Open output file (gzipped or plain)
    if compress:
        out_handle = gzip.open(output_file, 'wt', encoding='utf-8')
    else:
        out_handle = open(output_file, 'w', encoding='utf-8')

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            print(f"[DEBUG] CSV fieldnames: {reader.fieldnames}")

            for row in reader:
                stats["total_rows"] += 1

                try:
                    product = transform_row_to_openai(row)

                    if product:
                        # Track product type stats
                        ptype = product.pop("_product_type", "unknown")
                        stats["by_product_type"][ptype] = stats["by_product_type"].get(ptype, 0) + 1

                        # Write JSONL line
                        out_handle.write(json.dumps(product) + "\n")
                        stats["transformed"] += 1
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    stats["errors"].append(f"Row {stats['total_rows']}: {str(e)}")
                    stats["skipped"] += 1

    finally:
        out_handle.close()

    stats["output_file"] = output_file

    # Map to frontend-expected keys
    return {
        "total_products": stats["total_rows"],
        "successful": stats["transformed"],
        "skipped": stats["skipped"],
        "errors": len(stats["errors"]),
        "error_details": stats["errors"][:10],  # First 10 errors for debugging
        "product_types": stats["by_product_type"],
        "output_file": stats["output_file"],
    }


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Transform product feed to OpenAI format")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output JSONL file path", default="output/openai_feed.jsonl")
    parser.add_argument("--no-compress", action="store_true", help="Don't gzip the output")

    args = parser.parse_args()

    print(f"Transforming {args.input}...")
    stats = transform_feed(args.input, args.output, compress=not args.no_compress)

    print(f"\nTransformation complete!")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Transformed: {stats['transformed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"\n  By product type:")
    for ptype, count in stats["by_product_type"].items():
        print(f"    {ptype}: {count}")
    print(f"\n  Output: {stats['output_file']}")

    if stats["errors"]:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"    - {err}")
        if len(stats["errors"]) > 5:
            print(f"    ... and {len(stats['errors']) - 5} more")


if __name__ == "__main__":
    main()
