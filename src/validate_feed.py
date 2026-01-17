"""
OpenAI Commerce Feed Validator
Validates product feed against OpenAI Commerce Feed specification
"""

import gzip
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse
from collections import defaultdict

# =============================================================================
# OpenAI Commerce Feed Schema Definition
# =============================================================================

OPENAI_COMMERCE_SCHEMA = {
    # Required fields
    "required": {
        "item_id": {"type": "string", "max_length": 50, "description": "Unique product identifier"},
        "title": {"type": "string", "max_length": 150, "description": "Product title"},
        "description": {"type": "string", "max_length": 5000, "description": "Product description"},
        "url": {"type": "url", "description": "Product page URL"},
        "image_url": {"type": "url", "description": "Main product image URL"},
        "price": {"type": "number", "min": 0, "description": "Product price"},
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "CNY", "HKD", "SGD", "AED"], "description": "Currency code"},
        "availability": {"type": "string", "enum": ["in_stock", "out_of_stock", "pre_order", "backorder", "unknown"], "description": "Stock status"},
    },

    # Recommended fields
    "recommended": {
        "brand": {"type": "string", "max_length": 70, "description": "Product brand"},
        "condition": {"type": "string", "enum": ["new", "used", "refurbished"], "description": "Product condition"},
        "store_name": {"type": "string", "max_length": 100, "description": "Store/seller name"},
        "seller_url": {"type": "url", "description": "Seller website URL"},
        "target_countries": {"type": "array", "items": "string", "description": "Target country codes"},
        "store_country": {"type": "string", "max_length": 2, "description": "Store country code"},
        "product_category": {"type": "string", "max_length": 750, "description": "Product category taxonomy"},
        "group_id": {"type": "string", "max_length": 70, "description": "Variant group ID"},
        "listing_has_variations": {"type": "boolean", "description": "Whether product has variants"},
    },

    # Policy fields (required for checkout eligibility)
    "policy": {
        "seller_privacy_policy": {"type": "url", "description": "Privacy policy URL"},
        "seller_tos": {"type": "url", "description": "Terms of service URL"},
        "return_policy": {"type": "string", "max_length": 5000, "description": "Return policy text"},
        "return_window": {"type": "integer", "min": 0, "max": 365, "description": "Return window in days"},
    },

    # LLM Enhancement fields
    "llm_enhancement": {
        "q_and_a": {"type": "string", "max_length": 10000, "description": "Q&A content for LLM reasoning"},
        "material": {"type": "string", "max_length": 200, "description": "Product material"},
        "color": {"type": "string", "max_length": 40, "description": "Product color"},
        "size": {"type": "string", "max_length": 20, "description": "Product size"},
        "dimensions": {"type": "string", "max_length": 100, "description": "Product dimensions"},
        "mpn": {"type": "string", "max_length": 70, "description": "Manufacturer part number"},
        "gtin": {"type": "string", "max_length": 50, "description": "Global trade item number"},
        "gender": {"type": "string", "enum": ["male", "female", "unisex"], "description": "Target gender"},
    },

    # Additional fields
    "optional": {
        "additional_image_urls": {"type": "string", "description": "Comma-separated additional image URLs"},
        "is_eligible_search": {"type": "boolean", "description": "Eligible for search"},
        "is_eligible_checkout": {"type": "boolean", "description": "Eligible for checkout"},
    }
}


def validate_url(value: str) -> Tuple[bool, str]:
    """Validate URL format."""
    if not value:
        return False, "Empty URL"
    try:
        result = urlparse(value)
        if not all([result.scheme, result.netloc]):
            return False, f"Invalid URL format: {value[:50]}"
        if result.scheme not in ['http', 'https']:
            return False, f"Invalid URL scheme: {result.scheme}"
        return True, ""
    except Exception as e:
        return False, f"URL parse error: {str(e)}"


def validate_field(field_name: str, value: Any, spec: Dict) -> List[str]:
    """Validate a single field against its specification."""
    errors = []
    field_type = spec.get("type")

    # Type validation
    if field_type == "string":
        if not isinstance(value, str):
            errors.append(f"{field_name}: Expected string, got {type(value).__name__}")
        else:
            max_len = spec.get("max_length")
            if max_len and len(value) > max_len:
                errors.append(f"{field_name}: Exceeds max length {max_len} (got {len(value)})")
            if "enum" in spec and value not in spec["enum"]:
                errors.append(f"{field_name}: Invalid value '{value}', expected one of {spec['enum']}")

    elif field_type == "url":
        if not isinstance(value, str):
            errors.append(f"{field_name}: Expected URL string, got {type(value).__name__}")
        else:
            valid, msg = validate_url(value)
            if not valid:
                errors.append(f"{field_name}: {msg}")

    elif field_type == "number":
        if not isinstance(value, (int, float)):
            errors.append(f"{field_name}: Expected number, got {type(value).__name__}")
        else:
            if "min" in spec and value < spec["min"]:
                errors.append(f"{field_name}: Value {value} below minimum {spec['min']}")

    elif field_type == "integer":
        if not isinstance(value, int):
            errors.append(f"{field_name}: Expected integer, got {type(value).__name__}")
        else:
            if "min" in spec and value < spec["min"]:
                errors.append(f"{field_name}: Value {value} below minimum {spec['min']}")
            if "max" in spec and value > spec["max"]:
                errors.append(f"{field_name}: Value {value} above maximum {spec['max']}")

    elif field_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{field_name}: Expected boolean, got {type(value).__name__}")

    elif field_type == "array":
        if not isinstance(value, list):
            errors.append(f"{field_name}: Expected array, got {type(value).__name__}")

    return errors


def validate_product(product: Dict, index: int) -> Dict:
    """Validate a single product against the schema."""
    result = {
        "index": index,
        "item_id": product.get("item_id", f"unknown_{index}"),
        "errors": [],
        "warnings": [],
        "missing_required": [],
        "missing_recommended": [],
        "fields_present": list(product.keys()),
    }

    # Check required fields
    for field, spec in OPENAI_COMMERCE_SCHEMA["required"].items():
        if field not in product:
            result["missing_required"].append(field)
            result["errors"].append(f"Missing required field: {field}")
        elif product[field] is None or product[field] == "":
            result["errors"].append(f"Required field '{field}' is empty")
        else:
            errors = validate_field(field, product[field], spec)
            result["errors"].extend(errors)

    # Check recommended fields
    for field, spec in OPENAI_COMMERCE_SCHEMA["recommended"].items():
        if field not in product:
            result["missing_recommended"].append(field)
        elif product[field] is not None and product[field] != "":
            errors = validate_field(field, product[field], spec)
            if errors:
                result["warnings"].extend(errors)

    # Check policy fields
    for field, spec in OPENAI_COMMERCE_SCHEMA["policy"].items():
        if field in product and product[field]:
            errors = validate_field(field, product[field], spec)
            if errors:
                result["warnings"].extend(errors)

    # Check LLM enhancement fields
    for field, spec in OPENAI_COMMERCE_SCHEMA["llm_enhancement"].items():
        if field in product and product[field]:
            errors = validate_field(field, product[field], spec)
            if errors:
                result["warnings"].extend(errors)

    # Check optional fields
    for field, spec in OPENAI_COMMERCE_SCHEMA["optional"].items():
        if field in product and product[field] is not None:
            errors = validate_field(field, product[field], spec)
            if errors:
                result["warnings"].extend(errors)

    return result


def load_feed(file_path: str) -> List[Dict]:
    """Load products from JSONL file (handles gzip)."""
    products = []
    path = Path(file_path)

    if path.suffix == '.gz':
        opener = gzip.open
    else:
        opener = open

    with opener(path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                product = json.loads(line)
                products.append(product)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")

    return products


def generate_report(products: List[Dict], validation_results: List[Dict]) -> str:
    """Generate a comprehensive validation report."""
    report = []
    report.append("=" * 80)
    report.append("OPENAI COMMERCE FEED VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    total = len(products)
    with_errors = sum(1 for r in validation_results if r["errors"])
    with_warnings = sum(1 for r in validation_results if r["warnings"])
    valid = total - with_errors

    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Products:     {total:,}")
    report.append(f"Valid Products:     {valid:,} ({valid/total*100:.1f}%)" if total > 0 else "Valid Products:     0")
    report.append(f"Products w/Errors:  {with_errors:,}")
    report.append(f"Products w/Warnings:{with_warnings:,}")
    report.append("")

    # Required field coverage
    report.append("REQUIRED FIELD COVERAGE")
    report.append("-" * 40)
    for field in OPENAI_COMMERCE_SCHEMA["required"]:
        present = sum(1 for p in products if field in p and p[field])
        pct = (present / total * 100) if total > 0 else 0
        status = "✓" if pct == 100 else "⚠" if pct > 90 else "✗"
        report.append(f"{status} {field:25} {present:,}/{total:,} ({pct:.1f}%)")
    report.append("")

    # Recommended field coverage
    report.append("RECOMMENDED FIELD COVERAGE")
    report.append("-" * 40)
    for field in OPENAI_COMMERCE_SCHEMA["recommended"]:
        present = sum(1 for p in products if field in p and p[field])
        pct = (present / total * 100) if total > 0 else 0
        report.append(f"  {field:25} {present:,}/{total:,} ({pct:.1f}%)")
    report.append("")

    # Policy field coverage
    report.append("POLICY FIELD COVERAGE (Required for Checkout)")
    report.append("-" * 40)
    for field in OPENAI_COMMERCE_SCHEMA["policy"]:
        present = sum(1 for p in products if field in p and p[field])
        pct = (present / total * 100) if total > 0 else 0
        status = "✓" if pct == 100 else "⚠" if pct > 0 else "✗"
        report.append(f"{status} {field:25} {present:,}/{total:,} ({pct:.1f}%)")
    report.append("")

    # LLM Enhancement field coverage
    report.append("LLM ENHANCEMENT FIELD COVERAGE")
    report.append("-" * 40)
    for field in OPENAI_COMMERCE_SCHEMA["llm_enhancement"]:
        present = sum(1 for p in products if field in p and p[field])
        pct = (present / total * 100) if total > 0 else 0
        report.append(f"  {field:25} {present:,}/{total:,} ({pct:.1f}%)")
    report.append("")

    # Common errors
    error_counts = defaultdict(int)
    for r in validation_results:
        for err in r["errors"]:
            # Normalize error messages for counting
            err_key = re.sub(r"'[^']*'", "'...'", err)
            err_key = re.sub(r"\d+", "N", err_key)
            error_counts[err_key] += 1

    if error_counts:
        report.append("COMMON ERRORS (Top 10)")
        report.append("-" * 40)
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            report.append(f"  [{count:,}x] {err}")
        report.append("")

    # Sample products
    report.append("SAMPLE PRODUCTS (First 3)")
    report.append("-" * 40)
    for i, product in enumerate(products[:3], 1):
        report.append(f"\n--- Product {i}: {product.get('item_id', 'N/A')} ---")
        report.append(f"Title:        {product.get('title', 'N/A')[:60]}...")
        report.append(f"Brand:        {product.get('brand', 'N/A')}")
        report.append(f"Price:        {product.get('price', 'N/A')} {product.get('currency', '')}")
        report.append(f"Availability: {product.get('availability', 'N/A')}")
        report.append(f"Condition:    {product.get('condition', 'N/A')}")
        report.append(f"URL:          {product.get('url', 'N/A')[:60]}...")
        report.append(f"Image:        {product.get('image_url', 'N/A')[:60] if product.get('image_url') else 'MISSING'}...")
        report.append(f"Category:     {product.get('product_category', 'N/A')}")
        if product.get('q_and_a'):
            qa_preview = product['q_and_a'][:100].replace('\n', ' ')
            report.append(f"Q&A Preview:  {qa_preview}...")
        report.append(f"Fields:       {len(product)} total")
    report.append("")

    # Validation result
    report.append("=" * 80)
    if with_errors == 0:
        report.append("✓ VALIDATION PASSED - Feed is valid OpenAI Commerce format")
    else:
        report.append(f"⚠ VALIDATION ISSUES - {with_errors} products have errors")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate OpenAI Commerce Feed")
    parser.add_argument("input", help="Input JSONL file (supports .gz)")
    parser.add_argument("-o", "--output", help="Output report file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all errors")

    args = parser.parse_args()

    print(f"Loading feed from {args.input}...")
    products = load_feed(args.input)
    print(f"Loaded {len(products):,} products")

    print("Validating products...")
    validation_results = [validate_product(p, i) for i, p in enumerate(products)]

    if args.json:
        output = {
            "total_products": len(products),
            "valid_products": sum(1 for r in validation_results if not r["errors"]),
            "products_with_errors": sum(1 for r in validation_results if r["errors"]),
            "products_with_warnings": sum(1 for r in validation_results if r["warnings"]),
            "validation_results": validation_results if args.verbose else None,
        }
        print(json.dumps(output, indent=2))
    else:
        report = generate_report(products, validation_results)
        print(report)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
