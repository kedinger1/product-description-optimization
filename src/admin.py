"""
LLM Feed Optimizer - Admin UI
Web-based interface for managing feed transformation rules
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from src.transformer import transform_feed, transform_row_to_openai, parse_json_field, COMPANY_CONFIG

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / 'config'
OUTPUT_DIR = BASE_DIR / 'output'
UPLOADS_DIR = BASE_DIR / 'uploads'

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load/save configuration
def load_config(filename):
    filepath = CONFIG_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_config(filename, data):
    filepath = CONFIG_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_rules():
    return load_config('rules.json')

def save_rules(rules):
    save_config('rules.json', rules)

def load_company_config():
    return load_config('company.json')

def save_company_config(config):
    save_config('company.json', config)


# Routes
@app.route('/')
def index():
    """Dashboard home page"""
    # Get recent transformation stats if available
    stats_file = OUTPUT_DIR / 'last_transform_stats.json'
    stats = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)

    return render_template('index.html', stats=stats)


@app.route('/rules')
def rules_page():
    """Transformation rules editor"""
    rules = load_rules()
    return render_template('rules.html', rules=rules)


@app.route('/api/rules', methods=['GET'])
def get_rules():
    """Get current transformation rules"""
    rules = load_rules()
    # Return default rules if none exist
    if not rules:
        rules = get_default_rules()
    return jsonify(rules)


@app.route('/api/rules', methods=['POST'])
def update_rules():
    """Save transformation rules"""
    rules = request.json
    save_rules(rules)
    return jsonify({'status': 'success', 'message': 'Rules saved successfully'})


@app.route('/settings')
def settings_page():
    """Company settings editor"""
    config = load_company_config()
    return render_template('settings.html', config=config)


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get company settings"""
    return jsonify(load_company_config())


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Save company settings"""
    config = request.json
    save_company_config(config)
    # Update in-memory config
    COMPANY_CONFIG.update(config)
    return jsonify({'status': 'success', 'message': 'Settings saved successfully'})


@app.route('/preview')
def preview_page():
    """Product preview/test page"""
    return render_template('preview.html')


@app.route('/api/preview', methods=['POST'])
def preview_transform():
    """Preview transformation for a single product"""
    product_data = request.json

    # Apply transformation
    try:
        result = transform_row_to_openai(product_data)
        return jsonify({
            'status': 'success',
            'input': product_data,
            'output': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/transform')
def transform_page():
    """Feed transformation page"""
    # List available feeds in uploads directory
    feeds = []
    for f in UPLOADS_DIR.glob('*.csv'):
        feeds.append({
            'name': f.name,
            'size': f.stat().st_size,
            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })

    # List output files
    outputs = []
    for f in OUTPUT_DIR.glob('*.jsonl*'):
        outputs.append({
            'name': f.name,
            'size': f.stat().st_size,
            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })

    return render_template('transform.html', feeds=feeds, outputs=outputs)


@app.route('/api/upload', methods=['POST'])
def upload_feed():
    """Upload a new feed file"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Only CSV files are supported'}), 400

    filepath = UPLOADS_DIR / file.filename
    file.save(filepath)

    # Count rows
    with open(filepath, 'r', encoding='utf-8') as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract header

    return jsonify({
        'status': 'success',
        'message': f'Uploaded {file.filename}',
        'rows': row_count
    })


@app.route('/api/transform', methods=['POST'])
def run_transform():
    """Run feed transformation"""
    data = request.json
    input_file = data.get('input_file')
    output_format = data.get('format', 'openai')
    compress = data.get('compress', True)

    if not input_file:
        return jsonify({'status': 'error', 'message': 'No input file specified'}), 400

    input_path = UPLOADS_DIR / input_file
    print(f"[DEBUG] Looking for file at: {input_path}")
    print(f"[DEBUG] UPLOADS_DIR = {UPLOADS_DIR}")
    print(f"[DEBUG] Files in uploads: {list(UPLOADS_DIR.glob('*')) if UPLOADS_DIR.exists() else 'DIR NOT FOUND'}")

    if not input_path.exists():
        return jsonify({'status': 'error', 'message': f'Input file not found at {input_path}'}), 404

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'{output_format}_feed_{timestamp}.jsonl'
    output_path = OUTPUT_DIR / output_filename

    try:
        stats = transform_feed(str(input_path), str(output_path), compress=compress)

        # Save stats for dashboard
        stats['timestamp'] = datetime.now().isoformat()
        stats['input_file'] = input_file
        with open(OUTPUT_DIR / 'last_transform_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download an output file"""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    return send_file(filepath, as_attachment=True)


@app.route('/api/validate/<filename>')
def validate_output(filename):
    """Validate an output file against OpenAI Commerce Feed schema"""
    from src.validate_feed import load_feed, validate_product, OPENAI_COMMERCE_SCHEMA

    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    try:
        products = load_feed(str(filepath))
        validation_results = [validate_product(p, i) for i, p in enumerate(products)]

        # Calculate field coverage
        field_coverage = {}
        for category, fields in OPENAI_COMMERCE_SCHEMA.items():
            field_coverage[category] = {}
            for field in fields:
                present = sum(1 for p in products if field in p and p[field])
                field_coverage[category][field] = {
                    "present": present,
                    "total": len(products),
                    "percentage": round(present / len(products) * 100, 1) if products else 0
                }

        return jsonify({
            'status': 'success',
            'total_products': len(products),
            'valid_products': sum(1 for r in validation_results if not r["errors"]),
            'products_with_errors': sum(1 for r in validation_results if r["errors"]),
            'products_with_warnings': sum(1 for r in validation_results if r["warnings"]),
            'field_coverage': field_coverage,
            'sample_products': products[:3],
            'sample_errors': [r for r in validation_results if r["errors"]][:5]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sample/<filename>')
def get_sample(filename):
    """Get sample rows from a feed file"""
    filepath = UPLOADS_DIR / filename
    if not filepath.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5:  # Get first 5 rows
                break
            samples.append(row)

    return jsonify({'samples': samples})


def get_default_rules():
    """Return default transformation rules"""
    return {
        "title": {
            "template": "{brand} {title}",
            "max_length": 150,
            "transform": "none"
        },
        "description": {
            "source": "long_description",
            "fallback": "short_description",
            "max_length": 5000,
            "strip_html": True
        },
        "product_type_detection": {
            "jewelry": {
                "conditions": [
                    {"field": "category", "contains": "jewelry"},
                    {"field": "id", "starts_with": "ns-j-"}
                ]
            },
            "handbag": {
                "conditions": [
                    {"field": "category", "contains": "handbag"},
                    {"field": "category", "contains": "bag"}
                ]
            },
            "rolex_cpo": {
                "conditions": [
                    {"field": "brand", "equals": "rolex"},
                    {"field": "isPreOwned", "equals": "true"}
                ]
            },
            "preowned_watch": {
                "conditions": [
                    {"field": "isPreOwned", "equals": "true"}
                ]
            },
            "new_watch": {
                "default": True
            }
        },
        "field_mappings": {
            "item_id": "id",
            "title": "title",
            "brand": "brand",
            "url": "link",
            "image_url": "image_link",
            "price": "price",
            "availability": "availability_status",
            "group_id": "item_group_id"
        }
    }


if __name__ == '__main__':
    # Initialize default rules if none exist
    if not (CONFIG_DIR / 'rules.json').exists():
        save_rules(get_default_rules())

    port = int(os.environ.get('PORT', 5000))
    print("Starting Feed Optimizer Admin UI...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(host='0.0.0.0', port=port)
