import json

def safe_load(stream):
    """Minimal YAML loader using JSON syntax."""
    return json.load(stream)
