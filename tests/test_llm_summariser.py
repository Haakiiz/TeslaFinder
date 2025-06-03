import json
import types
import sys
from pathlib import Path
import importlib.util

import pytest

# Helper fixture to load the module despite the space in filename
@pytest.fixture
def llm_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / 'LLM Summariser.py'

    # Provide stub openai and anthropic modules so the import succeeds
    fake_openai = types.ModuleType('openai')
    fake_openai.ChatCompletion = types.SimpleNamespace(create=lambda **kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(message={'content': 'ok'})]))
    fake_openai.api_key = None
    monkeypatch.setitem(sys.modules, 'openai', fake_openai)

    fake_anthropic = types.ModuleType('anthropic')
    fake_anthropic.Anthropic = lambda api_key=None: types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(message={'content': 'ok'})])))
    monkeypatch.setitem(sys.modules, 'anthropic', fake_anthropic)

    spec = importlib.util.spec_from_file_location('llm_summariser', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_build_prompt_empty(llm_module):
    assert llm_module.build_prompt([]) == 'No new or changed listings.'

def test_build_prompt_contains_fields(llm_module):
    listings = [{
        'ad_id': '123',
        'title': 'Great car',
        'year': 2023,
        'mileage': 10000,
        'price': 500000,
        'location': 'Oslo',
        'url': 'http://example.com'
    }]
    prompt = llm_module.build_prompt(listings)
    assert '123 | Great car | 2023 | 10000 km | 500000 kr | Oslo | http://example.com' in prompt


def test_load_delta_valid(tmp_path, llm_module):
    data = [{'ad_id': '1'}]
    path = tmp_path / 'delta.json'
    path.write_text(json.dumps(data))
    assert llm_module.load_delta(str(path)) == data

def test_load_delta_missing_file(llm_module):
    with pytest.raises(SystemExit):
        llm_module.load_delta('nonexistent.json')

def test_load_delta_bad_json(tmp_path, llm_module):
    bad = tmp_path / 'bad.json'
    bad.write_text('{bad json}')
    with pytest.raises(SystemExit):
        llm_module.load_delta(str(bad))
