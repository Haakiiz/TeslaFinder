import sys
import types

# Provide minimal stubs for external dependencies so modules import without them.

# yaml stub
if 'yaml' not in sys.modules:
    yaml_stub = types.ModuleType('yaml')
    yaml_stub.safe_load = lambda *a, **k: {}
    sys.modules['yaml'] = yaml_stub

# playwright.async_api stub
if 'playwright.async_api' not in sys.modules:
    pw_async_stub = types.ModuleType('playwright.async_api')
    pw_async_stub.async_playwright = lambda: None
    class Browser: pass
    class Page: pass
    pw_async_stub.Browser = Browser
    pw_async_stub.Page = Page
    sys.modules.setdefault('playwright.async_api', pw_async_stub)
    # also add parent package
    if 'playwright' not in sys.modules:
        pw_pkg = types.ModuleType('playwright')
        pw_pkg.async_api = pw_async_stub
        sys.modules['playwright'] = pw_pkg
