"""Cloud inference must not acquire a dependency on a local inference engine."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def test_cloud_import_and_compile_without_engine_imports():
    script = '''
import importlib.abc, json, sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm_alp', 'vllm', 'vllm_omni', 'mlx', 'mlx_serve_alp', 'xgrammar', 'torch', 'transformers'}:
            raise AssertionError('Forbidden cloud dependency: ' + fullname)
sys.meta_path.insert(0, NoEngine())
from semantic_router_alp.cloud import CloudAdapter
from alp_schema_mcp.runtime.catalog import ServerConfig
config = ServerConfig.from_file(sys.argv[1])
adapter = CloudAdapter(config, projection='typed')
request = {'model':'example', 'messages':[{'role':'user','content':'Finish.'}],
           'alp':{'allowed_operations':['agent_final'],'catalog_ref':'product-assistant'}}
assert adapter.prepare(json.dumps(request))['body']['tools']
'''
    subprocess.run([sys.executable, "-c", script,
                    str(Path(__file__).parents[1] / "examples/catalogs.json")], check=True)


def test_patch_manifest_is_complete_and_rejects_tampering():
    import hashlib
    import pytest
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("prepare", root / "scripts/prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    manifest = json.loads((root / "patches/checksums.json").read_text())
    series = (root / "patches/series").read_text()
    assert prepare.checked_patches(series, manifest, lambda n: (root / "patches" / n).read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        prepare.checked_patches(series, manifest, lambda n: b"modified")
    with pytest.raises(ValueError, match="series"):
        prepare.checked_patches(series + series, manifest, lambda n: b"")
    with pytest.raises(ValueError, match="basenames"):
        prepare.checked_patches("../escape", {"../escape": hashlib.sha256(b"").hexdigest()}, lambda n: b"")
