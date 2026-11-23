"""Use the independently checked Redrock 0.15.4 convolution-reuse wrapper."""
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/"output"
sys.path.insert(0,str(HERE))
import fit_batch as base


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""): h.update(block)
    return h.hexdigest()


def main():
    check=OUT/"redrock_cache_validation.json"
    evidence=json.loads(check.read_text())
    wrapper=HERE/"rrdesi_015_cached_spawn.py"
    if evidence["status"]!="passed" or evidence["wrapper_sha256"]!=sha(wrapper):
        raise ValueError("Redrock convolution reuse has not passed its pinned comparison")
    base.WRAPPER=wrapper
    base.main()
    # Add audit evidence after the unchanged common Redrock output processor.
    pos=sys.argv.index("--output-prefix")
    runtime=Path(sys.argv[pos+1]).with_suffix(".json")
    result=json.loads(runtime.read_text())
    result.update({"backend":"x86_64 Redrock 0.15.4 with exact convolution reuse",
        "cached_wrapper_sha256":sha(wrapper),"cache_validation_sha256":sha(check),
        "cached_fit_script_sha256":sha(Path(__file__))})
    runtime.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
