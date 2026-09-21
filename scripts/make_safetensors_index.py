#!/usr/bin/env python3
"""The HF repo Huangxs/MedPLIB-7b-2e ships safetensors shards but only a *.bin index.
transformers 4.31 resolves shards through an index file, so this writes
model.safetensors.index.json from pytorch_model.bin.index.json after verifying, from the
safetensors headers, that every tensor of the index really is in the renamed shard.
Dependency-free (parses the safetensors header directly)."""
import json, struct, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/peilin/HF_MODEL/MedPLIB-7b-2e")
idx = json.load(open(d / "pytorch_model.bin.index.json"))
wm = {k: v.replace(".bin", ".safetensors") for k, v in idx["weight_map"].items()}
shards = sorted(set(wm.values()))
present = {}
total = 0
for s in shards:
    p = d / s
    assert p.exists(), f"missing shard {p}"
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    present[s] = set(header)
    total += sum(1 for _ in header)
missing = [k for k, s in wm.items() if k not in present[s]]
extra = {s: present[s] - {k for k, v in wm.items() if v == s} for s in shards}
print(f"index tensors {len(wm)}, safetensors tensors {total}, missing {len(missing)}, extra {sum(len(v) for v in extra.values())}")
if missing:
    print("missing examples:", missing[:10]); sys.exit(1)
for s, e in extra.items():
    if e:
        print(f"  extra in {s}: {sorted(e)[:5]} ...")
        for k in e:
            wm[k] = s
out = {"metadata": idx.get("metadata", {}), "weight_map": wm}
json.dump(out, open(d / "model.safetensors.index.json", "w"), indent=2)
print("wrote", d / "model.safetensors.index.json")
