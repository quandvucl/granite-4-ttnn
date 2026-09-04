import glob
import json
import os

files = sorted(glob.glob("./report_results/**/*.json", recursive=True))
# Exclude the output file itself if it already exists
files = [f for f in files if not f.endswith("summary.json")]

summary = {}

for path in files:
    d = json.load(open(path))
    key = os.path.relpath(path, "./report_results")
    entry = {
        "file": key,
        "model": d.get("model"),
        "hardware": d.get(
            "hardware",
            "ttnn" if "ttnn" in key else ("cuda" if "cuda" in key else "cpu"),
        ),
        "chunk_size": d.get("chunk_size"),
        "use_conv1d_kernel": d.get("use_conv1d_kernel"),
        "use_ssm_kernel": d.get("use_ssm_kernel"),
        "torch_compile": d.get("torch_compile"),
        "results": [
            {
                "prompt": r["prompt"],
                "tokens": r.get("tokens"),
                "ttft_ms": round(r["ttft_ms"], 1),
                "decode_toks": round(r["decode_toks"], 2),
            }
            for r in d.get("results", [])
        ],
    }
    summary[key] = entry

out_path = "./report_results/summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Written {len(summary)} entries to {out_path}")
