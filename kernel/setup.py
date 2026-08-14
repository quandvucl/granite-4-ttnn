"""
Build the mamba2_ssm_decode fused op shared library.

Usage (from /work/tt-granite/kernel/):
    source /work/tt-granite/env/bin/activate
    python setup.py

Produces: libmamba2_ssm_decode.so
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).parent.resolve()
HDR = HERE / "tt-metal-headers"
TTNN_SITE = Path("/work/tt-granite/env/lib/python3.10/site-packages/ttnn")

def build():
    """Compile mamba2_ssm_decode.cpp into libmamba2_ssm_decode.so."""
    include_dirs = [
        str(HDR / "ttnn/cpp"),
        str(HDR / "ttnn/api"),
        str(HDR / "tt_stl"),
        str(HDR / "tt_metal/api"),
        str(TTNN_SITE.parent / "torch/include"),
        "/work/tt-metal/.cpmcache/tt-logger/d2339ce68562cae34cd95f3fece7fd94eb0529b7/include",
        "/work/tt-metal/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include",
        "/work/tt-metal/.cpmcache/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0",
        "/work/tt-metal/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include",
        "/work/tt-metal/.cpmcache/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include",
        str(HERE / "tt-metal-headers/umd"),
        "/work/tt-metal/tt_metal/third_party/tracy/public",
        str(TTNN_SITE / "tt_metal/api"),
        str(TTNN_SITE / "tt_metal/hostdevcommon/api"),
    ]

    cxx_flags = [
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-DFMT_HEADER_ONLY=1",
        "-w",
    ]

    link_flags = [
        f"-L{TTNN_SITE}/build/lib",
        "-l:_ttnncpp.so",
        f"-Wl,-rpath,{TTNN_SITE}/build/lib",
    ]

    src = str(HERE / "mamba2_ssm_decode.cpp")
    out = str(HERE / "libmamba2_ssm_decode.so")

    inc_args = [f"-I{d}" for d in include_dirs]
    cmd = ["g++-12", "-shared", *cxx_flags, *inc_args, src, *link_flags, "-o", out]

    print("Building mamba2_ssm_decode.so ...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Build failed (exit {result.returncode})")
    print(f"Built: {out}")

if __name__ == "__main__":
    build()
