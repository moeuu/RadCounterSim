# RadInterAct local inference runtime

This directory is the product-owned runtime layout. Large or platform-specific
artifacts are deliberately not committed to Git.

```text
runtime/llm/
├── bin/
│   ├── linux-x86_64-cpu/llama-server
│   ├── linux-x86_64-vulkan/llama-server
│   └── linux-x86_64-cuda/llama-server
├── models/Qwen3-4B-Q4_K_M.gguf
└── logs/llama-server.log
```

Prepare a development or release tree with:

```bash
./scripts/build_llama_runtime.sh
uv run python scripts/fetch_llm_model.py
```

The build script pins llama.cpp and installs verified official CPU and Vulkan
release binaries, then optionally produces a CUDA binary when the CUDA compiler
is present. The model downloader pins the official Qwen repository revision and
verifies the complete GGUF SHA-256 before installation.

At runtime RadInterAct prefers CUDA, falls back to Vulkan GPU offload, and uses
the CPU binary when no GPU is available. Set
`RADCOUNTER_LLM_GPU_MODE=cpu|hybrid|auto|gpu` to override layer placement. A
developer may set `RADCOUNTER_LLM_ENDPOINT` to use an already-running loopback
server; this is not required in the packaged app.
