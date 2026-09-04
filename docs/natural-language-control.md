# Natural-language application control

RadInterAct accepts English instructions through one local,
schema-constrained command surface. The language model proposes a plan; it
never receives direct Python, shell, USD, or robot-controller access.

## Product architecture

```text
RadInterAct UI
  -> public scene/action snapshot
  -> loopback OpenAI-compatible request
  -> bundled llama.cpp + Qwen3-4B GGUF
  -> strict CommandPlan validation
  -> preview / physical-action confirmation
  -> existing Isaac workflow services
```

The final package uses the user's separately installed Isaac Sim. RadInterAct
owns and starts `llama-server` as a hidden child process, selects the packaged
CPU, Vulkan, or CUDA binary, assigns a random loopback port, waits for its
health check, and stops it with the application. No Ollama daemon, Python
inference library, PyTorch installation, API key, or internet connection is
required after the release assets are installed.

RadInterAct is BSD-licensed, but its OSS status does not change NVIDIA's
terms. The package deliberately does not contain Isaac Sim or Omniverse Kit;
each operator installs and accepts the terms for their own copy. NVIDIA's
[Isaac Sim license FAQ](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/common/license-faq.html)
states that internal R&D use is free, while redistributing Isaac Sim with
Omniverse Kit to third parties requires NVIDIA AI Enterprise licensing. A
future all-in-one RadInterAct installer must therefore keep this same
user-installed-Isaac boundary unless separate redistribution rights are
obtained.

## Source-tree setup

Install the pinned release assets once:

```bash
./scripts/build_llama_runtime.sh
uv run python scripts/fetch_llm_model.py
```

The model downloader installs the official Qwen `Qwen3-4B-Q4_K_M.gguf` and
verifies its SHA-256. The llama.cpp setup installs verified official CPU and
Vulkan binaries and, when `nvcc` is available, also produces an NVIDIA CUDA
build. These files live under
`runtime/llm` and are excluded from Git; a binary release includes them with
their upstream license and notice files.

Launch the application with:

```bash
OMNI_KIT_ACCEPT_EULA=YES ./scripts/run_app.sh
```

After NVIDIA's EULA marker exists, subsequent launches do not need that
environment variable. A desktop entry can be installed with:

```bash
uv run python scripts/install_desktop_entry.py
```

## Operator behavior

Examples:

- `Move to the protected area and measure for 2 seconds.`
- `Decontaminate the contaminated surface, then measure.`
- `Decontaminate the irregular wall source up to three times until at least 70% is removed.`
- `Place the shield at 25% of the source-to-protected-area line, then move it to 65%.`
- `Visit every measurement station in order and measure for 2 seconds at each station.`
- `Pause the simulation.`
- `Move the contaminated drum to the disposal area.`

Pause, status, measurement, and visualization commands run immediately.
Stage replacement, reset, and every scene-derived physical action require an
explicit confirmation. A single instruction may contain up to 24 ordered
steps, including multiple physical actions. The complete workflow is confirmed
once; the live scene and candidate feasibility are regenerated after every step
before the next operation can execute.

Complex plans may contain up to 24 logical steps and at most 48 bounded
executions after repeats are expanded. One decontamination step may run up to
five complete passes. It can stop early on an allowlisted public result:
removed fraction, cumulative remaining activity fraction, or active-face
coverage. Measurement repeats may stop on a public measured-rate threshold.
The host evaluates these conditions; model text can neither claim success nor
create an unbounded loop. If the attempt limit is reached first, the GUI reports
that the requested condition remains unmet.

Shield candidates expose the physical panel and their source-to-protected-area
placement fraction. The first successful deployment is a `place_shield`
operation; later corrections of that same panel are regenerated as
`move_shield` and do not consume another inventory unit. Pickup poses, routes,
grasp frames, placement poses, and collision checks remain host-derived.

The model sees only:

- session state and stage name;
- allowlisted application capabilities;
- action IDs, public labels, target paths, and feasibility flags generated from
  the current planning boundary.

It does not see `TruthState`. Original instructions, validated plans,
confirmation decisions, public results, and failures are appended to
`artifacts/ui/natural_language_commands.jsonl`.

## Hardware behavior

The default model is a 4-bit 4B GGUF so it remains practical beside Isaac Sim
on an RTX 3080-class machine. `llama-server` uses automatic GPU layer fitting
through CUDA or Vulkan when NVIDIA hardware is available and uses the packaged
CPU build otherwise.
The context is capped at 8192 tokens so multi-room scenes with several surface,
shield, object, and detector candidates still fit without granting the model a
chat-sized or open-ended context.

Overrides for development and diagnosis:

- `RADCOUNTER_LLM_GPU_MODE=cpu|hybrid|auto|gpu`
- `RADCOUNTER_LLM_MODEL=/absolute/model.gguf`
- `RADCOUNTER_LLAMA_SERVER=/absolute/llama-server`
- `RADCOUNTER_LLM_RUNTIME_DIR=/absolute/runtime/llm`
- `RADCOUNTER_LLM_ENDPOINT=http://127.0.0.1:PORT`

Only loopback inference endpoints are accepted by default.
