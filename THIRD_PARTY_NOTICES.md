# Third-party notices

RadCounterSim is distributed under the BSD 3-Clause License in `LICENSE`.
That license covers RadCounterSim's own source code only.

## Architectural reference

OceanSim was consulted as an architectural reference for structuring an
NVIDIA Isaac Sim robotics simulator extension. OceanSim is licensed under the
BSD 3-Clause License. No OceanSim source code or assets are included in this
repository.

- Project: https://github.com/umfieldrobotics/OceanSim
- License: https://github.com/umfieldrobotics/OceanSim/blob/main/LICENSE

## External runtimes and libraries

The following independently distributed software can be used by
RadCounterSim. It is not relicensed by RadCounterSim, and is not included in
this repository unless a file explicitly states otherwise.

- NVIDIA Isaac Sim and Omniverse Kit: governed by NVIDIA's applicable license
  agreements and bundled third-party notices.
- Intel Embree: Apache License 2.0.
- ROS 2 and its packages: governed by each package's declared license.
- Intel oneAPI Threading Building Blocks: Apache License 2.0.
- Python packages resolved by `uv.lock`: governed by each package's metadata
  and license files.

## Optional packaged local-inference components

Release packages may include these independently licensed components under
`runtime/llm`. They are not stored in this Git repository.

- llama.cpp, including `llama-server`: MIT License,
  https://github.com/ggml-org/llama.cpp
- Qwen3-4B and its official GGUF quantization: Apache License 2.0,
  https://huggingface.co/Qwen/Qwen3-4B-GGUF

The release packaging process pins and verifies both inputs. Their copyright,
license, and notice files must accompany redistributed binaries and models.

## Optional Fukushima Daiichi CAD environment

The system catalog can fetch and import the independently distributed
`Qualot/fukushima_daiichi_solidworks` generic Fukushima Daiichi CAD model.
Neither its SolidWorks source nor derived USD assets are stored in this Git
repository.

- Project: https://github.com/Qualot/fukushima_daiichi_solidworks
- Pinned integration revision: `f6541deb6159c5d908a4f028d021e3d2c9f7f8e8`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Attribution: Qualot/fukushima_daiichi_solidworks contributors

The fetch and Linux conversion scripts preserve source revision, license,
attribution, converter metadata, and SHA-256 provenance. Anyone redistributing
a derived USD, render, or other adaptation is responsible for retaining the CC BY 4.0
attribution and indicating modifications.

Users are responsible for reviewing and accepting the terms of external
software before installing or running it.
