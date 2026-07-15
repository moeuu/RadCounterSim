# Native Embree backend

The optional `_radcounter_embree` module traces finite source-to-detector segments
through closed triangle meshes. It accumulates distance by material and evaluates
`exp(-sum(mu * distance))` independently for every energy bin.

## Prerequisite

Install Embree 4 with its CMake package visible through `CMAKE_PREFIX_PATH` or
`embree_DIR`. Python build tooling is acquired through `uv`; do not install it with
`pip`.

```bash
./scripts/build_native.sh
export PYTHONPATH="$PWD/build/native/python${PYTHONPATH:+:$PYTHONPATH}"
uv run python -c 'from radcounter.core.radiation.embree_native import native_embree_available; print(native_embree_available())'
```

Meshes are copied into native buffers. Call `commit()` after any geometry change.
The scene revision advances on each commit and can be included in transfer-cache
keys. Rays are assumed to start outside consistently closed meshes; a target inside
a mesh receives attenuation through the remaining entry-to-target segment.
