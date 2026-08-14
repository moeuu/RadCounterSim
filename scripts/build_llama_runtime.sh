#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_ref="${RADCOUNTER_LLAMA_CPP_REF:-b9637}"
runtime_root="${RADCOUNTER_LLM_RUNTIME_DIR:-$repository_root/runtime/llm}"
download_root="$repository_root/.cache/llama.cpp-$llama_ref-releases"
mkdir -p "$download_root" "$runtime_root/licenses"

install_release() {
  local backend="$1"
  local expected_sha256="$2"
  local archive="llama-$llama_ref-bin-ubuntu-$backend-x64.tar.gz"
  if [[ "$backend" == "cpu" ]]; then
    archive="llama-$llama_ref-bin-ubuntu-x64.tar.gz"
  fi
  local url="https://github.com/ggml-org/llama.cpp/releases/download/$llama_ref/$archive"
  local cached="$download_root/$archive"
  local destination="$runtime_root/bin/linux-x86_64-$backend"
  local extract_root
  extract_root="$(mktemp -d)"

  if [[ ! -f "$cached" ]]; then
    curl --fail --location --retry 3 --output "$cached.part" "$url"
    mv "$cached.part" "$cached"
  fi
  printf '%s  %s\n' "$expected_sha256" "$cached" | sha256sum --check --status
  tar -xzf "$cached" -C "$extract_root"
  local server
  server="$(find "$extract_root" -type f -name llama-server -print -quit)"
  if [[ -z "$server" ]]; then
    printf '%s\n' "llama-server was not present in $archive" >&2
    exit 1
  fi
  mkdir -p "$destination"
  cmake -E copy "$server" "$destination/llama-server"
  find "$(dirname "$server")" -maxdepth 1 \( -type f -o -type l \) -name 'lib*.so*' \
    -exec cmake -E copy {} "$destination" \;
  chmod +x "$destination/llama-server"
  rm -rf "$extract_root"
  printf '%s\n' "Installed verified $backend llama-server in $destination"
}

install_release cpu a50ee14f021a9d8e92e30f622f7e3be1318ee1125bb9a9ba8d2025388df48743
install_release vulkan 6ca268d758aae9e8518afa43042678e8b60b47f0d34df7d6efff4ca622c74313

license_path="$runtime_root/licenses/llama.cpp-MIT.txt"
license_url="https://raw.githubusercontent.com/ggml-org/llama.cpp/$llama_ref/LICENSE"
if [[ ! -f "$license_path" ]]; then
  curl --fail --location --retry 3 --output "$license_path.part" "$license_url"
  mv "$license_path.part" "$license_path"
fi
printf '%s  %s\n' \
  94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d \
  "$license_path" | sha256sum --check --status

build_backend() {
  local backend="$1"
  local cuda_flag="$2"
  local build_root="$repository_root/.cache/llama.cpp-$llama_ref-build-$backend"
  local destination="$runtime_root/bin/linux-x86_64-$backend"

  cmake -S "$source_root" -B "$build_root" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA="$cuda_flag" \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build "$build_root" --config Release --target llama-server -j
  mkdir -p "$destination"
  cmake -E copy "$build_root/bin/llama-server" "$destination/llama-server"
  find "$build_root/bin" -maxdepth 1 \( -type f -o -type l \) -name 'lib*.so*' \
    -exec cmake -E copy {} "$destination" \;
  chmod +x "$destination/llama-server"
  printf '%s\n' "Installed $backend llama-server in $destination"
}

if command -v nvcc >/dev/null 2>&1; then
  source_root="$repository_root/.cache/llama.cpp-$llama_ref"
  if [[ ! -d "$source_root/.git" ]]; then
    git clone --depth 1 --branch "$llama_ref" \
      https://github.com/ggml-org/llama.cpp.git "$source_root"
  fi
  build_backend cuda ON
else
  printf '%s\n' "CUDA compiler not found; using packaged Vulkan GPU offload." >&2
fi
