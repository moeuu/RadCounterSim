#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/.cache/datasets/jsi-triga-2026}"
mkdir -p "${output}"

download() {
  local file_id="$1"
  local name="$2"
  local md5="$3"
  if [[ ! -f "${output}/${name}" ]]; then
    curl -fL --retry 3 --continue-at - \
      -o "${output}/${name}" \
      "https://ndownloader.figshare.com/files/${file_id}"
  fi
  printf '%s  %s\n' "${md5}" "${output}/${name}" | md5sum -c -
}

download 65671431 lidar_maps_combined_cleaned.pcd 4bd9f684ec69f4d800774d6fbb789aef
download 65671392 rad_data_combined.csv c3a24ee04e7a1681a70021565826c3fc
download 65671425 map.pgm 2452b25f9cf0cefc3a4b3595037944f2
download 65671422 map.yaml d2f578779df60c61ddf0e838d85f3ebd

cat >"${output}/ATTRIBUTION.txt" <<'EOF'
JSI TRIGA Mark II survey data

Oakes, Seb (2026). Gaussian Process Regression of 3D radiation data,
gathered robotically in-situ at a nuclear test reactor. University of
Manchester. https://doi.org/10.48420/32727696.v1

Collected in May 2026 at the Jozef Stefan Institute TRIGA Mark II research
reactor using a modified Clearpath Jackal robot.

License: BSD 3-Clause
Source: https://figshare.manchester.ac.uk/articles/dataset/32727696

The dataset remains the work of its original author. It is downloaded into
.cache and is not redistributed as part of RadInterAct.
EOF

printf 'Dataset ready: %s\n' "${output}"
