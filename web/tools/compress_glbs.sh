#!/usr/bin/env bash
# Compress the uncompressed GLB files in web/models/ with Draco, using
# high-precision quantization settings so that quality loss is below the
# reconstruction noise floor of the MVS pipeline.
#
# Quantization bits (higher = better quality, larger file):
#   position: 14  → max positional error ≈ bbox / 16384 (sub-millimeter)
#   normal:   10  → imperceptible shading difference
#   color:     8  → matches sRGB / display precision exactly
#   generic:  12  → safe default for any other attributes
#
# Run from the project root:
#     bash web/tools/compress_glbs.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
MODELS_DIR="web/models"
GLTF_TRANSFORM="web/node_modules/.bin/gltf-transform"

for name in dino temple templeSparseRing; do
    in_file="${MODELS_DIR}/${name}.glb"
    out_file="${MODELS_DIR}/${name}.draco.glb"

    if [[ ! -f "$in_file" ]]; then
        echo "[${name}] skipping — ${in_file} not found"
        continue
    fi

    echo "[${name}] compressing ${in_file}..."
    "$GLTF_TRANSFORM" draco "$in_file" "$out_file" \
        --method edgebreaker \
        --quantize-position 14 \
        --quantize-normal 10 \
        --quantize-color 8 \
        --quantize-texcoord 12 \
        --quantize-generic 12

    orig_size=$(stat -f%z "$in_file" 2>/dev/null || stat -c%s "$in_file")
    new_size=$(stat -f%z "$out_file" 2>/dev/null || stat -c%s "$out_file")
    orig_mb=$(awk "BEGIN{printf \"%.1f\", $orig_size/1048576}")
    new_mb=$(awk "BEGIN{printf \"%.1f\", $new_size/1048576}")
    ratio=$(awk "BEGIN{printf \"%.1f\", $orig_size/$new_size}")
    echo "[${name}] ${orig_mb} MB → ${new_mb} MB (${ratio}× smaller)"
    echo

    # Replace the uncompressed GLB with the compressed one
    mv "$out_file" "$in_file"
done

echo "Done."