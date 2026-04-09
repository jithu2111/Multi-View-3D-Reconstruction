"""
Convert the three MVS reconstruction PLYs in web/models/ to GLB so they
can be fed into gltf-transform for Draco compression.

Only the format is changed here — no geometry decimation, no color
quantization. Draco compression happens as a separate pass afterwards
(see web/tools/compress_glbs.sh).

Run from the project root:
    python web/tools/convert_ply_to_glb.py
"""
from pathlib import Path

import numpy as np
import trimesh

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
TARGETS = ["dino", "temple", "templeSparseRing"]


def convert(name: str) -> None:
    ply_path = MODELS_DIR / f"{name}.ply"
    glb_path = MODELS_DIR / f"{name}.glb"

    print(f"[{name}] loading {ply_path.name}...")
    mesh = trimesh.load(ply_path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected a single Trimesh for {name}, got {type(mesh).__name__}")

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    colors = np.asarray(mesh.visual.vertex_colors)  # (N, 4) uint8 RGBA

    print(f"[{name}] verts={len(verts):,} faces={len(faces):,} colors={colors.shape}")

    # trimesh exports glb via mesh.export, which serializes vertex colors
    # as a COLOR_0 attribute in the glTF buffer. We keep face indices as-is
    # (no decimation) and let Draco handle the compression later.
    glb_bytes = mesh.export(file_type="glb")
    glb_path.write_bytes(glb_bytes)

    size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"[{name}] wrote {glb_path.name} ({size_mb:.1f} MB)")


def main() -> None:
    for name in TARGETS:
        convert(name)
        print()


if __name__ == "__main__":
    main()