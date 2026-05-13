#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_NAME="mps_ttn_training_export"
BUILD_DIR="$ROOT_DIR/dist/$BUNDLE_NAME"
ARCHIVE_PATH="$ROOT_DIR/dist/${BUNDLE_NAME}.tar.gz"
ZIP_PATH="$ROOT_DIR/dist/${BUNDLE_NAME}.zip"

FILES=(
  "src/spectroscopy_qml/__init__.py"
  "src/spectroscopy_qml/ir/__init__.py"
  "src/spectroscopy_qml/ir/MPS_TTN/__init__.py"
  "src/spectroscopy_qml/ir/MPS_TTN/run.py"
  "src/spectroscopy_qml/ir/MPS_TTN/shared_data.py"
  "src/spectroscopy_qml/ir/MPS_TTN/train_mps.py"
  "src/spectroscopy_qml/ir/MPS_TTN/train_ttn102.py"
  "src/spectroscopy_qml/ir/mps_encoder_final/__init__.py"
  "src/spectroscopy_qml/ir/mps_encoder_final/config.py"
  "src/spectroscopy_qml/ir/mps_encoder_final/data_loader.py"
  "src/spectroscopy_qml/ir/mps_encoder_final/model.py"
  "src/spectroscopy_qml/ir/mps_encoder_final/train.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/__init__.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/__init__.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment5/__init__.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment5/losses.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment5/model.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment6/__init__.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment6/model.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10/__init__.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10/model.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/__init__.py"
  "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/model.py"
  "export_assets/mps_ttn_training_export/README.md"
  "export_assets/mps_ttn_training_export/requirements.txt"
  "export_assets/mps_ttn_training_export/train_both.sh"
  "export_assets/mps_ttn_training_export/train_sequential.sh"
  "export_assets/mps_ttn_training_export/train_mps.sh"
  "export_assets/mps_ttn_training_export/train_ttn.sh"
)

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

for rel_path in "${FILES[@]}"; do
  src_path="$ROOT_DIR/$rel_path"
  if [[ ! -f "$src_path" ]]; then
    echo "Missing required file: $rel_path" >&2
    exit 1
  fi
  if [[ "$rel_path" == export_assets/* ]]; then
    target_rel="${rel_path#export_assets/mps_ttn_training_export/}"
  else
    target_rel="$rel_path"
  fi
  mkdir -p "$BUILD_DIR/$(dirname "$target_rel")"
  cp "$src_path" "$BUILD_DIR/$target_rel"
done

cat > "$BUILD_DIR/manifest.txt" <<'EOF'
This archive contains the minimal code required to train the IR MPS and TTN 10.2
pipelines from src/spectroscopy_qml/ir/MPS_TTN on another machine.

Data is not included. Provide parquet files under data/raw on the target machine.
EOF

chmod +x \
  "$BUILD_DIR/train_both.sh" \
  "$BUILD_DIR/train_sequential.sh" \
  "$BUILD_DIR/train_mps.sh" \
  "$BUILD_DIR/train_ttn.sh"

mkdir -p "$ROOT_DIR/dist"
tar -czf "$ARCHIVE_PATH" -C "$ROOT_DIR/dist" "$BUNDLE_NAME"
rm -f "$ZIP_PATH"
(cd "$ROOT_DIR/dist" && zip -qr "${BUNDLE_NAME}.zip" "$BUNDLE_NAME")
echo "Created: $ARCHIVE_PATH"
echo "Created: $ZIP_PATH"
