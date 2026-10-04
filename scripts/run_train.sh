#!/usr/bin/env bash
set -euo pipefail

# Basit run script: virtualenv'i manuel aktif edin (README/SETUP.md'e bakın)
# Kullanım: bash scripts/run_train.sh --manifest data/processed/manifest.csv --output-dir models --feature-extract-epochs 5 --finetune-epochs 10 --device auto

# Varsayılanlar
MANIFEST="data/processed/manifest.csv"
OUTPUT_DIR="models"
FEATURE_EXTRACT_EPOCHS=5
FINETUNE_EPOCHS=10
DEVICE="auto"
SCHEDULER="none"
PATIENCE=3

# Arg parsing (basit)
while [[ $# -gt 0 ]]; do
  case $1 in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --feature-extract-epochs)
      FEATURE_EXTRACT_EPOCHS="$2"
      shift 2
      ;;
    --finetune-epochs)
      FINETUNE_EPOCHS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--manifest FILE] [--output-dir DIR] [--feature-extract-epochs N] [--finetune-epochs N] [--device auto|cpu|cuda|mps] [--scheduler none|cosine] [--patience N]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

# python.org Python kurulumu macOS sistem sertifikalarını kullanmıyor; pretrained ağırlık
# indirirken SSL doğrulaması başarısız olmasın diye venv'in certifi paketini işaret ediyoruz.
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
# print() çıktısı dosyaya/pipe'a yönlendirildiğinde buffer'da birikmesin, epoch ilerlemesini canlı görelim.
export PYTHONUNBUFFERED=1

python -m src.training.train --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" \
  --feature-extract-epochs "$FEATURE_EXTRACT_EPOCHS" --finetune-epochs "$FINETUNE_EPOCHS" \
  --device "$DEVICE" --scheduler "$SCHEDULER" --patience "$PATIENCE"
