#!/usr/bin/env bash
set -euo pipefail

# Test seti üzerinde tek seferlik değerlendirme (ancak eğitim kesinleştikten sonra çalıştırın).
# Kullanım: bash scripts/run_eval.sh --manifest data/processed/manifest.csv --checkpoint models/best_model.pt --device auto

# Varsayılanlar
MANIFEST="data/processed/manifest.csv"
CHECKPOINT="models/best_model.pt"
OUTPUT="models/test_metrics.json"
DEVICE="auto"

# Arg parsing (basit)
while [[ $# -gt 0 ]]; do
  case $1 in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--manifest FILE] [--checkpoint FILE] [--output FILE] [--device auto|cpu|cuda|mps]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

# python.org Python kurulumu macOS sistem sertifikalarını kullanmıyor; build_model() ImageNet
# ağırlıklarını yüklerken SSL doğrulaması başarısız olmasın diye venv'in certifi paketini işaret ediyoruz.
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export PYTHONUNBUFFERED=1

python -m src.evaluation.evaluate --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" --device "$DEVICE"
