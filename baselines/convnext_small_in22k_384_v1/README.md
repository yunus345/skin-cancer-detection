# convnext_small_in22k_384_v1 (2026-10-20) - ConvNeXt-Small 22k 384 + EMA

`convnext_tiny_in22k_384_v1` ile aynı tarif ve seed (42); farklar: ConvNeXt-Small
(`convnext_small.fb_in22k_ft_in1k_384`, 49.5M param), `--drop-path 0.2`, `--ema-decay 0.999`
(EMA gölge kopya eğitimi etkilemez). Kaggle kernel `skin-cancer-train-small` Version 1, T4,
3.3 dk/epoch (Tiny'nin ~1.6 katı). Kod: commit `d3bafda`. Karşılaştırma ve karar için
`baselines/convnext_tiny_in22k_384_seed2_ema/README.md` (ortak tablo orada).

Dosyalar: `best_model.pt` (en iyi epoch 25, val F1 0.801), `best_model_ema.pt` (EMA en iyi epoch 22,
0.794), `predictions.npz` / `predictions_ema.npz`. Ağırlıklar yerel disk (198MB, git'te değil).

**Karar: Small kabul edilmedi.** F1 iki Tiny koşusunun arasında (son 5 epoch 0.780; Tiny 0.788 /
0.775), maliyet 1.6 kat. Test sonuçlarına bakılmadı.
