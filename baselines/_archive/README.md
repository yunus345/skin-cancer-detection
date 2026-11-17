# Arşiv: final modelde kullanılmayan v1 deneyleri

2026-11-17'de `baselines/` altından buraya taşındı (içerik değişmedi). Her deneyin özeti ve kararı
[EXPERIMENTS.md](../../EXPERIMENTS.md)'de; ayrıntısı her klasörün kendi README'sinde.

| Klasör | Ne | Karar |
|---|---|---|
| `baseline_v1/` | EfficientNet-B0, 224 px, ilk kalıcı referans | ensemble_v1 üyesi |
| `convnext_tiny_v1/` | ConvNeXt-Tiny ImageNet-1k, 224 px | ensemble_v1 üyesi |
| `efficientnet_v2_s_v1/` | EfficientNetV2-S, 384 px | ensemble_v1 üyesi |
| `swin_v2_t_v1/` | Swin V2-Tiny, 256 px | ensemble_v1 üyesi |
| `ensemble_v1/` | Yukarıdaki 4 modelin kalibre ortalaması; ilk dış test ölçümü burada | final_v1 ile değiştirildi |
| `convnext_tiny_aug_v2/` | Augmentation v2 | ret |
| `convnext_tiny_384_v1/` | ConvNeXt 1k, 384 px, eski tarif | ret |
| `convnext_tiny_isic2019_v1/` | + ISIC 2019 ek veri | ret, v2'de doğru şekilde tekrar (ROADMAP C2) |
| `convnext_tiny_recipe2_v1/` | Tarif v2, 224 px | tarif kabul, model kullanılmadı |
| `convnext_small_in22k_384_v1/` | ConvNeXt-Small 22k 384 | ret (gürültü içinde, 1.6 kat pahalı) |

Final modelin iki üyesi arşivde değil, `baselines/` altında: `convnext_tiny_in22k_384_v1/` (seed 42) ve
`convnext_tiny_in22k_384_seed2_ema/` (seed 2). Denetim raporundaki (`docs/AUDIT_2026-10-26.md`) yollar taşıma
öncesine aittir; o rapor üzerine yazılmadığı için düzeltilmedi.
