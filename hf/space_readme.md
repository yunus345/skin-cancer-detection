---
title: Skin Lesion Classifier (research demo)
emoji: 🔬
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
license: cc-by-nc-4.0
models:
  - emreozkn/skin-lesion-convnext-v1
short_description: HAM10000 7-class dermoscopy classifier with Grad-CAM
---

# Cilt lezyonu sınıflandırma (araştırma demosu)

**Araştırma ve öğrenme amaçlıdır, tanı aracı değildir.** Gerçek hasta görüntüsü yüklemeyin.

Bir öğrenci projesi: HAM10000 ile eğitilmiş, iki ConvNeXt-Tiny modelinin kalibre ortalaması. Dış testte
(ISIC 2018, 1512 görüntü) macro F1 0.795, balanced accuracy 0.796.

- Model: https://huggingface.co/emreozkn/skin-lesion-convnext-v1 (`v1` etiketi)
- Kod, deney geçmişi ve sınırlamalar: https://github.com/yunus345/skin-cancer-detection

Bu Space'in dosyaları GitHub deposundaki `hf/build_space.py` ile oluşturulur; burada elle değiştirilmez.
