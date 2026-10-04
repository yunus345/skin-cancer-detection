Skin Cancer Detection - Proje dizin yapısı

Bu proje, deri kanseri tespiti için derin öğrenme modelleri geliştirmek üzere hazırlanmıştır. Veri yüklemeniz sonrası aşağıdaki yapı kullanılacaktır.

Ana dizinlerin açıklaması:
- data/: Ham ve işlenmiş veriler (verileri buraya yükleyin). raw/ -> orijinal dosyalar; processed/ -> ön işlem sonrası veriler; external/ -> ek veri kaynakları.
- notebooks/: Keşifsel analiz (EDA) ve deney notları için Jupyter notebooklar.
- src/: Proje kaynak kodu
  - src/data/: Veri yükleme, ön işleme ve augmentation kodları
  - src/models/: Model mimarileri ve yardımcı fonksiyonlar
  - src/training/: Eğitim döngüleri, eğitim betikleri
  - src/evaluation/: Değerlendirme metrikleri, inference scriptleri
  - src/utils/: Yardımcı araçlar (logging, seed, config)
- models/: Eğitilmiş modellerin saklanacağı yer (genelde git'e eklenmez)
- experiments/: Deney konfigürasyonları, deneme kayıtları
- configs/: YAML/JSON konfigürasyon dosyaları
- tests/: Birim testleri
- docs/: Proje dokümantasyonu
- deployments/: Modeli üretime taşıma veya örnek inference servisleri
- results/: Grafikler, metrik tabloları, raporlar
- logs/: Eğitim/deney logları

README ve şablon dosyalarını inceleyip özelleştirebilirsiniz.
