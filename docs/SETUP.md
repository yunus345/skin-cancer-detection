Quickstart - Kurulum ve Çalıştırma

1) Python sanal ortam oluşturma (bash):

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

2) Veri yerleşimi:

   - Ham verileri data/raw/ içine koyun.
   - İlk ön işleme için data/processed/ dizinini oluşturun veya aşağıdaki örnek loader ile dosyaları taşıyın.

3) Eğitim (örnek):

   - Basit bir başlangıç için scripts/run_train.sh kullanabilirsiniz:
     bash scripts/run_train.sh --manifest data/processed/manifest.csv --output-dir models/

4) Değerlendirme:

   - scripts/run_eval.sh ile değerlendirme çalıştırılabilir (henüz basit iskelet).

5) İpuçları:

   - Çalıştırma izinleri gerekiyorsa: chmod +x scripts/*.sh
   - Daha ileri otomasyon ve deney takibi için plan.md'de tanımlanan adımları izleyin.
