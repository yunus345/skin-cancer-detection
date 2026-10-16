# Dış test seti: ISIC 2018 Task 3 resmi test seti (KENARA AYRILDI, 2026-10-16)

1512 dermoskopi görüntüsü, bizim 7 sınıfımızla aynı etiketler:
nv 909, bkl 217, mel 171, bcc 93, df 44, akiec 43, vasc 35.
Bu klasörde sadece görüntü kimlikleri ve etiketler var (`isic2018_task3_test_labels.csv`,
MD5 `c6b7b4dd8cb825be0a9338a62c73ca9b`); görüntüler (401MB) en sondaki ölçümde indirilecek:
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip

## Neden

Kendi test setimiz HAM10000'den ayrılmış ve birçok deneme boyunca tekrar tekrar bakıldı. Bu set
modellerimizin hiç görmediği, hiçbir seçimde kullanılmamış tek ölçüm olacak. Ayrıca ISIC 2018
yarışmasının resmi ölçüm seti: sonucumuz yarışmadaki skorlarla (ör. ek verisiz 0.845 balanced
accuracy) aynı sette kıyaslanabilir.

## Kurallar

1. **Bir kez kullanılır:** proje kesinleştikten sonra (ensemble, ek veri, eşikler hepsi
   donduktan sonra). Sonuca bakıp geri dönüp ayar yapmak bu seti geçersiz kılar.
2. **Hiçbir veriye karışamaz:** eğitim, validation ya da kendi test setimiz. Kod bunu
   zorluyor: `src/data/validate_manifest.py` manifest'te bu kimliklerden biri varsa hata verir
   (her Kaggle koşusunun başında çalışır); `src/data/external_test.py` ek veri kodunda
   kullanılacak.
3. **Kontrol edildi (2026-10-16):** HAM10000 (10015 görüntü) ile ortak görüntü 0;
   ISIC 2019 eğitim seti (25331 görüntü, ek veri adayı) ile ortak görüntü 0.

## Bilinen sınırlama

Bu set de HAM10000'i hazırlayan Viyana grubundan; tamamen farklı bir hastane değil, ama ayrı
toplanmış bir set. Test etiketlerinde lezyon kimliği olmadığı için, aynı lezyonun başka bir
fotoğrafının HAM10000'de olup olmadığı kontrol edilemiyor.

Lisans: CC-BY-NC 4.0, atıf `ATTRIBUTION.txt`.
