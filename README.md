# YOLO Veri Artırma (Data Augmentation) Aracı  
Bu Python betiği, Albumentations kütüphanesini kullanarak YOLO formatındaki nesne tespit veri setlerini zenginleştirmek için tasarlanmıştır. Görüntülere geometrik dönüşümler, hava durumu efektleri ve kamera bozulmaları eklerken etiket dosyalarını (.txt) otomatik olarak günceller.  
  
🚀 **Özellikler**  

**YOLO Desteği:** Etiketleri otomatik olarak yeniden hesaplar ve sınırlayıcı kutuları (bounding boxes) günceller.  
  
**Gelişmiş Filtreler:**  

+ Geometrik: Yatay çevirme, perspektif değişimi, rotasyon.  
  
+ Hava Durumu: Yağmur, kar ve sis efektleri.  
  
+ Bozulmalar: Hareket bulanıklığı (motion blur), Gauss gürültüsü ve ölü pikseller.  
  
+ Akıllı Filtreleme: min_visibility parametresi ile nesnenin büyük kısmı dışarıda kalırsa etiketi otomatik siler.  
  
+ Çoğaltma: Her bir görselden belirtilen sayıda (AUGMENT_COUNT) farklı varyasyon üretir.  
  
🛠️ **Kurulum**  
  
Gerekli kütüphaneleri yüklemek için:

```
pip install opencv-python albumentations tqdm
```  

📂 **Dosya Yapısı**
Kodun çalışması için veri setinizin aşağıdaki yapıda olması gerekir:  
  
```plaintext
yolo_test_1/
├── images/       # Orijinal görseller (.jpg, .png vb.)
├── labels/       # Orijinal etiketler (.txt)
└── classes.txt   # Sınıf isimleri  
```  

⚙️ **Kullanım**  
  
*INPUT_DIR* ve *OUTPUT_DIR* değişkenlerini kendi klasör yollarınıza göre düzenleyin.  
  
*AUGMENT_COUNT* değerini değiştirerek görsel başına kaç kopya istediğinizi belirleyin.  
  
Betiği çalıştırın:  
  
`python augmentation_script.py`  
  
İşlem tamamlandığında **augmented_data** klasöründe hem orijinal hem de türetilmiş verileri hazır bir şekilde bulacaksınız.  
  
⚠️ **Önemli Notlar**  
  
Orijinal Veriler: Kod, orijinal görselleri ve etiketleri çıktı klasörüne otomatik olarak kopyalar.  
  
Etiketsiz Görseller: Eğer bir görselin etiketi yoksa, artırma işlemine dahil edilmez ancak çıktı klasörüne kopyalanır.  
  
Görünürlük: Varsayılan olarak, bir nesnenin en az %30'u görsel içinde kalıyorsa etiketi korunur.  