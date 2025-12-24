# 🎭 Yuz Ifadeleri ile Duygu Tanima (Facial Emotion Recognition)

Bu projede, bir yüz fotoğrafından kişinin **duygusunu (mutlu, üzgün, kızgın, nötr)** tahmin eden
CNN (Convolutional Neural Network) tabanlı bir duygu tanıma sistemi geliştirilmiştir.

Proje, **Görüntü İşleme** dersi kapsamında hazırlanmıştır.

---

## 📌 Proje Özeti

- Girdi: Yüz fotoğrafı (48x48, gri tonlama)
- Çıktı: Duygu tahmini (olasılık tabanlı)
- Kullanılan yöntem: **Derin Öğrenme (CNN)**
- Öğrenme türü: **Denetimli Öğrenme (Supervised Learning)**

Model, her duygu için bir olasılık üretir ve en yüksek olasılığa sahip duygu sonuç olarak gösterilir.

---

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Açıklama |
|---------|---------|
| Python 3 | Ana programlama dili |
| TensorFlow / Keras | Derin öğrenme modeli |
| CNN | Görüntü sınıflandırma mimarisi |
| NumPy | Sayısal işlemler |
| PIL (Pillow) | Görüntü okuma ve ön işleme |
| MTCNN | Yüz tespiti |
| Matplotlib | Grafik ve sonuç görselleştirme |
| Tkinter | Grafiksel kullanıcı arayüzü |

> **Not:** OpenCV kullanımı planlanmış ancak Windows ortamında yaşanan bağımlılık/DLL sorunları
nedeniyle görüntü işleme aşamasında daha stabil olan **PIL (Pillow)** tercih edilmiştir.

---

## 📂 Veri Seti

- **Adı:** FER2013 (Facial Expression Recognition 2013)
- **Kaynak:** Kaggle
- **Görüntü Boyutu:** 48x48 piksel
- **Renk:** Gri tonlama
- **Duygu Sınıfları:**  
  - 😠 Angry  
  - 😊 Happy  
  - 😐 Neutral  
  - 😢 Sad  

Veri seti %80 eğitim, %20 test olacak şekilde ayrılmıştır.

⚠️ Veri seti **dengesizdir** (mutlu sınıfı daha fazladır). Bu durum modelin bazı sınıflara bias yapmasına
sebep olmaktadır.

---

## 📁 Proje Dosya Yapısı


