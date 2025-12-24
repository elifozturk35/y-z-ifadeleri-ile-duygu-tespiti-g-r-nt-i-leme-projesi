# 🎭 Yüz İfadesine Göre Duygu Tanıma Projesi

Bu proje, CNN (Convolutional Neural Network) kullanarak yüz görüntülerinden duygu tanıma yapar.

## 📋 Duygu Sınıfları (4 Sınıf)

- 😊 **Happy** (Mutlu)
- 😢 **Sad** (Üzgün)
- 😠 **Angry** (Kızgın)
- 😐 **Neutral** (Nötr)

## 📁 Dosya Yapısı

```
görüntü_işleme_proje/
├── duygu_tanima.py      # Ana eğitim programı
├── duygu_tahmin.py      # GUI tahmin programı
├── dataset_olustur.py   # Dataset klasör yapısı oluşturucu
├── README.md            # Bu dosya
└── dataset/
    ├── train/
    │   ├── happy/
    │   ├── sad/
    │   ├── angry/
    │   └── neutral/
    └── test/
        ├── happy/
        ├── sad/
        ├── angry/
        └── neutral/
```

## 🛠️ Kurulum

### Gereksinimler

- Python 3.8 veya üzeri

### Kütüphaneler

```bash
pip install tensorflow numpy pillow mtcnn matplotlib scikit-learn seaborn
```

## 🚀 Kullanım

### 1. Dataset Yapısını Oluşturun

```bash
python dataset_olustur.py
```

### 2. Dataset'e Görüntü Ekleyin

Her klasöre ilgili duyguyu gösteren yüz fotoğrafları koyun.

### 3. Modeli Eğitin

```bash
python duygu_tanima.py
```

### 4. Tahmin Yapın (GUI)

```bash
python duygu_tahmin.py
```

## 🧠 Model Mimarisi

```
Input (48x48x1) - Gri tonlamalı görüntü
    │
    ▼
Conv2D(32) + BatchNorm + ReLU + MaxPool + Dropout
    │
    ▼
Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout
    │
    ▼
Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout
    │
    ▼
Flatten + Dense(256) + Dropout
    │
    ▼
Dense(128) + Dropout
    │
    ▼
Dense(4) + Softmax → Çıkış (4 duygu sınıfı)
```

## 📊 Çıktılar

Eğitim sonrasında şu dosyalar oluşturulur:
- `duygu_tanima_modeli.keras` - Eğitilmiş model
- `egitim_grafigi.png` - Accuracy ve Loss grafikleri
- `confusion_matrix.png` - Karışıklık matrisi
- `ornek_tahminler.png` - Örnek tahmin görselleri

## ⚙️ Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| IMG_SIZE | 48 | Görüntü boyutu |
| BATCH_SIZE | 32 | Mini-batch boyutu |
| EPOCHS | 25 | Eğitim epoch sayısı |
| LEARNING_RATE | 0.001 | Öğrenme oranı |

## ⚙️ Teknik Detaylar

- **Yüz Tespiti:** MTCNN (Multi-task Cascaded Convolutional Networks)
- **Derin Öğrenme:** TensorFlow/Keras tabanlı CNN modeli
- **Ön İşleme:** Gri tonlama + 48x48 yeniden boyutlandırma

## 📝 Notlar

- Yüz tespit edilemeyen görüntüler otomatik olarak atlanır
- Early stopping ve learning rate scheduling uygulanır
- Confusion matrix ve classification report otomatik üretilir

Bu proje eğitim amaçlı oluşturulmuştur.

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilir. Büyük değişiklikler için önce bir issue açınız.

---

**Geliştirici:** Görüntü İşleme Projesi
**Tarih:** 2025
