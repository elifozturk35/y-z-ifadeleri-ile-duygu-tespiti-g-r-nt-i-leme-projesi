"""
================================================================================
        YÜZ İFADESİNE GÖRE DUYGU TANIMA - GÖRÜNTÜ İŞLEME PROJESİ
================================================================================
Hazırlayan: Görüntü İşleme Dersi
Tarih: 2025

Bu proje, yüz görüntülerinden duygu tanıma yapmak için bir CNN modeli eğitir.
Duygu sınıfları: Happy (Mutlu), Sad (Üzgün), Angry (Kızgın), Neutral (Nötr)

Gereksinimler:
    - tensorflow
    - numpy
    - pillow
    - mtcnn
    - matplotlib
    - scikit-learn
================================================================================
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TensorFlow ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# MTCNN yüz tespiti
from mtcnn import MTCNN

# ============================================================================
# SABÎT DEĞERLER
# ============================================================================
IMG_SIZE = 48                                    # Görüntü boyutu (48x48)
BATCH_SIZE = 32                                  # Mini-batch boyutu
EPOCHS = 25                                      # Eğitim epoch sayısı
LEARNING_RATE = 0.001                           # Öğrenme oranı

# Duygu sınıfları (4 sınıf)
SINIFLAR = ['angry', 'happy', 'neutral', 'sad']
SINIF_TR = {
    'angry': 'Kızgın',
    'happy': 'Mutlu', 
    'neutral': 'Nötr',
    'sad': 'Üzgün'
}
SINIF_SAYISI = len(SINIFLAR)

# Dataset yolu
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")


# ============================================================================
# YÜZ TESPİTİ SINIFI
# ============================================================================
class YuzTespitci:
    """MTCNN ile yüz tespiti yapan sınıf"""
    
    def __init__(self):
        print("Yüz tespitçisi yükleniyor...")
        self.detector = MTCNN()
        print("Yüz tespitçisi hazır!")
    
    def yuz_bul(self, goruntu):
        """
        Görüntüden yüz bölgesini tespit eder.
        
        Args:
            goruntu: PIL Image veya numpy array
            
        Returns:
            Yüz bölgesi (numpy array) veya None
        """
        # PIL Image'ı numpy array'e çevir
        if isinstance(goruntu, Image.Image):
            goruntu = np.array(goruntu)
        
        # RGB'ye çevir (gerekirse)
        if len(goruntu.shape) == 2:
            goruntu = np.stack([goruntu] * 3, axis=-1)
        elif goruntu.shape[2] == 4:
            goruntu = goruntu[:, :, :3]
        
        # Yüz tespit et
        sonuclar = self.detector.detect_faces(goruntu)
        
        if not sonuclar:
            return None
        
        # En büyük yüzü al
        en_buyuk = max(sonuclar, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = en_buyuk['box']
        
        # Sınırları kontrol et
        x = max(0, x)
        y = max(0, y)
        x2 = min(goruntu.shape[1], x + w)
        y2 = min(goruntu.shape[0], y + h)
        
        # Yüz bölgesini kes
        yuz = goruntu[y:y2, x:x2]
        
        return yuz


# ============================================================================
# VERİ HAZIRLAMA FONKSİYONLARI
# ============================================================================
def goruntu_hazirla(goruntu_yolu, yuz_tespitci=None):
    """
    Tek bir görüntüyü işler: gri tonlama, yeniden boyutlandırma.
    FER2013 görüntüleri zaten yüz içerdiği için yüz tespiti opsiyonel.
    
    Args:
        goruntu_yolu: Görüntü dosyasının yolu
        yuz_tespitci: YuzTespitci nesnesi (opsiyonel)
        
    Returns:
        İşlenmiş görüntü (48x48 gri) veya None
    """
    try:
        # Görüntüyü yükle
        img = Image.open(goruntu_yolu)
        
        # Gri tonlamaya çevir
        img_gri = img.convert('L')
        
        # 48x48 boyutuna getir
        img_boyutlu = img_gri.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Numpy array'e çevir ve normalize et
        img_array = np.array(img_boyutlu, dtype=np.float32) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Hata ({goruntu_yolu}): {e}")
        return None


def dataset_yukle(klasor_yolu, yuz_tespitci=None):
    """
    Belirtilen klasördeki tüm görüntüleri yükler ve işler.
    
    Args:
        klasor_yolu: Dataset klasörü (train veya test)
        yuz_tespitci: YuzTespitci nesnesi (opsiyonel)
        
    Returns:
        X: Görüntü verileri (N, 48, 48, 1)
        y: Etiketler (N,)
    """
    X = []
    y = []
    
    print(f"\nDataset yükleniyor: {klasor_yolu}")
    print("-" * 50)
    
    for sinif_idx, sinif_adi in enumerate(SINIFLAR):
        sinif_yolu = os.path.join(klasor_yolu, sinif_adi)
        
        if not os.path.exists(sinif_yolu):
            print(f"  [!] Klasör bulunamadı: {sinif_yolu}")
            continue
        
        dosyalar = [f for f in os.listdir(sinif_yolu) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        yuklenen = 0
        atlanan = 0
        
        for dosya in dosyalar:
            dosya_yolu = os.path.join(sinif_yolu, dosya)
            
            # Görüntüyü işle
            goruntu = goruntu_hazirla(dosya_yolu, yuz_tespitci)
            
            if goruntu is not None:
                X.append(goruntu)
                y.append(sinif_idx)
                yuklenen += 1
            else:
                atlanan += 1
        
        print(f"  {SINIF_TR[sinif_adi]:8s}: {yuklenen:4d} yüklendi, {atlanan:3d} atlandı")
    
    # Numpy array'e çevir
    X = np.array(X)
    y = np.array(y)
    
    # Kanal boyutunu ekle (48, 48) -> (48, 48, 1)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    print("-" * 50)
    print(f"Toplam: {len(X)} görüntü yüklendi")
    
    return X, y


# ============================================================================
# MODEL MİMARİSİ
# ============================================================================
def model_olustur():
    """
    CNN modelini oluşturur.
    
    Mimari:
        - 3 Convolutional blok (Conv2D + BatchNorm + MaxPool + Dropout)
        - 2 Fully Connected katman
        - Softmax çıkış katmanı
        
    Returns:
        Derlenmiş Keras modeli
    """
    model = keras.Sequential([
        # Giriş katmanı
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # ========== 1. Convolutional Blok ==========
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # ========== 2. Convolutional Blok ==========
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # ========== 3. Convolutional Blok ==========
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # ========== Fully Connected Katmanlar ==========
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # ========== Çıkış Katmanı ==========
        layers.Dense(SINIF_SAYISI, activation='softmax')
    ])
    
    # Modeli derle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# EĞİTİM GRAFİKLERİ
# ============================================================================
def egitim_grafigi_ciz(history):
    """
    Eğitim sürecinin grafiklerini çizer.
    
    Args:
        history: model.fit() çıktısı
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Doğruluk Grafiği =====
    axes[0].plot(history.history['accuracy'], 'b-', label='Eğitim', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], 'r-', label='Doğrulama', linewidth=2)
    axes[0].set_title('Model Doğruluğu', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Doğruluk')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ===== Kayıp Grafiği =====
    axes[1].plot(history.history['loss'], 'b-', label='Eğitim', linewidth=2)
    axes[1].plot(history.history['val_loss'], 'r-', label='Doğrulama', linewidth=2)
    axes[1].set_title('Model Kaybı (Loss)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Kayıp')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('egitim_grafigi.png', dpi=150)
    plt.close()  # GUI penceresi açmadan kapat
    print("Grafik kaydedildi: egitim_grafigi.png")


def confusion_matrix_ciz(y_true, y_pred):
    """
    Confusion matrix'i görselleştirir.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    """
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[SINIF_TR[s] for s in SINIFLAR],
                yticklabels=[SINIF_TR[s] for s in SINIFLAR])
    plt.title('Confusion Matrix (Karışıklık Matrisi)', fontsize=14, fontweight='bold')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()  # GUI penceresi açmadan kapat
    print("Kaydedildi: confusion_matrix.png")
    
    # Classification report yazdır
    print("\n" + "=" * 60)
    print("SINIFLANDIRMA RAPORU")
    print("=" * 60)
    sinif_isimleri = [SINIF_TR[s] for s in SINIFLAR]
    print(classification_report(y_true, y_pred, target_names=sinif_isimleri))


def ornek_tahminler_goster(model, X_test, y_test, adet=12):
    """
    Test setinden örnek tahminler gösterir.
    
    Args:
        model: Eğitilmiş model
        X_test: Test görüntüleri
        y_test: Test etiketleri
        adet: Gösterilecek örnek sayısı
    """
    # Rastgele örnekler seç
    indices = np.random.choice(len(X_test), min(adet, len(X_test)), replace=False)
    
    # Tahmin yap
    tahminler = model.predict(X_test[indices], verbose=0)
    tahmin_siniflar = np.argmax(tahminler, axis=1)
    
    # Görselleştir
    satir = 3
    sutun = 4
    fig, axes = plt.subplots(satir, sutun, figsize=(14, 10))
    
    for i, ax in enumerate(axes.flat):
        if i >= len(indices):
            ax.axis('off')
            continue
        
        idx = indices[i]
        gercek = y_test[idx]
        tahmin = tahmin_siniflar[i]
        guven = tahminler[i][tahmin] * 100
        
        # Görüntüyü göster
        ax.imshow(X_test[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        
        # Başlık (doğru/yanlış renklendirme)
        renk = 'green' if gercek == tahmin else 'red'
        baslik = f"Gerçek: {SINIF_TR[SINIFLAR[gercek]]}\n"
        baslik += f"Tahmin: {SINIF_TR[SINIFLAR[tahmin]]} (%{guven:.1f})"
        ax.set_title(baslik, fontsize=10, color=renk)
        ax.axis('off')
    
    plt.suptitle('Örnek Tahminler (Yeşil=Doğru, Kırmızı=Yanlış)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ornek_tahminler.png', dpi=150)
    plt.close()  # GUI penceresi açmadan kapat
    print("Kaydedildi: ornek_tahminler.png")


# ============================================================================
# ANA PROGRAM
# ============================================================================
def main():
    print("=" * 70)
    print("     YÜZ İFADESİNE GÖRE DUYGU TANIMA - EĞİTİM PROGRAMI")
    print("=" * 70)
    print(f"  Sınıflar: {', '.join([SINIF_TR[s] for s in SINIFLAR])}")
    print(f"  Görüntü Boyutu: {IMG_SIZE}x{IMG_SIZE} (Gri)")
    print(f"  Epoch: {EPOCHS}, Batch: {BATCH_SIZE}")
    print("=" * 70)
    
    # ----- 1. Dataset Kontrolü -----
    if not os.path.exists(DATASET_PATH):
        print(f"\n[!] HATA: '{DATASET_PATH}' klasörü bulunamadı!")
        print("\nDataset yapısı şu şekilde olmalı:")
        print("  dataset/")
        print("    train/")
        print("      happy/")
        print("      sad/")
        print("      angry/")
        print("      neutral/")
        print("    test/")
        print("      happy/")
        print("      sad/")
        print("      angry/")
        print("      neutral/")
        return
    
    # ----- 2. Eğitim Verisini Yükle -----
    # FER2013 görüntüleri zaten yüz içeriyor, yüz tespiti gerekmiyor
    X_train, y_train = dataset_yukle(TRAIN_PATH)
    
    if len(X_train) == 0:
        print("\n[!] HATA: Eğitim verisi yüklenemedi!")
        return
    
    # ----- 3. Test Verisini Yükle -----
    X_test, y_test = dataset_yukle(TEST_PATH)
    
    if len(X_test) == 0:
        print("\n[!] HATA: Test verisi yüklenemedi!")
        return
    
    # Veri özeti
    print("\n" + "=" * 50)
    print("VERİ ÖZETİ")
    print("=" * 50)
    print(f"  Eğitim seti  : {X_train.shape[0]:,} görüntü")
    print(f"  Test seti    : {X_test.shape[0]:,} görüntü")
    print(f"  Görüntü boyutu: {X_train.shape[1:3]}")
    
    # Sınıf dağılımı
    print("\n  Eğitim seti sınıf dağılımı:")
    for i, sinif in enumerate(SINIFLAR):
        sayi = np.sum(y_train == i)
        print(f"    {SINIF_TR[sinif]:8s}: {sayi:4d}")
    
    # ----- 5. Model Oluştur -----
    print("\n" + "=" * 50)
    print("MODEL MİMARİSİ")
    print("=" * 50)
    
    model = model_olustur()
    model.summary()
    
    # ----- 6. Modeli Eğit -----
    print("\n" + "=" * 50)
    print("EĞİTİM BAŞLIYOR")
    print("=" * 50)
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Learning rate düşürme
    lr_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),  # Test setini doğrulama için kullan
        callbacks=[early_stop, lr_reduce],
        verbose=1
    )
    
    # ----- 7. Eğitim Grafiği -----
    print("\n" + "=" * 50)
    print("EĞİTİM GRAFİĞİ")
    print("=" * 50)
    egitim_grafigi_ciz(history)
    
    # ----- 8. Test Seti Değerlendirmesi -----
    print("\n" + "=" * 50)
    print("TEST SETİ DEĞERLENDİRMESİ")
    print("=" * 50)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Kaybı    : {test_loss:.4f}")
    print(f"  Test Doğruluğu: %{test_acc * 100:.2f}")
    
    # Tahminler
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # ----- 9. Confusion Matrix -----
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    confusion_matrix_ciz(y_test, y_pred)
    
    # ----- 10. Örnek Tahminler -----
    print("\n" + "=" * 50)
    print("ÖRNEK TAHMİNLER")
    print("=" * 50)
    ornek_tahminler_goster(model, X_test, y_test, adet=12)
    
    # ----- 11. Modeli Kaydet -----
    model.save('duygu_tanima_modeli.keras')
    print("\n" + "=" * 50)
    print("Model kaydedildi: duygu_tanima_modeli.keras")
    print("=" * 50)
    
    print("\n✅ Eğitim tamamlandı!")


# ============================================================================
# ÇALIŞTIR
# ============================================================================
if __name__ == "__main__":
    main()
