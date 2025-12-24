"""
Basit Duygu Tahmin Testi - GUI olmadan konsol çıktısı
"""
import os
import numpy as np
from PIL import Image
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras

# Sabitler
IMG_SIZE = 48
SINIFLAR = ['angry', 'happy', 'neutral', 'sad']
SINIF_TR = {'angry': 'Kızgın', 'happy': 'Mutlu', 'neutral': 'Nötr', 'sad': 'Üzgün'}
EMOJI = {'angry': '😠', 'happy': '😊', 'neutral': '😐', 'sad': '😢'}

print("=" * 50)
print("  TEST: Duygu Tanıma Modeli")
print("=" * 50)

# Model yükle
print("\nModel yükleniyor...")
model = keras.models.load_model('duygu_tanima_modeli.keras')
print("Model yüklendi!")

# Test klasöründen rastgele görüntü seç
test_path = 'dataset/test'
for sinif in SINIFLAR:
    sinif_path = os.path.join(test_path, sinif)
    dosyalar = os.listdir(sinif_path)
    if dosyalar:
        rastgele = random.choice(dosyalar)
        goruntu_yolu = os.path.join(sinif_path, rastgele)
        
        # Görüntüyü yükle ve işle
        img = Image.open(goruntu_yolu).convert('L')  # Gri tonlama
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # Tahmin yap
        tahmin = model.predict(img_array, verbose=0)
        tahmin_sinif = SINIFLAR[np.argmax(tahmin)]
        guven = np.max(tahmin) * 100
        
        # Sonuç
        dogru = "✓" if tahmin_sinif == sinif else "✗"
        print(f"\n{dogru} Gerçek: {SINIF_TR[sinif]:8s} → Tahmin: {EMOJI[tahmin_sinif]} {SINIF_TR[tahmin_sinif]:8s} (%{guven:.1f})")

print("\n" + "=" * 50)
print("Test tamamlandı!")
