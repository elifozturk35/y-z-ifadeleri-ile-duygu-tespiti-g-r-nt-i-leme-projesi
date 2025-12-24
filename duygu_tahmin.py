"""
================================================================================
        DUYGU TANIMA - TAHMİN PROGRAMI (GUI)
================================================================================
Eğitilmiş model ile yeni görüntülerden duygu tahmini yapar.
================================================================================
"""

# Dosya ve dizin işlemleri için os modülü
import os

# Sayısal hesaplamalar ve dizi işlemleri için NumPy
import numpy as np

# Görüntü işleme ve GUI'de görüntü gösterimi için PIL (Pillow) kütüphanesi
from PIL import Image, ImageDraw, ImageFont, ImageTk

# GUI (Grafiksel Kullanıcı Arayüzü) oluşturmak için Tkinter
import tkinter as tk
from tkinter import filedialog, messagebox  # Dosya seçme ve mesaj kutuları

# Uyarı mesajlarını gizlemek için
import warnings
warnings.filterwarnings('ignore')

# TensorFlow uyarı mesajlarını susturmak için ortam değişkenleri
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow log seviyesini ayarla (sadece hataları göster)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN optimizasyonlarını devre dışı bırak

# Derin öğrenme modeli için TensorFlow ve Keras
import tensorflow as tf
from tensorflow import keras

# Yüz tespiti için MTCNN (Multi-task Cascaded Convolutional Networks)
from mtcnn import MTCNN

# ============================================================================
# SABİT DEĞERLER
# ============================================================================
IMG_SIZE = 48  # Model girdi boyutu (48x48 piksel gri tonlamalı görüntü)
SINIFLAR = ['angry', 'happy', 'neutral', 'sad']  # Modelin tahmin edebileceği duygu sınıfları (İngilizce)
SINIF_TR = {'angry': 'Kızgın', 'happy': 'Mutlu', 'neutral': 'Nötr', 'sad': 'Üzgün'}  # Türkçe karşılıkları
EMOJI = {'angry': '😠', 'happy': '😊', 'neutral': '😐', 'sad': '😢'}  # Her duygu için emoji
RENKLER = {'angry': '#e74c3c', 'happy': '#f1c40f', 'neutral': '#95a5a6', 'sad': '#3498db'}  # GUI renk kodları


# ============================================================================
# TAHMİN SINIFI
# ============================================================================
class DuyguTahminEdici:
    """Eğitilmiş model ile duygu tahmini yapar"""
    
    def __init__(self):
        """Sınıf başlatıcı - değişkenleri None/False olarak tanımla"""
        self.model = None           # Keras duygu tanıma modeli
        self.yuz_tespitci = None    # MTCNN yüz tespit modeli
        self.hazir = False          # Sistem kullanıma hazır mı?
    
    def yukle(self, model_yolu='duygu_tanima_modeli.keras'):
        """Model ve yüz tespitçisini yükler"""
        try:
            # Eğitilmiş Keras modelini diskten yükle
            print("Model yükleniyor...")
            self.model = keras.models.load_model(model_yolu)
            
            # MTCNN yüz tespitçisini başlat (görüntüdeki yüzleri bulmak için)
            print("Yüz tespitçisi yükleniyor...")
            self.yuz_tespitci = MTCNN()
            
            # Her şey başarılı, sistem hazır
            self.hazir = True
            print("Sistem hazır!")
            return True
        except Exception as e:
            # Hata durumunda kullanıcıya bilgi ver
            print(f"Yükleme hatası: {e}")
            return False
    
    def yuz_bul(self, goruntu):
        """Görüntüden yüz bölgesini bulur"""
        # PIL Image ise NumPy dizisine dönüştür (MTCNN NumPy dizisi bekler)
        if isinstance(goruntu, Image.Image):
            goruntu = np.array(goruntu.convert('RGB'))
        
        # MTCNN ile görüntüdeki tüm yüzleri tespit et
        sonuclar = self.yuz_tespitci.detect_faces(goruntu)
        
        # Yüz bulunamadıysa None döndür
        if not sonuclar:
            return None, None
        
        # Birden fazla yüz varsa en büyük olanı seç (ana özne olması muhtemel)
        # box[2] = genişlik, box[3] = yükseklik, çarpımı = alan
        en_buyuk = max(sonuclar, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = en_buyuk['box']  # Yüz bölgesinin koordinatları
        
        # Koordinatların görüntü sınırları içinde kalmasını sağla
        x = max(0, x)                           # Sol kenar negatif olamaz
        y = max(0, y)                           # Üst kenar negatif olamaz
        x2 = min(goruntu.shape[1], x + w)       # Sağ kenar görüntü genişliğini aşamaz
        y2 = min(goruntu.shape[0], y + h)       # Alt kenar görüntü yüksekliğini aşamaz
        
        # Yüz bölgesini kes ve koordinatları döndür
        yuz = goruntu[y:y2, x:x2]               # NumPy dilimleme ile yüz bölgesini al
        kutu = (x, y, x2, y2)                   # Çerçeve çizmek için koordinatlar
        
        return yuz, kutu
    
    def tahmin_et(self, goruntu_yolu):
        """
        Görüntüden duygu tahmini yapar
        
        Returns:
            dict: {'sinif': str, 'tr': str, 'guven': float, 'tum_olasiliklar': dict, 'kutu': tuple}
        """
        # Model hazır değilse işlem yapma
        if not self.hazir:
            return None
        
        # Görüntüyü diskten yükle ve RGB formatına dönüştür
        img = Image.open(goruntu_yolu).convert('RGB')
        img_array = np.array(img)  # NumPy dizisine çevir
        
        # Görüntüde yüz ara
        yuz, kutu = self.yuz_bul(img_array)
        
        # Yüz bulunamadıysa None döndür
        if yuz is None:
            return None
        
        # ===== YÜZ ÖN İŞLEME (Model girişi için hazırlık) =====
        yuz_img = Image.fromarray(yuz)                                    # NumPy -> PIL Image
        yuz_gri = yuz_img.convert('L')                                    # Gri tonlamaya çevir (model gri görüntü bekler)
        yuz_boyutlu = yuz_gri.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)  # 48x48'e boyutlandır
        yuz_array = np.array(yuz_boyutlu, dtype=np.float32) / 255.0       # Normalizasyon: [0-255] -> [0-1]
        yuz_input = yuz_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)           # Model girdi şekli: (1, 48, 48, 1)
        
        # ===== MODEL TAHMİNİ =====
        olasiliklar = self.model.predict(yuz_input, verbose=0)[0]         # 4 sınıf için olasılık değerleri
        sinif_idx = np.argmax(olasiliklar)                                # En yüksek olasılıklı sınıfın indeksi
        sinif = SINIFLAR[sinif_idx]                                       # İndeksten sınıf adına çevir
        
        # Sonuçları sözlük olarak döndür
        return {
            'sinif': sinif,                                               # Tahmin edilen sınıf (İngilizce)
            'tr': SINIF_TR[sinif],                                        # Türkçe karşılığı
            'guven': float(olasiliklar[sinif_idx]),                       # Güven skoru (0-1 arası)
            'tum_olasiliklar': {SINIFLAR[i]: float(olasiliklar[i]) for i in range(len(SINIFLAR))},  # Tüm sınıfların olasılıkları
            'kutu': kutu                                                  # Yüz çerçevesi koordinatları
        }


# ============================================================================
# GUI UYGULAMASI
# ============================================================================
class DuyguTanimaApp:
    """Tkinter GUI uygulaması"""
    
    def __init__(self, root):
        """GUI uygulamasını başlat ve yapılandır"""
        self.root = root                              # Ana pencere referansı
        self.root.title("🎭 Duygu Tanıma - Tahmin")   # Pencere başlığı
        self.root.geometry("950x650")                 # Pencere boyutu (genişlik x yükseklik)
        self.root.configure(bg='#1a1a2e')             # Arka plan rengi (koyu tema)
        
        self.tahminedici = DuyguTahminEdici()         # Tahmin sınıfı örneği oluştur
        self.sonuc = None                             # Son tahmin sonucunu sakla
        
        self._ui_olustur()                            # Arayüz elemanlarını oluştur
        self._model_yukle()                           # Modeli arka planda yükle
    
    def _model_yukle(self):
        """Modeli arka planda yükler (UI donmasını önlemek için)"""
        import threading  # Çoklu iş parçacığı için
        
        def yukle():
            """Arka plan iş parçacığında çalışacak fonksiyon"""
            if self.tahminedici.yukle():
                # Model başarıyla yüklendi - ana iş parçacığında UI güncelle
                self.root.after(0, lambda: self.durum.config(text="✅ Model hazır!", fg="#4ecca3"))
            else:
                # Model yüklenemedi - hata mesajı göster
                self.root.after(0, lambda: self.durum.config(
                    text="❌ Model bulunamadı! Önce eğitim yapın.", fg="#e74c3c"))
        
        # Daemon thread olarak başlat (ana program kapanınca otomatik sonlanır)
        threading.Thread(target=yukle, daemon=True).start()
    
    def _ui_olustur(self):
        """Arayüzü oluşturur - tüm görsel elemanları yerleştirir"""
        
        # ===== BAŞLIK ÇUBUĞU =====
        baslik = tk.Frame(self.root, bg='#16213e', height=60)  # Üst başlık çerçevesi
        baslik.pack(fill='x')                                   # Yatayda tam genişlet
        baslik.pack_propagate(False)                            # Sabit yükseklik koru
        tk.Label(baslik, text="🎭 Yüz İfadesine Göre Duygu Tanıma", 
                 font=('Segoe UI', 18, 'bold'), bg='#16213e', fg='#e94560').pack(expand=True)
        
        # ===== ANA İÇERİK ALANI =====
        ana = tk.Frame(self.root, bg='#1a1a2e')                 # Ana içerik çerçevesi
        ana.pack(fill='both', expand=True, padx=15, pady=15)    # Her yöne genişle
        
        # ===== SOL PANEL - GÖRÜNTÜ ALANI =====
        sol = tk.Frame(ana, bg='#16213e')                       # Sol panel çerçevesi
        sol.pack(side='left', fill='both', expand=True, padx=(0, 8))
        
        tk.Label(sol, text="📷 Görüntü", font=('Segoe UI', 12, 'bold'),
                 bg='#16213e', fg='white').pack(pady=8)         # Bölüm başlığı
        
        # Görüntünün gösterileceği etiket (Label widget'ı resim tutabilir)
        self.goruntu_lbl = tk.Label(sol, bg='#0f0f1a')
        self.goruntu_lbl.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        
        # Resim seçme butonu - tıklanınca _resim_sec fonksiyonunu çağırır
        tk.Button(sol, text="🖼️ Resim Seç ve Tahmin Et", font=('Segoe UI', 12, 'bold'),
                  bg='#4ecca3', fg='#1a1a2e', height=2, relief='flat',
                  command=self._resim_sec).pack(fill='x', padx=10, pady=10)
        
        # ===== SAĞ PANEL - SONUÇ ALANI =====
        sag = tk.Frame(ana, bg='#16213e', width=300)           # Sabit genişlikli sağ panel
        sag.pack(side='right', fill='y')                        # Dikeyde genişle
        sag.pack_propagate(False)                               # Sabit genişlik koru
        
        tk.Label(sag, text="📊 Tahmin Sonucu", font=('Segoe UI', 12, 'bold'),
                 bg='#16213e', fg='white').pack(pady=12)        # Bölüm başlığı
        
        # Tespit edilen duygunun emojisini büyük göster
        self.emoji_lbl = tk.Label(sag, text="🎭", font=('Segoe UI Emoji', 50), bg='#16213e')
        self.emoji_lbl.pack()
        
        # Duygu adını göster (örn: "Mutlu", "Kızgın")
        self.sonuc_lbl = tk.Label(sag, text="Resim Yükleyin", font=('Segoe UI', 18, 'bold'),
                                   bg='#16213e', fg='#f39c12')
        self.sonuc_lbl.pack()
        
        # Güven yüzdesini göster (örn: "Güven: %95.2")
        self.guven_lbl = tk.Label(sag, text="", font=('Segoe UI', 12),
                                   bg='#16213e', fg='#aaa')
        self.guven_lbl.pack(pady=(5, 15))
        
        # ===== OLASILIK BARLARI (Her duygu için görsel çubuk) =====
        self.bar_frame = tk.Frame(sag, bg='#16213e')           # Barları içeren çerçeve
        self.bar_frame.pack(fill='x', padx=15)
        
        self.barlar = {}  # Her sınıf için bar ve yüzde label'ı sakla
        
        # Her duygu sınıfı için bir satır oluştur
        for sinif in SINIFLAR:
            satir = tk.Frame(self.bar_frame, bg='#16213e')     # Tek satırlık çerçeve
            satir.pack(fill='x', pady=4)
            
            # Duygu adı etiketi (emoji + Türkçe ad)
            tk.Label(satir, text=f"{EMOJI[sinif]} {SINIF_TR[sinif]}", 
                     font=('Segoe UI', 10), bg='#16213e', fg='white',
                     width=10, anchor='w').pack(side='left')
            
            # Bar arka planı (koyu renk)
            bar_bg = tk.Frame(satir, bg='#0f0f1a', height=18, width=120)
            bar_bg.pack(side='left', padx=5)
            bar_bg.pack_propagate(False)                        # Sabit boyut
            
            # Doluluk barı (olasılığa göre genişliği değişir)
            bar = tk.Frame(bar_bg, bg=RENKLER[sinif], height=18)
            bar.place(x=0, y=0, width=0)                        # Başlangıçta genişlik 0
            
            # Yüzde değeri etiketi
            yuzde = tk.Label(satir, text="0%", font=('Segoe UI', 10, 'bold'),
                              bg='#16213e', fg='#666', width=6)
            yuzde.pack(side='right')
            
            # Bar ve yüzde etiketini sözlükte sakla (sonra güncellemek için)
            self.barlar[sinif] = (bar, yuzde)
        
        # ===== DURUM ÇUBUĞU (Alt bilgi) =====
        self.durum = tk.Label(sag, text="🔄 Model yükleniyor...", font=('Segoe UI', 10),
                               bg='#16213e', fg='#f39c12')
        self.durum.pack(side='bottom', pady=12)                # En alta yerleştir
    
    def _resim_sec(self):
        """Kullanıcıdan resim seçmesini ister ve duygu tahmini yapar"""
        
        # Model hazır değilse uyarı göster
        if not self.tahminedici.hazir:
            messagebox.showwarning("Uyarı", "Model henüz hazır değil!")
            return
        
        # Dosya seçme penceresi aç (sadece resim dosyaları göster)
        dosya = filedialog.askopenfilename(
            title="Resim Seç",
            filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        # Kullanıcı iptal ettiyse çık
        if not dosya:
            return
        
        # Analiz başladığını göster
        self.durum.config(text="🔍 Analiz yapılıyor...", fg="#f39c12")
        self.root.update()  # UI'ı hemen güncelle
        
        # Seçilen görüntüden duygu tahmini yap
        sonuc = self.tahminedici.tahmin_et(dosya)
        
        # Yüz bulunamadıysa hata mesajı göster
        if sonuc is None:
            self.durum.config(text="⚠️ Yüz tespit edilemedi!", fg="#e74c3c")
            return
        
        self.sonuc = sonuc  # Sonucu sakla
        
        # ===== GÖRÜNTÜ ÜZERİNE ÇİZİM =====
        img = Image.open(dosya).convert('RGB')                  # Görüntüyü tekrar yükle
        draw = ImageDraw.Draw(img)                              # Çizim nesnesi oluştur
        
        # Yüz çerçevesi çiz
        x1, y1, x2, y2 = sonuc['kutu']                          # Yüz koordinatları
        renk = RENKLER[sonuc['sinif']]                          # Duyguya göre renk seç
        draw.rectangle([x1, y1, x2, y2], outline=renk, width=3) # Dikdörtgen çiz
        
        # Metin için font yükle
        try:
            font = ImageFont.truetype("segoeui.ttf", 16)        # Segoe UI fontu dene
        except:
            font = ImageFont.load_default()                      # Yoksa varsayılan font
        
        # Yüz üstüne etiket yaz (emoji + duygu + yüzde)
        etiket = f"{EMOJI[sonuc['sinif']]} {sonuc['tr']} %{sonuc['guven']*100:.0f}"
        bbox = draw.textbbox((x1, y1-24), etiket, font=font)    # Metin boyutunu hesapla
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=renk)  # Arka plan kutusu
        draw.text((x1, y1-24), etiket, fill='white', font=font) # Metni yaz
        
        # Görüntüyü GUI'ye sığacak şekilde küçült ve göster
        img.thumbnail((500, 400), Image.Resampling.LANCZOS)     # En-boy oranını koru
        photo = ImageTk.PhotoImage(img)                          # Tkinter uyumlu formata çevir
        self.goruntu_lbl.config(image=photo)                     # Label'a ata
        self.goruntu_lbl.image = photo                           # Referansı tut (garbage collection önleme)
        
        # ===== SONUÇ PANELİNİ GÜNCELLE =====
        self.emoji_lbl.config(text=EMOJI[sonuc['sinif']])        # Emoji güncelle
        self.sonuc_lbl.config(text=sonuc['tr'], fg=renk)         # Duygu adı güncelle
        self.guven_lbl.config(text=f"Güven: %{sonuc['guven']*100:.1f}")  # Güven yüzdesi
        
        # Her duygu için olasılık barını güncelle
        for sinif, (bar, yuzde) in self.barlar.items():
            val = sonuc['tum_olasiliklar'][sinif]                # Bu sınıfın olasılığı
            bar.place(width=int(120 * val))                      # Bar genişliğini ayarla (max 120px)
            yuzde.config(text=f"%{val*100:.1f}", fg=RENKLER[sinif] if val > 0.1 else '#666')  # Yüzde metni
        
        # İşlem tamamlandı mesajı
        self.durum.config(text=f"✅ Tespit: {sonuc['tr']}", fg="#4ecca3")


# ============================================================================
# ÇALIŞTIR
# ============================================================================
def main():
    """Programın ana giriş noktası - GUI'yi başlatır"""
    # Konsola hoş geldin mesajı yazdır
    print("=" * 50)
    print("  🎭 Duygu Tanıma - Tahmin Programı")
    print("=" * 50)
    
    root = tk.Tk()                      # Tkinter ana pencere oluştur
    app = DuyguTanimaApp(root)          # Uygulama sınıfını başlat
    root.mainloop()                     # GUI döngüsünü başlat (pencere kapanana kadar çalışır)


# Bu dosya doğrudan çalıştırılırsa (import edilmezse) main() fonksiyonunu çağır
if __name__ == "__main__":
    main()
