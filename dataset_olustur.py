"""
================================================================================
        ÖRNEK DATASET OLUŞTURUCU
================================================================================
Bu script, FER-2013 veya benzeri bir datasetten 4 sınıflı (happy, sad, angry, 
neutral) bir dataset yapısı oluşturmanıza yardımcı olur.

Eğer kendi görüntülerinizi kullanacaksanız:
1. dataset/ klasörünü oluşturun
2. İçine train/ ve test/ klasörleri oluşturun  
3. Her birinin içine happy/, sad/, angry/, neutral/ klasörleri oluşturun
4. Her klasöre ilgili duyguyu gösteren yüz fotoğrafları koyun

Örnek yapı:
    dataset/
    ├── train/
    │   ├── happy/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   ├── sad/
    │   ├── angry/
    │   └── neutral/
    └── test/
        ├── happy/
        ├── sad/
        ├── angry/
        └── neutral/
================================================================================
"""

import os

def klasor_yapisi_olustur(ana_klasor="dataset"):
    """Dataset klasör yapısını oluşturur"""
    
    siniflar = ['happy', 'sad', 'angry', 'neutral']
    setler = ['train', 'test']
    
    for set_adi in setler:
        for sinif in siniflar:
            klasor_yolu = os.path.join(ana_klasor, set_adi, sinif)
            os.makedirs(klasor_yolu, exist_ok=True)
            
            # Bilgilendirme dosyası oluştur
            bilgi_dosyasi = os.path.join(klasor_yolu, "BURAYA_RESIM_KOYUN.txt")
            with open(bilgi_dosyasi, 'w', encoding='utf-8') as f:
                f.write(f"Bu klasöre '{sinif.upper()}' duygusunu gösteren yüz fotoğrafları koyun.\n")
                f.write("Desteklenen formatlar: .jpg, .jpeg, .png, .bmp\n")
    
    print(f"✅ Dataset klasör yapısı oluşturuldu: {ana_klasor}/")
    print("\nYapı:")
    print(f"  {ana_klasor}/")
    print("  ├── train/")
    for sinif in siniflar:
        print(f"  │   ├── {sinif}/")
    print("  └── test/")
    for i, sinif in enumerate(siniflar):
        if i < len(siniflar) - 1:
            print(f"      ├── {sinif}/")
        else:
            print(f"      └── {sinif}/")
    
    print("\n📌 Her klasöre ilgili duyguyu gösteren yüz fotoğrafları koyun.")
    print("   Örnek: dataset/train/happy/ içine mutlu yüz fotoğrafları")


if __name__ == "__main__":
    klasor_yapisi_olustur()
