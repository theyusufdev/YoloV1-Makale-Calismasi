# YOLOv1 – Makale ve Sıfırdan Gerçekleştirme Çalışması

<img width="1128" height="726" alt="image" src="https://github.com/user-attachments/assets/fc8d7e80-6415-494e-813c-de081339c776" />

Bu proje, Joseph Redmon ve arkadaşları tarafından 2016 yılında yayımlanan **You Only Look Once: Unified, Real-Time Object Detection** makalesinde tanıtılan YOLOv1 mimarisini incelemek ve PyTorch kullanarak sıfırdan gerçekleştirmek amacıyla geliştirilmiştir.

Projenin temel amacı, nesne tespitinin çalışma mantığını yalnızca hazır kütüphaneler üzerinden kullanmak yerine; model mimarisi, veri işleme, hedef kodlama, kayıp fonksiyonu ve tahmin süreçlerini uygulamalı olarak öğrenmektir.

## Model Özellikleri

Proje kapsamında YOLOv1 mimarisi PyTorch kullanılarak gerçekleştirilmiştir.

<img width="1113" height="851" alt="image" src="https://github.com/user-attachments/assets/520091d1-9d75-43ad-86ab-650352e998ee" />

## Model Özellikleri ve Performans İstatistikleri

YOLOv1 mimarisi PyTorch kullanılarak sıfırdan gerçekleştirilmiştir. Modelin mimari analizi sonucunda elde edilen istatistikler aşağıda verilmiştir.

### Mimari Yapılandırma

| Özellik | Değer |
|---|---:|
| Model | YOLOv1 |
| Mimari | CNN – One-Stage Object Detector |
| Giriş boyutu | 448 × 448 × 3 |
| Izgara boyutu (S) | 7 × 7 |
| Bounding Box sayısı (B) | 2 |
| Sınıf sayısı (C) | 20 |
| Çıkış boyutu | 7 × 7 × 30 |
| Convolution katmanları | 24 |
| Fully Connected katmanları | 2 |
| Aktivasyon fonksiyonu | Leaky ReLU |
| Dropout | 0.5 |
| Toplam parametre | 271.703.550 |
| Veri seti | Pascal VOC 2007 |
| Optimizasyon algoritması | SGD |
| Başlangıç öğrenme oranı | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| Gerçekleştirilen eğitim | 10 Epoch |
| Tahmini toplam model boyutu | 1137.03 MB |

### Eğitim Donanımı ve Kısıtlamalar

Modelin 271 milyondan fazla eğitilebilir parametreye sahip olması, eğitim sürecinde yüksek hesaplama ve bellek gereksinimleri oluşturmaktadır.

Kişisel bilgisayarımın donanım kaynakları nedeniyle tam eğitim süreci tamamlanamamıştır. Bunun yerine modelin ilk 10 epoch boyunca gerçekleştirdiği eğitim sonuçları elde edilmiş ve paylaşılmıştır.

Bu çalışma, tam eğitilmiş bir nesne tespit modeli sunmaktan ziyade YOLOv1 mimarisinin teorik olarak incelenmesini ve PyTorch ile sıfırdan uygulanmasını amaçlamaktadır.

### Model Çıktısı

Model, her görüntü için 7 × 7 × 30 boyutunda bir çıktı üretir.

Bu çıktı üç temel bileşenden oluşur:

- 20 sınıf olasılığı
- 2 bounding box tahmini
- Her bounding box için 5 değer: x, y, genişlik, yükseklik ve güven skoru

Böylece her grid hücresi toplam 30 değer üretir.

## Projenin Kapsamı

Bu çalışma kapsamında YOLOv1 mimarisinin temel bileşenleri ele alınmaktadır:

- **Model Mimarisi:** YOLOv1 ağ yapısının PyTorch ile gerçekleştirilmesi.
- **Veri Seti:** Pascal VOC 2007 veri setinin kullanılması.
- **Veri Ön İşleme:** Görüntülerin ve etiketlerin modele uygun hâle getirilmesi.
- **Hedef Kodlama:** Nesne koordinatlarının ızgara (grid) hücreleri üzerinden temsil edilmesi.
- **Kayıp Fonksiyonu:** YOLOv1 makalesinde tanımlanan Sum-Squared Error (SSE) tabanlı kayıp fonksiyonunun uygulanması.
- **Eğitim Süreci:** Modelin eğitimi ve kayıp değerlerinin takip edilmesi.
- **Tahmin ve Görselleştirme:** Model çıktılarının işlenmesi ve tespit sonuçlarının görüntülenmesi.

## Kullanılan Teknolojiler

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pascal VOC 2007

## YOLOv1 Mimarisi

<img width="926" height="424" alt="yolo_mimarisi" src="https://github.com/user-attachments/assets/f901f6bb-296c-4dd7-a0f1-414650157b25" />


YOLOv1, nesne tespitini tek aşamalı bir problem olarak ele alır. Görüntüyü tek bir sinir ağı üzerinden işleyerek nesnelerin sınıflarını ve sınırlayıcı kutu (bounding box) koordinatlarını aynı anda tahmin eder.

Bu projede kullanılan temel yapılandırma:

| Parametre | Değer |
|---|---:|
| Izgara boyutu (S) | 7 |
| Hücre başına bounding box (B) | 2 |
| Sınıf sayısı (C) | 20 |
| Çıktı boyutu | 7 × 7 × 30 |

## Projenin Amacı

Bu proje, derin öğrenme ve bilgisayarlı görü alanındaki temel nesne tespit yaklaşımlarını teorik ve uygulamalı olarak anlamaya yönelik kişisel bir makale inceleme ve uygulama çalışmasıdır.

Geliştirme sürecinde öncelik, mimarinin arkasındaki matematiksel ve algoritmik mantığı anlamak ve bileşenleri hazır çözümlere başvurmadan gerçekleştirmektir.


## Eğitim Süreci ve Donanım Kısıtlamaları

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e6c9b896-7282-439a-a162-103dd1eb483f" />

YOLOv1 mimarisinin eğitimi, modelin parametre sayısı ve Pascal VOC veri setinin boyutu nedeniyle yüksek hesaplama gücü ve uzun eğitim süresi gerektirmektedir.

Modelin tam eğitimini kişisel bilgisayarımda gerçekleştirmek, mevcut donanım kaynakları nedeniyle oldukça uzun süreceğinden tam eğitim süreci tamamlanmamıştır.

Bununla birlikte, uygulamanın çalışma mantığını ve eğitim sürecini doğrulamak amacıyla modelin **ilk 10 epoch boyunca gerçekleştirdiği eğitim sonuçları** elde edilmiş ve paylaşılmıştır.

Bu nedenle proje, tam eğitilmiş bir YOLOv1 modeli olarak değil, mimarinin sıfırdan uygulanması ve eğitim sürecinin deneysel olarak incelenmesi çalışması olarak değerlendirilmelidir.

## Referans

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).

**You Only Look Once: Unified, Real-Time Object Detection.**

IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[Makalenin bağlantısı](https://arxiv.org/abs/1506.02640)

---

<text color="secondary" size="xs">Not: Bu çalışma eğitim ve araştırma amaçlıdır. YOLOv1 mimarisinin özgün makalesi temel alınarak geliştirilmektedir.</text>
