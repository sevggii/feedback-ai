"""
Sentetik sosyal medya yorum verisi oluşturucu.
Gerçekçi örnekler üretir: organik vs spam/destek yorumlar.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path

# Sentetik veri şablonları
GENUINE_COMMENTS = [
    # Pozitif organik yorumlar
    "Bu ürünü 3 aydır kullanıyorum, gerçekten memnunum. Kalitesi çok iyi.",
    "Harika bir deneyim yaşadım! Özellikle müşteri hizmetleri çok ilgiliydi.",
    "Fiyat performans açısından mükemmel. Kesinlikle tavsiye ederim.",
    "Uzun süredir arıyordum böyle bir şey. Sonunda buldum ve çok mutluyum.",
    "Kargo hızlıydı, paketleme güzeldi. Ürün de beklediğim gibi çıktı.",
    "Bu markayı yıllardır takip ediyorum. Her zaman kaliteli ürünler sunuyorlar.",
    "Kızımın doğum günü için aldım, çok beğendi. Teşekkürler!",
    "İlk defa denedim ama çok etkilendim. Devamını da alacağım.",
    "Çok detaylı inceleme yapmışsınız, teşekkürler. Çok faydalı oldu.",
    "Bu fiyata bu kalite gerçekten süper. Arkadaşlarıma da önerdim.",
    
    # Negatif organik yorumlar
    "Maalesef beklediğim gibi çıkmadı. Kalite biraz düşük geldi.",
    "Kargo çok yavaştı, 2 hafta bekledim. Ürün de hasarlı geldi.",
    "Fiyatına göre kalitesi yetersiz. Bir daha almam.",
    "Müşteri hizmetleri hiç yardımcı olmadı. Çok hayal kırıklığı.",
    "Ürün açıklaması ile gerçek arasında fark var. Memnun kalmadım.",
    "Renk beklediğimden farklıydı. Fotoğraflarda daha güzel görünüyordu.",
    "Kullanım kılavuzu eksikti, kurulum zor oldu.",
    "Garanti süresi çok kısa, bu fiyata daha uzun olmalıydı.",
    "Ambalajı açtığımda ürün kırıktı. Değişim süreci de uzun sürdü.",
    "Bu markanın diğer ürünlerini beğeniyordum ama bu hayal kırıklığı.",
    
    # Nötr organik yorumlar
    "Ürün normal, fiyatına göre makul. Özel bir şey yok.",
    "Beklediğim gibiydi. Ne çok iyi ne çok kötü.",
    "Orta kalitede bir ürün. Bu fiyata normal.",
    "Kullanıyorum ama çok da etkilenmedim. İdare eder.",
    "Standart bir ürün. Özel bir beklentim yoktu zaten.",
    "Normal bir deneyim yaşadım. Ne fazla ne eksik.",
    "Beklentilerimi karşıladı. Daha fazlasını beklemiyordum.",
    "Ortalama bir ürün. Bu segment için normal.",
    "Kullanılabilir ama çok da öne çıkmıyor.",
    "Standart kalitede. Bu fiyat bandında normal."
]

SPAM_COMMENTS = [
    # Spam yorumlar
    "🔥🔥🔥 MÜKEMMEL! 👏👏👏 KESINLIKLE ALIN! 💯💯💯",
    "Çok güzel ürün! @arkadaş1 @arkadaş2 @arkadaş3 bakın buraya!",
    "Harika! Link: https://bit.ly/spam-link123",
    "Süper! DM'den yazın detayları vereyim! 📱",
    "Mükemmel! Follow for follow! @kullanici1 @kullanici2",
    "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥",
    "Çok güzel! Çekilişe katılın! @sponsor @marka",
    "Harika ürün! Telegram: @kanal123",
    "Süper! WhatsApp: +90xxx xxx xxxx",
    "Mükemmel! İnstagram: @hesap123",
    "Çok güzel! YouTube: youtube.com/spam",
    "Harika! TikTok: @spam_hesap",
    "Süper! Facebook: facebook.com/spam",
    "Mükemmel! Twitter: @spam_twitter",
    "Çok güzel! LinkedIn: linkedin.com/spam",
    "Harika! Pinterest: pinterest.com/spam",
    "Süper! Snapchat: @spam_snap",
    "Mükemmel! Discord: discord.gg/spam",
    "Çok güzel! Twitch: twitch.tv/spam",
    "Harika! Reddit: reddit.com/r/spam"
]

SUPPORT_COMMENTS = [
    # Destek/niyetli etkileşim yorumları
    "Çok güzel ürün! Kesinlikle tavsiye ederim!",
    "Harika! Herkese öneriyorum!",
    "Mükemmel! Kesinlikle alın!",
    "Süper! Tavsiye ederim!",
    "Çok güzel! Kesinlikle deneyin!",
    "Harika! Herkese söylüyorum!",
    "Mükemmel! Kesinlikle almalısınız!",
    "Süper! Tavsiye ediyorum!",
    "Çok güzel! Deneyin!",
    "Harika! Öneriyorum!",
    "Mükemmel! Alın!",
    "Süper! Tavsiye!",
    "Çok güzel! Deneyin!",
    "Harika! Öneriyorum!",
    "Mükemmel! Alın!",
    "Süper! Tavsiye!",
    "Çok güzel! Deneyin!",
    "Harika! Öneriyorum!",
    "Mükemmel! Alın!",
    "Süper! Tavsiye!"
]

# Platform ve konu şablonları
PLATFORMS = ["instagram", "twitter", "facebook", "tiktok", "youtube", "linkedin"]
TOPICS = ["fashion", "tech", "food", "travel", "beauty", "fitness", "gaming", "music"]


def generate_synthetic_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Sentetik veri üret.
    
    Args:
        num_samples: Üretilecek örnek sayısı
        
    Returns:
        Sentetik veri listesi
    """
    data = []
    
    # Dağılım: %40 organik, %30 spam, %30 destek
    genuine_count = int(num_samples * 0.4)
    spam_count = int(num_samples * 0.3)
    support_count = num_samples - genuine_count - spam_count
    
    # Organik yorumlar
    for i in range(genuine_count):
        text = random.choice(GENUINE_COMMENTS)
        sentiment = "pos" if i < genuine_count // 2 else ("neg" if i < genuine_count * 3 // 4 else "neu")
        
        data.append({
            "text": text,
            "lang": random.choice(["tr", "en", "mixed"]),
            "meta": {
                "platform": random.choice(PLATFORMS),
                "post_topic": random.choice(TOPICS)
            },
            "labels": {
                "sentiment": sentiment,
                "authenticity": "genuine"
            }
        })
    
    # Spam yorumlar
    for i in range(spam_count):
        text = random.choice(SPAM_COMMENTS)
        sentiment = random.choice(["pos", "neu"])  # Spam genelde pozitif veya nötr
        
        data.append({
            "text": text,
            "lang": random.choice(["tr", "en", "mixed"]),
            "meta": {
                "platform": random.choice(PLATFORMS),
                "post_topic": random.choice(TOPICS)
            },
            "labels": {
                "sentiment": sentiment,
                "authenticity": "spam"
            }
        })
    
    # Destek yorumları
    for i in range(support_count):
        text = random.choice(SUPPORT_COMMENTS)
        sentiment = "pos"  # Destek yorumları genelde pozitif
        
        data.append({
            "text": text,
            "lang": random.choice(["tr", "en", "mixed"]),
            "meta": {
                "platform": random.choice(PLATFORMS),
                "post_topic": random.choice(TOPICS)
            },
            "labels": {
                "sentiment": sentiment,
                "authenticity": "support"
            }
        })
    
    # Veriyi karıştır
    random.shuffle(data)
    
    return data


def save_synthetic_data(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Sentetik veriyi JSONL formatında kaydet.
    
    Args:
        data: Veri listesi
        file_path: Kaydetme yolu
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(data)} sentetik örnek {file_path} dosyasına kaydedildi.")


if __name__ == "__main__":
    # Sentetik veri üret ve kaydet
    data = generate_synthetic_data(1000)
    save_synthetic_data(data, Path("src/data/samples/comments_demo.jsonl"))
    
    # İstatistikler
    genuine_count = sum(1 for item in data if item["labels"]["authenticity"] == "genuine")
    spam_count = sum(1 for item in data if item["labels"]["authenticity"] == "spam")
    support_count = sum(1 for item in data if item["labels"]["authenticity"] == "support")
    
    print(f"\n📊 Veri Dağılımı:")
    print(f"  Organik: {genuine_count} (%{genuine_count/len(data)*100:.1f})")
    print(f"  Spam: {spam_count} (%{spam_count/len(data)*100:.1f})")
    print(f"  Destek: {support_count} (%{support_count/len(data)*100:.1f})")
    
    # Sentiment dağılımı
    pos_count = sum(1 for item in data if item["labels"]["sentiment"] == "pos")
    neg_count = sum(1 for item in data if item["labels"]["sentiment"] == "neg")
    neu_count = sum(1 for item in data if item["labels"]["sentiment"] == "neu")
    
    print(f"\n😊 Sentiment Dağılımı:")
    print(f"  Pozitif: {pos_count} (%{pos_count/len(data)*100:.1f})")
    print(f"  Negatif: {neg_count} (%{neg_count/len(data)*100:.1f})")
    print(f"  Nötr: {neu_count} (%{neu_count/len(data)*100:.1f})")
