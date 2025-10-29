"""
Güncellenmiş sentetik veri oluşturucu.
Post analizi + yorum-post ilgisi + spam tespiti.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path

# Post şablonları (farklı konular)
POSTS = {
    "fashion": [
        "Yeni koleksiyonumuz çıktı! Bu sezonun trend renkleri neler?",
        "Sonbahar modası için 5 temel parça önerisi",
        "Sürdürülebilir moda markaları hakkında ne düşünüyorsunuz?",
        "Bu kışın en çok tercih edilen mont modelleri",
        "Minimalist gardırop nasıl oluşturulur?"
    ],
    "tech": [
        "Yeni iPhone 15 özelliklerini incelediniz mi?",
        "AI teknolojilerinin geleceği hakkında görüşleriniz?",
        "Hangi programlama dilini öğrenmeliyim?",
        "Gaming laptop önerileri arıyorum",
        "Blockchain teknolojisi nasıl çalışır?"
    ],
    "food": [
        "Evde pizza yapımı için en iyi tarif",
        "Sağlıklı beslenme önerileri",
        "Türk mutfağının en lezzetli yemekleri",
        "Vegan tarifler paylaşalım",
        "Kahve çekirdeği önerileri"
    ],
    "travel": [
        "Türkiye'de görülmesi gereken yerler",
        "Bütçeli seyahat ipuçları",
        "Yurt dışı seyahat için gerekli belgeler",
        "Kamp yapmak için en güzel yerler",
        "Solo seyahat deneyimleri"
    ],
    "beauty": [
        "Cilt bakım rutini önerileri",
        "Doğal makyaj ürünleri",
        "Saç bakımı için ipuçları",
        "Anti-aging krem önerileri",
        "Makyaj fırçası temizliği"
    ]
}

# İlgili yorumlar (post ile alakalı)
RELEVANT_COMMENTS = {
    "fashion": [
        "Bu renkler gerçekten çok güzel, ben de alacağım",
        "Sonbahar için kahverengi tonları harika",
        "Sürdürülebilir moda çok önemli, teşekkürler",
        "Mont seçimi için hangi markayı önerirsiniz?",
        "Minimalist gardırop için hangi parçalar temel?"
    ],
    "tech": [
        "iPhone 15'in kamerası gerçekten çok iyi",
        "AI gelecekte çok şey değiştirecek",
        "Python ile başlamanızı öneririm",
        "Gaming laptop için RTX kartı şart",
        "Blockchain çok karmaşık ama önemli"
    ],
    "food": [
        "Pizza tarifi için hangi unu kullanmalıyım?",
        "Sağlıklı beslenme için protein önemli",
        "Türk mutfağı dünyanın en iyisi",
        "Vegan tarifler için hangi malzemeler?",
        "Kahve çekirdeği için hangi marka?"
    ],
    "travel": [
        "Kapadokya gerçekten görülmeli",
        "Bütçeli seyahat için hostel önerisi",
        "Pasaport işlemleri çok uzun sürüyor",
        "Kamp için çadır önerisi var mı?",
        "Solo seyahat güvenli mi?"
    ],
    "beauty": [
        "Cilt bakımı için hangi ürünleri kullanmalıyım?",
        "Doğal makyaj ürünleri nereden alınır?",
        "Saç bakımı için hangi şampuan?",
        "Anti-aging krem için hangi marka?",
        "Makyaj fırçası temizliği nasıl yapılır?"
    ]
}

# İlgisiz yorumlar (spam)
IRRELEVANT_COMMENTS = [
    "🔥🔥🔥 MÜKEMMEL! 👏👏👏 KESINLIKLE ALIN! 💯💯💯",
    "Çok güzel ürün! @arkadaş1 @arkadaş2 @arkadaş3 bakın buraya!",
    "Harika! Link: https://bit.ly/spam-link123",
    "Süper! DM'den yazın detayları vereyim! 📱",
    "Mükemmel! Follow for follow! @kullanici1 @kullanici2",
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
    "Harika! Reddit: reddit.com/r/spam",
    "Süper! Tavsiye ederim!"
]

# Destek/niyetli etkileşim yorumları
SUPPORT_COMMENTS = [
    "Çok güzel paylaşım! Kesinlikle tavsiye ederim!",
    "Harika! Herkese öneriyorum!",
    "Mükemmel! Kesinlikle takip edin!",
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

# Platform ve dil şablonları
PLATFORMS = ["instagram", "twitter", "facebook", "tiktok", "youtube", "linkedin"]
LANGUAGES = ["tr", "en", "mixed"]


def generate_post_comment_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Post + yorum verisi üret.
    
    Args:
        num_samples: Üretilecek örnek sayısı
        
    Returns:
        Veri listesi
    """
    data = []
    
    # Dağılım: %50 ilgili, %30 spam, %20 destek
    relevant_count = int(num_samples * 0.5)
    spam_count = int(num_samples * 0.3)
    support_count = num_samples - relevant_count - spam_count
    
    topics = list(POSTS.keys())
    
    # İlgili yorumlar
    for i in range(relevant_count):
        topic = random.choice(topics)
        post = random.choice(POSTS[topic])
        comment = random.choice(RELEVANT_COMMENTS[topic])
        
        # Sentiment: ilgili yorumlar genelde pozitif veya nötr
        sentiment = random.choice(["pos", "neu"]) if random.random() < 0.8 else "neg"
        
        data.append({
            "post": {
                "text": post,
                "topic": topic,
                "platform": random.choice(PLATFORMS),
                "lang": random.choice(LANGUAGES)
            },
            "comment": {
                "text": comment,
                "lang": random.choice(LANGUAGES)
            },
            "meta": {
                "platform": random.choice(PLATFORMS),
                "post_topic": topic
            },
            "labels": {
                "sentiment": sentiment,
                "relevance": "relevant",  # Post ile ilgili
                "authenticity": "genuine"  # Gerçek yorum
            }
        })
    
    # Spam yorumlar (post ile ilgisiz)
    for i in range(spam_count):
        topic = random.choice(topics)
        post = random.choice(POSTS[topic])
        comment = random.choice(IRRELEVANT_COMMENTS)
        
        # Spam genelde pozitif veya nötr
        sentiment = random.choice(["pos", "neu"])
        
        data.append({
            "post": {
                "text": post,
                "topic": topic,
                "platform": random.choice(PLATFORMS),
                "lang": random.choice(LANGUAGES)
            },
            "comment": {
                "text": comment,
                "lang": random.choice(LANGUAGES)
            },
            "meta": {
                "platform": random.choice(PLATFORMS),
                "post_topic": topic
            },
            "labels": {
                "sentiment": sentiment,
                "relevance": "irrelevant",  # Post ile ilgisiz
                "authenticity": "spam"  # Spam yorum
            }
        })
    
    # Destek yorumları (genel ama niyetli)
    for i in range(support_count):
        topic = random.choice(topics)
        post = random.choice(POSTS[topic])
        comment = random.choice(SUPPORT_COMMENTS)
        
        # Destek yorumları genelde pozitif
        sentiment = "pos"
        
        data.append({
            "post": {
                "text": post,
                "topic": topic,
                "platform": random.choice(PLATFORMS),
                "lang": random.choice(LANGUAGES)
            },
            "comment": {
                "text": comment,
                "lang": random.choice(LANGUAGES)
            },
            "meta": {
                "platform": random.choice(PLATFORMS),
                "post_topic": topic
            },
            "labels": {
                "sentiment": sentiment,
                "relevance": "general",  # Genel destek
                "authenticity": "support"  # Destek yorumu
            }
        })
    
    # Veriyi karıştır
    random.shuffle(data)
    
    return data


def save_post_comment_data(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Post + yorum verisini JSONL formatında kaydet.
    
    Args:
        data: Veri listesi
        file_path: Kaydetme yolu
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(data)} post+yorum örneği {file_path} dosyasına kaydedildi.")


if __name__ == "__main__":
    # Post + yorum verisi üret ve kaydet
    data = generate_post_comment_data(1000)
    save_post_comment_data(data, Path("src/data/samples/post_comments_demo.jsonl"))
    
    # İstatistikler
    relevant_count = sum(1 for item in data if item["labels"]["relevance"] == "relevant")
    irrelevant_count = sum(1 for item in data if item["labels"]["relevance"] == "irrelevant")
    general_count = sum(1 for item in data if item["labels"]["relevance"] == "general")
    
    print(f"\n📊 Relevance Dağılımı:")
    print(f"  İlgili: {relevant_count} (%{relevant_count/len(data)*100:.1f})")
    print(f"  İlgisiz: {irrelevant_count} (%{irrelevant_count/len(data)*100:.1f})")
    print(f"  Genel: {general_count} (%{general_count/len(data)*100:.1f})")
    
    # Authenticity dağılımı
    genuine_count = sum(1 for item in data if item["labels"]["authenticity"] == "genuine")
    spam_count = sum(1 for item in data if item["labels"]["authenticity"] == "spam")
    support_count = sum(1 for item in data if item["labels"]["authenticity"] == "support")
    
    print(f"\n🔍 Authenticity Dağılımı:")
    print(f"  Gerçek: {genuine_count} (%{genuine_count/len(data)*100:.1f})")
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
    
    # Topic dağılımı
    topic_counts = {}
    for item in data:
        topic = item["post"]["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"\n📝 Topic Dağılımı:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} (%{count/len(data)*100:.1f})")

