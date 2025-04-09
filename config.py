import os
from dotenv import load_dotenv

# .env faylni yuklash
load_dotenv()

# O'zgaruvchilarni olish
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Tekshirish
print(f"Bot token: {BOT_TOKEN}")
