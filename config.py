import os
from dotenv import load_dotenv

# .env faylni yuklash
load_dotenv()

# O'zgaruvchilarni olish
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# Tekshirish
print(f"Bot token: {BOT_TOKEN}")
print(f"Admin ID: {ADMIN_ID}")

