import os
from dotenv import load_dotenv

# .env faylni yuklash
load_dotenv()

# O'zgaruvchilarni olish
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_IDS = [int(admin_id.strip()) for admin_id in os.getenv("ADMIN_IDS", "").split(',') if admin_id.strip().isdigit()]


# Tekshirish
print(f"Bot token: {BOT_TOKEN}")
print(f"Admin ID: {ADMIN_IDS}")

