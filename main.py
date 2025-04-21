import os
from os import environ

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import zscore
from numba import njit
import warnings
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
import asyncio
from io import BytesIO
from config import BOT_TOKEN, ADMIN_IDS

warnings.filterwarnings('ignore')

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

def is_admin(user_id: int):
    return user_id in ADMIN_IDS

# Your Rasch Model implementation (unchanged)
class FastRaschModel:
    def __init__(self):
        self.item_difficulty = None
        self.person_ability = None

    @staticmethod
    @njit
    def _calculate_log_likelihood(X, beta, theta):
        log_lik = 0.0
        n_persons, n_items = X.shape
        for i in range(n_persons):
            for j in range(n_items):
                diff = theta[i] - beta[j]
                if diff > 20:
                    p = 1.0
                elif diff < -20:
                    p = 0.0
                else:
                    p = 1.0 / (1.0 + np.exp(-diff))

                if X[i, j] == 1:
                    log_lik += np.log(p)
                else:
                    log_lik += np.log(1.0 - p)
        return -log_lik

    def fit(self, X, max_iter=50, tol=1e-3, batch_size=2000):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        n_persons, n_items = X_np.shape

        initial_beta = np.zeros(n_items)
        initial_theta = np.zeros(n_persons)
        initial_guess = np.concatenate([initial_beta, initial_theta])

        bounds = [(-5, 5)] * n_items + [(-5, 5)] * n_persons

        if n_persons > batch_size:
            for iteration in range(max_iter):
                batch_idx = np.random.choice(n_persons, size=batch_size, replace=False)
                X_batch = X_np[batch_idx, :]

                def batch_neg_log_lik(params):
                    beta = params[:n_items]
                    theta = params[n_items:]
                    theta_batch = theta[batch_idx]
                    return self._calculate_log_likelihood(X_batch, beta, theta_batch)

                result = minimize(batch_neg_log_lik, initial_guess,
                                  method='L-BFGS-B',
                                  bounds=bounds,
                                  options={'maxiter': 10, 'gtol': tol})

                initial_guess = result.x
                if result.fun < tol:
                    break
        else:
            def full_neg_log_lik(params):
                beta = params[:n_items]
                theta = params[n_items:]
                return self._calculate_log_likelihood(X_np, beta, theta)

            result = minimize(full_neg_log_lik, initial_guess,
                              method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': max_iter, 'gtol': tol})

        self.item_difficulty = result.x[:n_items]
        self.person_ability = result.x[n_items:]


# Create keyboard
rasch_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Rasch Result")]],
    resize_keyboard=True
)


# Handlers
@dp.message(Command("start", "help"))
async def send_welcome(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.answer("Assalomu Alaykum!\n❌ Siz admin emassiz!\n Shuning uchun bizning botdan foydalana olmaysiz!")
    await message.answer(
        "Rasch Model Bot ga Xush Kelibsiz! \nQuyidagi 'Rasch Result' tugmasini bosing va menga analiz qilish uchun Excel file yuboring.",
        reply_markup=rasch_keyboard
    )


@dp.message(lambda message: message.text == "Rasch Result")
async def request_file(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.answer("❌ Siz admin emassiz!\n Shuning uchun bizning botdan foydalana olmaysiz!")
    await message.answer("Iltimos Excel file yuboring.")


@dp.message(lambda message: message.document is not None)
async def handle_document(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.answer("❌ Siz admin emassiz!\n Shuning uchun bizning botdan foydalana olmaysiz!")
    if not message.document.file_name.endswith(('.xlsx', '.xls')):
        await message.answer("Iltimos Excel file yuboring (.xlsx or .xls)")
        return

    await message.answer("Sizning faylingiz analiz qilinyapti...")

    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = f"temp_{message.from_user.id}.xlsx"

    await bot.download(file, destination=file_path)

    try:
        print("Loading data...")
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Data loaded successfully with {len(df)} rows")
        print("Faylda mavjud ustunlar:", df.columns.tolist())

        # Agar ustun nomlari boshqacha bo'lsa, ularni moslashtiring
        required_columns = ['№', 'F.I.O.', 'Duris']
        available_columns = df.columns.tolist()

        # Ustun nomlarini tekshirish va moslashtirish
        if not all(col in available_columns for col in required_columns):
            # Agar standart nomlar topilmasa, birinchi 3 ustundan foydalaning
            if len(df.columns) >= 3:
                df.columns = ['№', 'F.I.O.', 'Duris'] + list(df.columns[3:])
                print("Ustun nomlari avtomatik moslashtirildi")
            else:
                raise ValueError("Faylda kamida 3 ta ustun bo'lishi kerak")

        # Column handling
        if len(df.columns) < 3:
            raise ValueError("File must have at least 3 columns")

        # Auto-detect response columns (assuming they start from column 3)
        response_cols = df.columns[3:]
        print(f"Detected {len(response_cols)} response columns")

        # Convert responses to binary (1 for correct, 0 for incorrect)
        response_data = df[response_cols].applymap(lambda x: 1 if x == 1 else 0)

        # Fit Rasch model with progress tracking
        print("Fitting Rasch model...")
        model = FastRaschModel()
        model.fit(response_data)

        # Calculate scores
        print("Calculating scores...")
        df['Theta'] = model.person_ability
        df['T_score'] = 50 + 10 * zscore(df['Theta'])

        noise = np.random.uniform(-0.07, 0.07, size=len(df))


        df['T_score'] = np.round(df['T_score'], 2)
        df['T_score'] += noise


        # Determine subject type based on max possible score
        max_possible = len(response_cols)
        subject_type = "1-fan" if max_possible >= 45 else "2-fan"

        # Calculate proportional scores
        theta_min = df['Theta'].min()
        theta_range = df['Theta'].max() - theta_min
        if theta_range > 0:
            df['Prop_Score'] = ((df['Theta'] - theta_min) / theta_range) * (max_possible - 65) + 65
        else:
            df['Prop_Score'] = 65  # Handle case where all abilities are equal

        # Assign grades
        bins = [0, 46, 50, 55, 60, 65, 70, 93]
        labels = ['D', 'C', 'C+', 'B', 'B+', 'A', 'A+']
        df['Grade'] = pd.cut(df['T_score'], bins=bins, labels=labels, right=False)

        # Save results
        result_cols = ['№', 'F.I.O.', 'T_score', 'Grade']
        if '№' not in df.columns:
            result_cols = [col for col in result_cols if col != '№']

        print("Saving results...")
        result_path = f'result_{message.from_user.id}.xlsx'

        df = df.sort_values(by='T_score', ascending=False)
        df[result_cols].to_excel(result_path, index=False, engine='openpyxl')

        result_file = types.FSInputFile(result_path)
        await message.answer_document(
            document=result_file,
            caption=f'Analysis tugadi!'
        )

    except ValueError as e:
        if "At least one sheet must be visible" in str(e):
            await message.answer(
                "Error: The Excel file has no visible sheets. Please ensure at least one sheet is visible and contains data."
            )
        else:
            await message.answer(f"Error processing file: {str(e)}")
    except Exception as e:
        await message.answer(f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if 'result_path' in locals() and os.path.exists(result_path):
            os.remove(result_path)

@dp.message()
async def handle_other_messages(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.answer("❌ Siz admin emassiz!\n Shuning uchun bizning botdan foydalana olmaysiz!")
    await message.answer("Quyidagi 'Rasch Result' tugmasini bosing va menga analiz qilish uchun Excel file yuboring.", reply_markup=rasch_keyboard)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())