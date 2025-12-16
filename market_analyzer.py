import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import matplotlib.font_manager as fm
from jdatetime import datetime
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
import os
import google.generativeai as genai
import time # اضافه کردن کتابخانه time برای وقفه بین پیام‌ها

# --- تنظیمات اولیه ---
now = datetime.now()
now_str_file = f'{now:%Y-%m-%d}'
update_time_str = f'{now:%Y/%m/%d | %H:%M}'
DATA_SOURCE_URL = "TradersArena.ir"

# --- خواندن اطلاعات حساس ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --------------------

# تنظیم فونت
font_path_bold = "Vazirmatn-FD-ExtraBold.ttf"
font_path_regular = "Vazirmatn-FD-Regular.ttf"

if os.path.exists(font_path_bold):
    font_prop_bold = fm.FontProperties(fname=font_path_bold)
else:
    font_prop_bold = fm.FontProperties()

if os.path.exists(font_path_regular):
    font_prop_regular = fm.FontProperties(fname=font_path_regular)
else:
    font_prop_regular = font_prop_bold

def reshape_text(text):
    return get_display(arabic_reshaper.reshape(str(text)))

def send_photo_to_telegram(token, chat_id, photo_path, caption=""):
    print("\nدر حال ارسال عکس به تلگرام...")
    api_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        # کپشن عکس هم محدودیت ۱۰۲۴ کاراکتری دارد، اما معمولا کمتر پیش می‌آید پر شود
        # اگر کپشن خیلی طولانی بود، فقط ۱۰۰۰ کاراکتر اول ارسال شود
        if len(caption) > 1000:
            caption = caption[:1000] + "..."
            
        with open(photo_path, 'rb') as photo_file:
            response = requests.post(api_url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                                     files={'photo': photo_file}, timeout=30)
            response.raise_for_status()
            if response.json().get("ok"): print("✅ عکس با موفقیت به تلگرام ارسال شد.")
            else: print(f"❌ خطا در ارسال عکس: {response.json()}")
    except Exception as e: print(f"خطا در فرآیند ارسال عکس: {e}")

# <<< تابع اصلاح شده ارسال پیام (با قابلیت تکه کردن پیام‌های طولانی) >>>
def send_message_to_telegram(token, chat_id, text):
    print("در حال پردازش و ارسال پیام متنی به تلگرام...")
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # محدودیت تلگرام 4096 است، ما برای اطمینان 4000 در نظر می‌گیریم
    MAX_LENGTH = 4000 

    if len(text) <= MAX_LENGTH:
        messages_to_send = [text]
    else:
        print("⚠️ پیام طولانی است و به چند بخش تقسیم می‌شود.")
        messages_to_send = []
        while len(text) > MAX_LENGTH:
            # تلاش برای پیدا کردن آخرین خط جدید (\n) قبل از مرز 4000 کاراکتر
            split_index = text[:MAX_LENGTH].rfind('\n')
            
            # اگر خط جدید پیدا نشد (خیلی بعید است)، آخرین فاصله را پیدا کن
            if split_index == -1:
                split_index = text[:MAX_LENGTH].rfind(' ')
            
            # اگر هیچ فاصله‌ای هم نبود، به ناچار در همان 4000 برش بزن
            if split_index == -1:
                split_index = MAX_LENGTH
            
            messages_to_send.append(text[:split_index])
            text = text[split_index:].strip() # حذف فاصله‌های اضافی ابتدای بخش بعدی
        
        # افزودن بخش باقیمانده
        if text:
            messages_to_send.append(text)

    # ارسال تک تک پیام‌ها
    for i, msg in enumerate(messages_to_send):
        payload = {'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}
        try:
            response = requests.post(api_url, json=payload, timeout=20)
            response.raise_for_status()
            if response.json().get("ok"): 
                print(f"✅ بخش {i+1} از {len(messages_to_send)} با موفقیت ارسال شد.")
            else: 
                print(f"❌ خطا در ارسال بخش {i+1}: {response.json()}")
            
            # وقفه کوتاه برای جلوگیری از اسپم شناخته شدن توسط تلگرام
            if len(messages_to_send) > 1:
                time.sleep(1) 
                
        except Exception as e: 
            print(f"خطا در فرآیند ارسال پیام (بخش {i+1}): {e}")

def get_gemini_analysis(last_row, previous_row, df):
    print("\nدر حال دریافت تحلیل از هوش مصنوعی Gemini...")
    if not GEMINI_API_KEY:
        print("❌ کلید API جمنای یافت نشد. تحلیل هوش مصنوعی انجام نمی‌شود.")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-flash-lite-latest')

        prompt = f"""
        شما یک تحلیلگر ارشد بازار سرمایه ایران هستید. لطفاً داده‌های زیر را تحلیل کنید.
        از تگ های HTML تلگرام (<b>, <i>, <code>) استفاده کن.
        
        **داده‌ها:**
        - تاریخ: {last_row['تاریخ']}
        - ارزش معاملات امروز: {last_row['ارزش معاملات']:,.1f} همت (دیروز: {previous_row['ارزش معاملات']:,.1f})
        - شاخص کل: {last_row['شاخص کل']:,.0f} (تغییر: {(last_row['شاخص کل'] - previous_row['شاخص کل']):+,.0f})
        - ورود پول حقیقی: {last_row['ورود پول']:,.1f} همت
        - قدرت خریدار: {last_row['قدرت خريد']:.2f}

        **درخواست:**
        یک تحلیل جامع بنویس شامل: 
        1. عنوان جذاب
        2. تحلیل سنتیمنت بازار
        3. نقاط قوت و ضعف
        4. پیش‌بینی فردا
        
        خروجی طولانی و دقیق باشد.
        """
        
        response = model.generate_content(prompt)
        print("✅ تحلیل هوش مصنوعی با موفقیت دریافت شد.")
        return response.text
    except Exception as e:
        print(f"❌ خطا در ارتباط با Gemini API: {e}")
        return "تحلیل هوش مصنوعی در حال حاضر در دسترس نیست."

# ... (بقیه توابع: parse_financial_string, parse_index_string, generate_proximity_alert, analyze_moving_averages, create_fear_greed_gauge_real_scale بدون تغییر) ...
# این بخش‌ها را از کد قبلی خود کپی کنید چون تغییری نیاز ندارند.
def parse_financial_string(s):
    if not isinstance(s, str): return 0.0
    s = s.strip().replace(',', '')
    try:
        if 'B' in s.upper(): return float(s.upper().replace('B', '').strip())
        if 'M' in s.upper(): return float(s.upper().replace('M', '').strip()) / 1000.0
        return float(s)
    except (ValueError, AttributeError): return 0.0

def parse_index_string(s):
    if not isinstance(s, str): return 0
    try:
        return int(s.strip().replace(',', ''))
    except (ValueError, AttributeError): return 0

def generate_proximity_alert(current_value, high_value, low_value, high_label, low_label, threshold_percent=10):
    alert_msg = ""
    if high_value > 0:
        dist_from_high = abs((high_value - current_value) / high_value) * 100
        if dist_from_high <= threshold_percent:
            alert_msg = (f"  ⚠️ <b>هشدار:</b> با فاصله {dist_from_high:.1f}% از <b>{high_label}</b>، "
                         f"<b>احتمال</b> افزایش ریسک اصلاح و عرضه وجود دارد.")
    if low_value > 0 and not alert_msg:
        dist_from_low = abs((current_value - low_value) / low_value) * 100
        if dist_from_low <= threshold_percent:
            alert_msg = (f"  💡 <b>نکته:</b> با فاصله {dist_from_low:.1f}% از <b>{low_label}</b>، "
                         f"<b>احتمال</b> برگشت بازار و پایان روند نزولی وجود دارد.")
    return alert_msg

def analyze_moving_averages(df):
    analysis_points = []
    if len(df) < 31: return analysis_points

    ma5 = df['ارزش معاملات'].rolling(window=5).mean()
    ma10 = df['ارزش معاملات'].rolling(window=10).mean()
    ma30 = df['ارزش معاملات'].rolling(window=30).mean()

    if ma5.iloc[-1] > ma10.iloc[-1]: analysis_points.append("<b>روند کوتاه‌مدت:</b> صعودی ✅.")
    else: analysis_points.append("<b>روند کوتاه‌مدت:</b> نزولی ❌.")
    
    # ... بقیه شرط های میانگین متحرک ...
    return analysis_points

def create_fear_greed_gauge_real_scale(current_value, file_str):
    # ... (کد رسم نمودار بدون تغییر) ...
    # برای جلوگیری از طولانی شدن پاسخ، کد رسم نمودار را اینجا تکرار نکردم
    # لطفا همان کد قبلی خودتان را اینجا قرار دهید.
    return None # در کد اصلی شما اینجا اسم فایل برمی‌گردد

# --- مراحل اصلی اجرا ---
def main():
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("❌ متغیرهای محیطی تلگرام تنظیم نشده‌اند.")
        return

    print("در حال دریافت داده‌ها...")
    # ... (کد دریافت و پردازش داده‌ها مشابه قبل) ...
    # برای تست سریع من قسمت دریافت داده را شبیه سازی میکنم. 
    # شما کد اصلی خودتان را نگه دارید.
    
    # فرض کنیم df ساخته شده است (کد اصلی خودتان را اینجا بگذارید)
    # ---------------------------------------------------------
    data = []
    try:
        html = requests.get('https://tradersarena.ir/market/history?type=1', timeout=30, params={'perPage': 3000})
        html.raise_for_status()
        for tr in BeautifulSoup(html.text, 'html.parser').find('table', class_='sticky market').find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) > 22 and parse_financial_string(tds[2].text) > 0:
                data.append({"تاریخ": tds[1].text, 'ارزش معاملات': parse_financial_string(tds[2].text), 'قدرت خريد': parse_financial_string(tds[15].text), 'قدرت 5 روزه': parse_financial_string(tds[16].text), 'قدرت 20 روزه': parse_financial_string(tds[17].text), 'ورود پول': parse_financial_string(tds[18].text), 'ورود پول 5 روزه': parse_financial_string(tds[19].text), 'ورود پول 20 روزه': parse_financial_string(tds[20].text), 'شاخص کل': parse_index_string(tds[21].text), 'شاخص هم‌وزن': parse_index_string(tds[22].text)})
        print(f"داده‌های {len(data)} روز با موفقیت دریافت شد.")
    except Exception as e: print(f"خطا در دریافت داده: {e}"); return
    
    if len(data) < 2: return

    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
    last_row, previous_row = df.iloc[-1], df.iloc[-2]
    # ---------------------------------------------------------

    last_value = last_row['ارزش معاملات']
    last_date = last_row['تاریخ']
    
    # 1. تولید و ارسال عکس
    # generated_filename = create_fear_greed_gauge_real_scale(last_value, now_str_file)
    # اگر تابع نمودار را در کد دارید خط بالا را از کامنت خارج کنید
    generated_filename = None # موقت

    if generated_filename:
        # کد ارسال عکس
        pass 
    
    # 2. ارسال داده‌های خام (پیام اول)
    full_message_blocks = []
    full_message_blocks.append(f"📅 <b>گزارش بازار - {last_date}</b>")
    full_message_blocks.append(f"💰 ارزش معاملات: {last_value:,.1f} میلیارد تومان")
    # ... سایر بلوک‌های داده ...
    
    data_message = "\n\n".join(full_message_blocks)
    send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, data_message)

    # 3. ارسال تحلیل هوش مصنوعی (پیام دوم - با قابلیت ارسال متن‌های طولانی)
    ai_analysis = get_gemini_analysis(last_row, previous_row, df)
    
    if ai_analysis:
        # اضافه کردن امضا به انتهای تحلیل
        final_ai_message = ai_analysis + "\n\n" + "\n".join([f"<i>این تحلیل توسط هوش مصنوعی (Google Gemini) تولید شده است.</i>", "🆔 @Data_Bors"])
        
        # تابع جدید send_message_to_telegram حالا خودش متن را چک میکند
        # اگر متن بیشتر از 4000 کاراکتر باشد، آن را تکه تکه ارسال میکند
        send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_ai_message)

    print(f"\n--- عملیات با موفقیت به پایان رسید. ---")

if __name__ == "__main__":
    main()
