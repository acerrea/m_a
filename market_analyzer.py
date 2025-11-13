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
from bs4 import BeautifulSoup as bs_for_clean
import asyncio
import edge_tts

# --- تنظیمات اولیه ---
now = datetime.now()
now_str_file = f'{now:%Y-%m-%d}'
update_time_str = f'{now:%Y/%m/%d | %H:%M}'
DATA_SOURCE_URL = "TradersArena.ir"

# --- خواندن اطلاعات حساس از متغیرهای محیطی ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- تنظیم فونت ---
font_path_bold = "Vazirmatn-FD-ExtraBold.ttf"
font_path_regular = "Vazirmatn-FD-Regular.ttf"

if not os.path.exists(font_path_bold) or not os.path.exists(font_path_regular):
    print("هشدار: فایل‌های فونت یافت نشدند! مطمئن شوید در ریشه مخزن قرار دارند.")
    font_prop_bold = fm.FontProperties()
    font_prop_regular = fm.FontProperties()
else:
    font_prop_bold = fm.FontProperties(fname=font_path_bold)
    font_prop_regular = fm.FontProperties(fname=font_path_regular)

def reshape_text(text):
    return get_display(arabic_reshaper.reshape(str(text)))

def send_photo_to_telegram(token, chat_id, photo_path, caption=""):
    print("\nدر حال ارسال عکس به تلگرام...")
    if not token or not chat_id:
        print("❌ توکن تلگرام یا آیدی چت تعریف نشده است.")
        return
    api_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo_file:
            response = requests.post(api_url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                                     files={'photo': photo_file}, timeout=30)
            response.raise_for_status()
            if response.json().get("ok"): print("✅ عکس با موفقیت به تلگرام ارسال شد.")
            else: print(f"❌ خطا در ارسال عکس: {response.json()}")
    except Exception as e: print(f"خطا در فرآیند ارسال عکس: {e}")

def send_message_to_telegram(token, chat_id, text):
    print("در حال ارسال پیام متنی به تلگرام...")
    if not token or not chat_id:
        print("❌ توکن تلگرام یا آیدی چت تعریف نشده است.")
        return
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()
        if response.json().get("ok"): print("✅ پیام متنی با موفقیت ارسال شد.")
        else: print(f"❌ خطا در ارسال پیام متنی: {response.json()}")
    except Exception as e: print(f"خطا در فرآیند ارسال پیام متنی: {e}")

def get_gemini_analysis(last_row, previous_row, df):
    print("\nدر حال دریافت تحلیل از هوش مصنوعی Gemini...")
    if not GEMINI_API_KEY:
        print("❌ کلید API جمنای یافت نشد. تحلیل هوش مصنوعی انجام نمی‌شود.")
        return "تحلیل هوش مصنوعی به دلیل عدم وجود کلید API در دسترس نیست."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-flash-lite-latest')
        prompt = f"""
        شما یک تحلیلگر ارشد بازار سرمایه ایران هستید. لطفاً داده‌های زیر را که مربوط به امروز و دیروز بازار سهام تهران است، تحلیل کنید. تحلیل شما باید حرفه‌ای، عمیق و به زبان فارسی روان باشد. از فرمت HTML تلگرام (<b>, <i>, <code>) برای برجسته‌سازی استفاده کنید.

        **داده‌های کلیدی:**
        - **تاریخ گزارش:** {last_row['تاریخ']}
        - **ارزش معاملات خرد امروز:** {last_row['ارزش معاملات']:,.1f} میلیارد تومان (دیروز: {previous_row['ارزش معاملات']:,.1f})
        - **شاخص کل امروز:** {last_row['شاخص کل']:,.0f} (تغییر: {(last_row['شاخص کل'] - previous_row['شاخص کل']):+,.0f})
        - **شاخص هم‌وزن امروز:** {last_row['شاخص هم‌وزن']:,.0f} (تغییر: {(last_row['شاخص هم‌وزن'] - previous_row['شاخص هم‌وزن']):+,.0f})
        - **ورود/خروج پول حقیقی امروز:** {last_row['ورود پول']:,.1f} میلیارد تومان
        - **قدرت خریدار به فروشنده امروز:** {last_row['قدرت خريد']:.2f}

        **وظیفه شما:**
        1.  یک عنوان جذاب و توصیفی برای تحلیل امروز انتخاب کنید.
        2.  **تحلیل جامع بازار:** سنتیمنت کلی بازار را تحلیل کنید.
        3.  **نقاط قوت و ضعف:** مهم‌ترین سیگنال‌های مثبت و منفی را لیست کنید.
        4.  **چشم‌انداز کوتاه‌مدت:** یک نتیجه‌گیری و چشم‌انداز ارائه دهید.
        
        **خروجی باید به این شکل باشد:**
        📝 <b>[عنوان جذاب شما]</b>
        [تحلیل جامع شما]
        🟢 <b>نقاط قوت:</b>
        - [نکته ۱]
        🔴 <b>نقاط ضعف:</b>
        - [نکته ۱]
        💡 <b>جمع‌بندی:</b>
        [نتیجه‌گیری نهایی]
        """
        response = model.generate_content(prompt)
        print("✅ تحلیل هوش مصنوعی با موفقیت دریافت شد.")
        return response.text
    except Exception as e:
        print(f"❌ خطا در ارتباط با Gemini API: {e}")
        return "تحلیل هوش مصنوعی در حال حاضر در دسترس نیست."

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

def create_fear_greed_gauge_real_scale(current_value, file_str):
    print(f"\nدر حال ایجاد شاخص ترس و طمع...")
    GAUGE_DISPLAY_MAX = 25000.0
    segments_real = [{'range': (0, 3000), 'color': '#d52b1e', 'label': 'ترس شدید'}, 
                     {'range': (3000, 5000), 'color': '#f3c316', 'label': 'ترس'}, 
                     {'range': (5000, 10000), 'color': '#808285', 'label': 'خنثی'}, 
                     {'range': (10000, 15000), 'color': '#0096a8', 'label': 'طمع'}, 
                     {'range': (15000, 20000), 'color': '#8dc63f', 'label': 'طمع شدید'}, 
                     {'range': (20000, GAUGE_DISPLAY_MAX), 'color': '#00a651', 'label': 'طمع\nخیلی شدید'}]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f0f0f0')
    ax.set_aspect('equal')
    ax.axis('off')
    center, radius, width = (0, 0), 1.0, 0.45
    for seg in segments_real:
        start_val, end_val = seg['range']
        start_angle = 180 - (end_val / GAUGE_DISPLAY_MAX * 180)
        end_angle = 180 - (start_val / GAUGE_DISPLAY_MAX * 180)
        wedge = Wedge(center=center, r=radius, theta1=start_angle, theta2=end_angle, width=width, facecolor=seg['color'], edgecolor=fig.get_facecolor(), lw=5)
        ax.add_patch(wedge)
        mid_angle_rad = np.deg2rad((start_angle + end_angle) / 2)
        x, y = (radius - width / 2) * np.cos(mid_angle_rad), (radius - width / 2) * np.sin(mid_angle_rad)
        ax.text(x, y, reshape_text(seg['label']), ha='center', va='center', fontproperties=font_prop_bold, fontsize=16, color='white', linespacing=0.95)
    needle_angle_rad = np.deg2rad(180 - (min(current_value, GAUGE_DISPLAY_MAX) / GAUGE_DISPLAY_MAX * 180))
    needle_x = (radius - 0.1) * np.cos(needle_angle_rad)
    needle_y = (radius - 0.1) * np.sin(needle_angle_rad)
    ax.plot([0, needle_x], [0, needle_y], color='black', lw=5, solid_capstyle='round', zorder=5)
    ax.add_patch(Circle((0, 0), 0.18, color='black', zorder=10))
    center_text = f"{current_value / 1000:.1f}\nهمت" if current_value >= 1000 else f"{int(current_value)}\nمیلیارد ت"
    ax.text(0, -0.02, reshape_text(center_text), ha='center', va='center', fontproperties=font_prop_bold, fontsize=22, color='white', zorder=11, linespacing=0.9)
    fig.text(0.5, 0.95, reshape_text("شاخص ترس و طمع بازار سهام"), ha='center', fontproperties=font_prop_bold, fontsize=28, color='#005a70')
    fig.text(0.5, 0.05, "Telegram: @Data_Bors", ha='center', fontproperties=font_prop_regular, fontsize=14, color='gray')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.2, 1.35)
    filename = f'Fear_Greed_Gauge-{file_str}.png'
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ شاخص نهایی با موفقیت در فایل '{filename}' ذخیره شد.")
    return filename

def clean_text_for_speech(html_text):
    soup = bs_for_clean(html_text, "html.parser")
    text = soup.get_text()
    return text

async def convert_text_to_speech_async(text, filename="analysis_audio.mp3"):
    """متن را با استفاده از Edge TTS به صورت ناهمگام به فایل صوتی تبدیل می‌کند."""
    print("در حال تبدیل متن به صوت با استفاده از Edge TTS...")
    try:
        communicate = edge_tts.Communicate(text, "fa-IR-DilaraNeural") # صدای زن
        await communicate.save(filename)
        print(f"✅ فایل صوتی با موفقیت در '{filename}' ذخیره شد.")
        return filename
    except Exception as e:
        print(f"❌ خطا در تبدیل متن به صوت: {e}")
        return None

def send_audio_to_telegram(token, chat_id, audio_path, caption=""):
    print("در حال ارسال فایل صوتی به تلگرام...")
    if not token or not chat_id:
        print("❌ توکن تلگرام یا آیدی چت تعریف نشده است.")
        return
    api_url = f"https://api.telegram.org/bot{token}/sendAudio"
    try:
        with open(audio_path, 'rb') as audio_file:
            response = requests.post(
                api_url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                files={'audio': audio_file}, timeout=60)
            response.raise_for_status()
            if response.json().get("ok"):
                print("✅ فایل صوتی با موفقیت به تلگرام ارسال شد.")
            else:
                print(f"❌ خطا در ارسال فایل صوتی: {response.json()}")
    except Exception as e:
        print(f"خطا در فرآیند ارسال فایل صوتی: {e}")

def main():
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("❌ متغیرهای محیطی تلگرام تنظیم نشده‌اند. برنامه متوقف می‌شود.")
        return

    print("--- شروع فرآیند تحلیل روزانه بازار ---")
    print("در حال دریافت داده‌ها از TradersArena.ir...")
    data = []
    try:
        html = requests.get('https://tradersarena.ir/market/history?type=1', timeout=30, params={'perPage': 3000})
        html.raise_for_status()
        soup = BeautifulSoup(html.text, 'html.parser')
        
        table = soup.find('table', class_='sticky market')
        if not table:
            print("❌❌❌ خطای بحرانی: جدول داده‌ها یافت نشد. احتمالاً ساختار سایت تغییر کرده است.")
            return
        
        for tr in table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) > 22 and parse_financial_string(tds[2].text) > 0:
                data.append({"تاریخ": tds[1].text.strip(), 'ارزش معاملات': parse_financial_string(tds[2].text), 'قدرت خريد': parse_financial_string(tds[15].text), 'ورود پول': parse_financial_string(tds[18].text), 'شاخص کل': parse_index_string(tds[21].text), 'شاخص هم‌وزن': parse_index_string(tds[22].text)})
        print(f"✅ داده‌های {len(data)} روز با موفقیت دریافت شد.")
    except Exception as e: 
        print(f"❌ خطا در دریافت داده: {e}")
        return
        
    if len(data) < 2: 
        print("❌ داده کافی برای تحلیل مقایسه‌ای وجود ندارد.")
        return

    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
    last_row, previous_row = df.iloc[-1], df.iloc[-2]
    
    # ... (بقیه کد بدون تغییر)
    last_value = last_row['ارزش معاملات']
    last_date = last_row['تاریخ']
    
    generated_filename = create_fear_greed_gauge_real_scale(last_value, now_str_file)
    if generated_filename and os.path.exists(generated_filename):
        status_short = "وضعیت: " + ("<b>ترس شدید</b> 🥶" if last_value < 3000 else "<b>ترس</b> 😟" if last_value < 5000 else "<b>خنثی</b> 😐" if last_value < 10000 else "<b>طمع</b> 😊" if last_value < 15000 else "<b>طمع شدید</b> 🤩🔥")
        photo_caption = "\n".join([f"<b>📊 شاخص ترس و طمع بازار سهام</b>", f"🗓️ تاریخ: {last_date}", f"<b>مقدار فعلی:</b> {last_value:,.1f} میلیارد تومان", status_short, "\n🆔 @Data_Bors"])
        send_photo_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, generated_filename, photo_caption)
        os.remove(generated_filename)
    
    # --- ساخت پیام داده‌های خام ---
    full_message_blocks = []
    block1_parts = ["📈 <b>آمار ارزش معاملات</b>"]
    change = last_value - previous_row['ارزش معاملات']; percent = (change / previous_row['ارزش معاملات'] * 100) if previous_row['ارزش معاملات'] else 0
    block1_parts.append(f"• <b>امروز:</b> {last_value:,.1f} میلیارد تومان")
    block1_parts.append(f"• <b>تغییر روزانه:</b> {abs(change):,.1f} میلیارد تومان {'کاهش' if change < 0 else 'افزایش'} {'⬇️' if change < 0 else '⬆️'} ({percent:+.1f}%)")
    full_message_blocks.append("\n".join(block1_parts))

    block_indices = ["📉 <b>آمار شاخص‌های بازار</b>"]
    for name, key in [('کل', 'شاخص کل'), ('هم‌وزن', 'شاخص هم‌وزن')]:
        current_idx, prev_idx = last_row[key], previous_row[key]
        idx_change, idx_percent = current_idx - prev_idx, (current_idx - prev_idx) / prev_idx * 100 if prev_idx else 0
        block_indices.append(f"⚪️ <b>شاخص {name}:</b> <code>{current_idx:,.0f}</code> ({idx_change:+,.0f} | {idx_percent:+.2f}%) {'⬆️' if idx_change >= 0 else '⬇️'}")
    full_message_blocks.append("\n".join(block_indices))
    
    block3_parts = ["📊 <b>آمار تکمیلی</b>"]
    p_power = last_row['قدرت خريد']
    p_money = last_row['ورود پول']
    block3_parts.append(f"{'✅' if p_power >= 1 else '❌'} <b>قدرت خریدار:</b> <b>{p_power:.2f}</b>")
    block3_parts.append(f"{'🟢' if p_money >= 0 else '🔴'} <b>ورود پول:</b> <b>{p_money:,.1f}</b> میلیارد تومان")
    full_message_blocks.append("\n".join(block3_parts))
    
    footer_parts = [f"<i>⏳ بروزرسانی: {update_time_str}</i>", f"🔗 منبع: <code>{DATA_SOURCE_URL}</code>", f"🆔 @Data_Bors"]
    full_message_blocks.append("\n".join(footer_parts))

    data_message = ("\n\n" + "-" * 25 + "\n\n").join(filter(None, full_message_blocks))
    send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, data_message)

    # --- دریافت، ارسال متن و ساخت صوت تحلیل هوش مصنوعی ---
    ai_analysis_html = get_gemini_analysis(last_row, previous_row, df)
    if ai_analysis_html:
        ai_message = ai_analysis_html + "\n\n" + "\n".join([f"<i>این تحلیل توسط هوش مصنوعی (Google Gemini) تولید شده است.</i>", "🆔 @Data_Bors"])
        send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ai_message)

        text_for_speech = clean_text_for_speech(ai_analysis_html)
        audio_filename = asyncio.run(convert_text_to_speech_async(text_for_speech))
        
        if audio_filename and os.path.exists(audio_filename):
            audio_caption = "🎧 <b>نسخه صوتی تحلیل روز</b>\n\n" \
                            "<i>(تولید شده با صدای هوش مصنوعی مایکروسافت)</i>\n\n" \
                            "🆔 @Data_Bors"
            send_audio_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, audio_filename, audio_caption)
            os.remove(audio_filename)

    print(f"\n--- عملیات با موفقیت به پایان رسید. ---")

if __name__ == "__main__":
    main()
