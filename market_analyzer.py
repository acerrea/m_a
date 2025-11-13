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

# --- تنظیمات اولیه ---
now = datetime.now()
now_str_file = f'{now:%Y-%m-%d}'
update_time_str = f'{now:%Y/%m/%d | %H:%M}'
DATA_SOURCE_URL = "TradersArena.ir"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

font_path_bold = "Vazirmatn-FD-ExtraBold.ttf"
font_path_regular = "Vazirmatn-FD-Regular.ttf"

if not os.path.exists(font_path_bold) or not os.path.exists(font_path_regular):
    print("هشدار: فایل‌های فونت یافت نشدند!")
    font_prop_bold = fm.FontProperties()
    font_prop_regular = fm.FontProperties()
else:
    font_prop_bold = fm.FontProperties(fname=font_path_bold)
    font_prop_regular = fm.FontProperties(fname=font_path_regular)

def reshape_text(text):
    return get_display(arabic_reshaper.reshape(str(text)))

# ... (توابع send_photo, send_message, parsers, ... بدون تغییر)
def send_photo_to_telegram(token, chat_id, photo_path, caption=""):
    print("\nدر حال ارسال عکس به تلگرام...")
    if not token or not chat_id: print("❌ توکن تلگرام یا آیدی چت تعریف نشده است."); return
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
    if not token or not chat_id: print("❌ توکن تلگرام یا آیدی چت تعریف نشده است."); return
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()
        if response.json().get("ok"): print("✅ پیام متنی با موفقیت ارسال شد.")
        else: print(f"❌ خطا در ارسال پیام متنی: {response.json()}")
    except Exception as e: print(f"خطا در فرآیند ارسال پیام متنی: {e}")

def get_gemini_analysis_text(last_row, previous_row, df):
    """فقط تحلیل متنی را از Gemini دریافت می‌کند."""
    print("\nدر حال دریافت تحلیل متنی از Gemini...")
    if not GEMINI_API_KEY: return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-flash-lite-latest') # استفاده از بهترین مدل برای کیفیت بالا
        
        prompt = f"""
        شما یک تحلیلگر ارشد بازار سرمایه ایران هستید. داده‌های زیر را تحلیل کرده و یک گزارش حرفه‌ای، خوانا و بدون اعراب‌گذاری برای نمایش در تلگرام آماده کن. از فرمت HTML (<b>, <i>, <code>) استفاده کن.
        - تاریخ: {last_row['تاریخ']}
        - ارزش معاملات: {last_row['ارزش معاملات']:,.1f} میلیارد تومان
        - شاخص کل: {last_row['شاخص کل']:,.0f} (تغییر: {last_row['شاخص کل'] - previous_row['شاخص کل']:+,.0f})
        - ورود پول: {last_row['ورود پول']:,.1f} میلیارد تومان
        - قدرت خریدار: {last_row['قدرت خريد']:.2f}
        """
        response = model.generate_content(prompt)
        print("✅ تحلیل متنی با موفقیت دریافت شد.")
        return response.text.strip()
            
    except Exception as e:
        print(f"❌ خطا در دریافت تحلیل متنی از Gemini: {e}")
        return None

def convert_text_to_speech_gemini(text, filename="analysis_audio.mp3"):
    """متن را با استفاده از مدل TTS داخلی Gemini به صوت تبدیل می‌کند."""
    print("در حال تبدیل متن به صوت با استفاده از Gemini TTS...")
    if not GEMINI_API_KEY: return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # استفاده از مدل تخصصی متن به صوت گوگل
        tts_model = genai.GenerativeModel('models/text-to-speech')

        response = tts_model.generate_content(
            text,
            voice="fa-IR-Standard-A"  # انتخاب صدای مرد فارسی
        )
        
        with open(filename, "wb") as f:
            f.write(response.audio_content)
        
        print(f"✅ فایل صوتی با موفقیت در '{filename}' ذخیره شد.")
        return filename
    except Exception as e:
        print(f"❌ خطا در تبدیل متن به صوت با Gemini TTS: {e}")
        return None
        
# ... (تمام توابع parse, proximity, moving_average, gauge ... بدون تغییر)
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
    if high_value > 0 and high_value > current_value:
        dist_from_high = abs((current_value - high_value) / high_value) * 100
        if dist_from_high <= threshold_percent:
            alert_msg = (f"  ⚠️ <b>هشدار:</b> با فاصله {dist_from_high:.1f}% از <b>{high_label}</b>.")
    if low_value > 0 and not alert_msg and current_value > low_value:
        dist_from_low = abs((current_value - low_value) / low_value) * 100
        if dist_from_low <= threshold_percent:
            alert_msg = (f"  💡 <b>نکته:</b> با فاصله {dist_from_low:.1f}% از <b>{low_label}</b>.")
    return alert_msg

def analyze_moving_averages(df):
    analysis_points = []
    if len(df) < 31: return analysis_points
    ma5 = df['ارزش معاملات'].rolling(window=5).mean().iloc[-1]
    ma10 = df['ارزش معاملات'].rolling(window=10).mean().iloc[-1]
    ma30 = df['ارزش معاملات'].rolling(window=30).mean().iloc[-1]
    if ma5 > ma10: analysis_points.append("<b>روند کوتاه‌مدت:</b> صعودی ✅")
    else: analysis_points.append("<b>روند کوتاه‌مدت:</b> نزولی ❌")
    if ma10 > ma30: analysis_points.append("<b>روند اصلی:</b> صعودی ✅")
    else: analysis_points.append("<b>روند اصلی:</b> نزولی ❌")
    return analysis_points

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
    ax.set_aspect('equal'); ax.axis('off')
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
    needle_x, needle_y = (radius - 0.1) * np.cos(needle_angle_rad), (radius - 0.1) * np.sin(needle_angle_rad)
    ax.plot([0, needle_x], [0, needle_y], color='black', lw=5, solid_capstyle='round', zorder=5)
    ax.add_patch(Circle((0, 0), 0.18, color='black', zorder=10))
    center_text = f"{current_value / 1000:.1f}\nهمت" if current_value >= 1000 else f"{int(current_value)}\nمیلیارد ت"
    ax.text(0, -0.02, reshape_text(center_text), ha='center', va='center', fontproperties=font_prop_bold, fontsize=22, color='white', zorder=11, linespacing=0.9)
    fig.text(0.5, 0.95, reshape_text("شاخص ترس و طمع بازار سهام"), ha='center', fontproperties=font_prop_bold, fontsize=28, color='#005a70')
    fig.text(0.5, 0.05, "Telegram: @Data_Bors", ha='center', fontproperties=font_prop_regular, fontsize=14, color='gray')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.2, 1.35)
    filename = f'Fear_Greed_Gauge-{file_str}.png'
    plt.savefig(filename, dpi=250, bbox_inches='tight'); plt.close(fig)
    print(f"✅ شاخص نهایی با موفقیت در فایل '{filename}' ذخیره شد.")
    return filename

def clean_text_for_speech(html_text):
    soup = bs_for_clean(html_text, "html.parser")
    return soup.get_text()

def send_audio_to_telegram(token, chat_id, audio_path, caption=""):
    print("در حال ارسال فایل صوتی به تلگرام...")
    if not token or not chat_id: print("❌ توکن تلگرام یا آیدی چت تعریف نشده است."); return
    api_url = f"https://api.telegram.org/bot{token}/sendAudio"
    try:
        with open(audio_path, 'rb') as audio_file:
            response = requests.post(
                api_url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                files={'audio': audio_file}, timeout=60)
            response.raise_for_status()
            if response.json().get("ok"): print("✅ فایل صوتی با موفقیت به تلگرام ارسال شد.")
            else: print(f"❌ خطا در ارسال فایل صوتی: {response.json()}")
    except Exception as e: print(f"خطا در فرآیند ارسال فایل صوتی: {e}")

def main():
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY]):
        print("❌ یکی از متغیرهای محیطی (تلگرام یا Gemini) تنظیم نشده است."); return

    print("--- شروع فرآیند تحلیل روزانه بازار ---")
    # ... (کد دریافت داده‌ها)
    print("در حال دریافت داده‌ها از TradersArena.ir...")
    data = []
    try:
        html = requests.get('https://tradersarena.ir/market/history?type=1', timeout=30, params={'perPage': 3000})
        html.raise_for_status()
        soup = BeautifulSoup(html.text, 'html.parser')
        table = soup.find('table', class_='sticky market')
        if not table:
            print("❌❌❌ خطای بحرانی: جدول داده‌ها یافت نشد."); return
        for tr in table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) > 22 and parse_financial_string(tds[2].text) > 0:
                data.append({"تاریخ": tds[1].text.strip(), 'ارزش معاملات': parse_financial_string(tds[2].text), 'قدرت خريد': parse_financial_string(tds[15].text), 'قدرت 5 روزه': parse_financial_string(tds[16].text), 'قدرت 20 روزه': parse_financial_string(tds[17].text), 'ورود پول': parse_financial_string(tds[18].text), 'ورود پول 5 روزه': parse_financial_string(tds[19].text), 'ورود پول 20 روزه': parse_financial_string(tds[20].text), 'شاخص کل': parse_index_string(tds[21].text), 'شاخص هم‌وزن': parse_index_string(tds[22].text)})
        print(f"✅ داده‌های {len(data)} روز با موفقیت دریافت شد.")
    except Exception as e: print(f"❌ خطا در دریافت داده: {e}"); return
    if len(data) < 2: print("❌ داده کافی برای تحلیل مقایسه‌ای وجود ندارد."); return

    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
    last_row, previous_row = df.iloc[-1], df.iloc[-2]
    
    # ... (کد ساخت و ارسال عکس و پیام داده‌ها)
    last_value, last_date = last_row['ارزش معاملات'], last_row['تاریخ']
    
    generated_filename = create_fear_greed_gauge_real_scale(last_value, now_str_file)
    if generated_filename and os.path.exists(generated_filename):
        status_short = "وضعیت: " + ("<b>ترس شدید</b> 🥶" if last_value < 3000 else "<b>ترس</b> 😟" if last_value < 5000 else "<b>خنثی</b> 😐" if last_value < 10000 else "<b>طمع</b> 😊" if last_value < 15000 else "<b>طمع شدید</b> 🤩🔥")
        photo_caption = "\n".join([f"<b>📊 شاخص ترس و طمع بازار سهام</b>", f"🗓️ تاریخ: {last_date}", f"<b>مقدار فعلی:</b> {last_value:,.1f} میلیارد تومان", status_short, "\n🆔 @Data_Bors"])
        send_photo_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, generated_filename, photo_caption)
        os.remove(generated_filename)
    
    full_message_blocks = [] # ... (کد کامل ساخت پیام داده‌ها)
    block1_parts = ["📈 <b>تحلیل ارزش معاملات</b>"]
    change = last_value - previous_row['ارزش معاملات']; percent = (change / previous_row['ارزش معاملات'] * 100) if previous_row['ارزش معاملات'] else 0
    block1_parts.append(f"• <b>مقدار امروز:</b> {last_value:,.1f} میلیارد.ت"); block1_parts.append(f"• <b>تغییر روزانه:</b> {abs(change):,.1f} میلیارد.ت {'کاهش' if change < 0 else 'افزایش'} {'⬇️' if change < 0 else '⬆️'} ({percent:+.1f}%)")
    if len(df) > 30:
        block1_parts.append("\n<b>میانگین‌های متحرک:</b>")
        for period in [5, 10, 30]:
            ma_series = df['ارزش معاملات'].rolling(window=period).mean(); current_avg, prev_avg = ma_series.iloc[-1], ma_series.iloc[-2]
            ma_trend = "⬆️" if current_avg > prev_avg else ("⬇️" if current_avg < prev_avg else "↔️")
            block1_parts.append(f"  - {period} روزه: <b>{current_avg:,.1f}</b> <i>(دیروز: {prev_avg:,.1f})</i> {ma_trend}")
        ma_analysis = analyze_moving_averages(df)
        if ma_analysis: block1_parts.append("\n" + "🔔 <b>تحلیل تکنیکال (ارزش معاملات):</b>"); block1_parts.extend([f"  - {point}" for point in ma_analysis])
    full_message_blocks.append("\n".join(block1_parts))

    block_indices = ["📉 <b>تحلیل شاخص‌های بازار</b>"]
    for name, key in [('کل', 'شاخص کل'), ('هم‌وزن', 'شاخص هم‌وزن')]:
        current_idx, prev_idx = last_row[key], previous_row[key]
        idx_change, idx_percent = current_idx - prev_idx, (current_idx - prev_idx) / prev_idx * 100 if prev_idx else 0
        ath_record_badge = ""; ath_message = ""
        if len(df) > 1:
            previous_ath = df[key][:-1].max()
            if current_idx > previous_ath: ath_record_badge = " (🚀 <b>رکورد جدید!</b>)"
            ath_message = f"  - سقف تاریخی: {int(max(current_idx, previous_ath)):,.0f}"
        else: ath_message = f"  - سقف تاریخی: {current_idx:,.0f}"
        yearly_subset = df.tail(252)
        yearly_low = yearly_subset[key].min()
        previous_yearly_high = yearly_subset[key][:-1].max() if len(yearly_subset) > 1 else yearly_low
        dist_from_high = (current_idx - previous_yearly_high) / previous_yearly_high * 100 if previous_yearly_high > 0 else 0
        dist_from_low = (current_idx - yearly_low) / yearly_low * 100 if yearly_low > 0 else 0
        yearly_high_message = f"📈<code>{int(previous_yearly_high):,.0f}</code> (<b>{dist_from_high:+.1f}%</b>)"
        if current_idx > previous_yearly_high: yearly_high_message = f"📈<code>{current_idx:,.0f}</code> (<b>رکورد جدید سال!</b>)"
        yearly_range_message = f"  - بازه یکساله (📉<code>{int(yearly_low):,.0f}</code> (<b>{dist_from_low:+.1f}%</b>) | {yearly_high_message})"
        idx_parts = [
            f"⚪️ <b>شاخص {name}</b>" if name == 'کل' else f"⚖️ <b>شاخص {name}</b>",
            f"  - مقدار فعلی: <code>{current_idx:,.0f}</code>{ath_record_badge} <b>({idx_change:+,.0f} | {idx_percent:+.2f}%)</b> {'⬆️' if idx_change >= 0 else '⬇️'}",
            ath_message, yearly_range_message
        ]
        proximity_alert = generate_proximity_alert(current_idx, previous_yearly_high, yearly_low, "سقف یکساله", "کف یکساله")
        if proximity_alert: idx_parts.append(proximity_alert)
        block_indices.append("\n".join(idx_parts))
    full_message_blocks.append("\n\n".join(block_indices))
    
    block3_parts = ["📊 <b>آمار تکمیلی بازار</b>"]
    p_power, p_power_prev = last_row['قدرت خريد'], previous_row['قدرت خريد']; p_money, p_money_prev = last_row['ورود پول'], previous_row['ورود پول']
    block3_parts.append(f"{'✅' if p_power >= 1 else '❌'} <b>قدرت خریدار:</b> <b>{p_power:.2f}</b> <i>(دیروز: {p_power_prev:.2f})</i> {'⬆️' if p_power > p_power_prev else '⬇️'}\n" f"    <i>میانگین ۵ روزه:</i>  {last_row['قدرت 5 روزه']:.2f}\n" f"    <i>میانگین ۲۰ روزه:</i> {last_row['قدرت 20 روزه']:.2f}")
    block3_parts.append(f"{'🟢' if p_money >= 0 else '🔴'} <b>ورود پول:</b> <b>{p_money:,.1f}</b> میلیارد.ت <i>(دیروز: {p_money_prev:,.1f})</i> {'⬆️' if p_money > p_money_prev else '⬇️'}\n" f"    <i>میانگین ۵ روزه:</i>  {last_row['ورود پول 5 روزه']:,.1f}\n" f"    <i>میانگین ۲۰ روزه:</i> {last_row['ورود پول 20 روزه']:,.1f}")
    full_message_blocks.append("\n\n".join(block3_parts))
    
    footer_parts = [f"<i>⏳ بروزرسانی: {update_time_str}</i>", f"🔗 منبع داده‌ها: <code>{DATA_SOURCE_URL}</code>", f"<i>#گزارش_روزانه_بازار</i>", f"🆔 @Data_Bors"]
    full_message_blocks.append("\n".join(footer_parts))

    data_message = ("\n\n" + "-" * 35 + "\n\n").join(filter(None, full_message_blocks))
    send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, data_message)

    # --- دریافت متن و سپس تولید صوت ---
    display_analysis_html = get_gemini_analysis_text(last_row, previous_row, df)
    
    if display_analysis_html:
        ai_message = display_analysis_html + "\n\n" + "\n".join([f"<i>این تحلیل توسط هوش مصنوعی (Google Gemini) تولید شده است.</i>", "🆔 @Data_Bors"])
        send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ai_message)

        text_for_speech_clean = clean_text_for_speech(display_analysis_html)
        audio_filename = convert_text_to_speech_gemini(text_for_speech_clean)
        
        if audio_filename and os.path.exists(audio_filename):
            audio_caption = "🎧 <b>نسخه صوتی تحلیل روز</b>\n\n" \
                            "<i>(تولید شده با صدای هوش مصنوعی گوگل)</i>\n\n" \
                            "🆔 @Data_Bors"
            send_audio_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, audio_filename, audio_caption)
            os.remove(audio_filename)

    print(f"\n--- عملیات با موفقیت به پایان رسید. ---")

if __name__ == "__main__":
    main()
