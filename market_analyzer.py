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

# --- تنظیمات اولیه ---
now = datetime.now()
now_str_file = f'{now:%Y-%m-%d}'
update_time_str = f'{now:%Y/%m/%d | %H:%M}'
DATA_SOURCE_URL = "TradersArena.ir"

# --- خواندن اطلاعات حساس از متغیرهای محیطی (برای امنیت در گیت‌هاب) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --------------------

# تنظیم فونت
font_path_bold = "Vazirmatn-FD-ExtraBold.ttf"
font_path_regular = "Vazirmatn-FD-Regular.ttf"

# بررسی وجود فونت‌ها قبل از استفاده
if os.path.exists(font_path_bold):
    font_prop_bold = fm.FontProperties(fname=font_path_bold)
else:
    print("هشدار: فونت Vazirmatn-FD-ExtraBold.ttf یافت نشد. از فونت پیش‌فرض استفاده می‌شود.")
    font_prop_bold = fm.FontProperties()

if os.path.exists(font_path_regular):
    font_prop_regular = fm.FontProperties(fname=font_path_regular)
else:
    print("هشدار: فونت Vazirmatn-FD-Regular.ttf یافت نشد. از فونت پیش‌فرض استفاده می‌شود.")
    font_prop_regular = font_prop_bold

def reshape_text(text):
    return get_display(arabic_reshaper.reshape(str(text)))

def send_photo_to_telegram(token, chat_id, photo_path, caption=""):
    print("\nدر حال ارسال عکس به تلگرام...")
    api_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo_file:
            response = requests.post(api_url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                                     files={'photo': photo_file}, timeout=30)
            response.raise_for_status()
            if response.json().get("ok"): print("✅ عکس با موفقیت به تلگرام ارسال شد.")
            else: print(f"❌ خطا در ارسال عکس: {response.json()}")
    except Exception as e: print(f"خطا در فرآیند ارسال عکس: {e}")

# <<< تابع اصلاح شده برای مدیریت پیام‌های طولانی >>>
def send_message_to_telegram(token, chat_id, text):
    print("در حال ارسال پیام متنی به تلگرام...")
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # تلگرام محدودیت 4096 کاراکتر دارد. ما 4000 در نظر می‌گیریم تا ایمن باشد.
    MAX_LENGTH = 4000
    
    messages_to_send = []

    if len(text) <= MAX_LENGTH:
        messages_to_send.append(text)
    else:
        print(f"⚠️ پیام طولانی است ({len(text)} کاراکتر). در حال بخش‌بندی پیام...")
        while len(text) > MAX_LENGTH:
            # پیدا کردن آخرین خط جدید (\n) قبل از رسیدن به محدودیت
            # این کار باعث می‌شود جملات یا تگ‌های HTML وسط خط قطع نشوند
            split_index = text[:MAX_LENGTH].rfind('\n')
            
            # اگر خط جدید پیدا نشد، آخرین فاصله (Space) را پیدا کن
            if split_index == -1:
                split_index = text[:MAX_LENGTH].rfind(' ')
            
            # اگر هیچ فاصله‌ای هم نبود، به ناچار در همان 4000 برش بزن
            if split_index == -1:
                split_index = MAX_LENGTH
            
            messages_to_send.append(text[:split_index])
            text = text[split_index:].lstrip() # حذف فاصله‌های اضافی ابتدای بخش بعدی
        
        # اضافه کردن بخش باقیمانده
        if text:
            messages_to_send.append(text)

    # ارسال تک تک بخش‌ها
    for i, msg in enumerate(messages_to_send):
        payload = {'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}
        try:
            response = requests.post(api_url, json=payload, timeout=20)
            response.raise_for_status()
            if response.json().get("ok"): 
                print(f"✅ پیام متنی (بخش {i+1} از {len(messages_to_send)}) با موفقیت ارسال شد.")
            else: 
                print(f"❌ خطا در ارسال پیام متنی بخش {i+1}: {response.json()}")
        except Exception as e: 
            print(f"خطا در فرآیند ارسال پیام متنی بخش {i+1}: {e}")

# <<< تابع تحلیل هوش مصنوعی Gemini (نسخه حرفه‌ای + ایموجی هوشمند) >>>
def get_gemini_analysis(last_row, previous_row, df):
    print("\nدر حال دریافت تحلیل جامع و جذاب از هوش مصنوعی Gemini...")
    if not GEMINI_API_KEY:
        print("❌ کلید API جمنای یافت نشد.")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-flash-lite-latest') 

        # --- محاسبات کمکی ---
        money_current = last_row['ورود پول']
        money_prev = previous_row['ورود پول']
        money_change = money_current - money_prev
        
        # وضعیت حجم
        vol_current = last_row['ارزش معاملات']
        vol_avg_5 = df['ارزش معاملات'].rolling(5).mean().iloc[-1]
        vol_change_pct = ((vol_current - vol_avg_5) / vol_avg_5) * 100
        vol_status_str = f"{abs(vol_change_pct):.1f}% {'بالاتر' if vol_change_pct > 0 else 'پایین‌تر'} از میانگین ۵ روزه"

        # --- اصلاح منطق قدرت خریدار (مخصوص دیتای شما) ---
        power_current = last_row['قدرت خريد']
        power_prev = previous_row['قدرت خريد']
        
        # ترجمه وضعیت قدرت خریدار برای هوش مصنوعی
        if power_current > 0:
            power_meaning = "🟢 خریداران قوی‌تر هستند (برتری تقاضا)"
            power_trend_text = "تقویت جبهه خرید"
        elif power_current < 0:
            power_meaning = "🔴 فروشندگان قوی‌تر هستند (برتری عرضه)"
            power_trend_text = "تسلط فروشندگان"
        else:
            power_meaning = "⚪️ جنگ برابر (قدرت خنثی)"
            power_trend_text = "تعادل"

        prompt = f"""
        نقش: شما سردبیر ارشد و تحلیلگر کانال VIP "دیتا بورس" هستید.
        مخاطب: معامله‌گران حرفه‌ای که از حرف‌های کلیشه‌ای خسته‌اند و دنبال اصل مطلب هستند.
        
        لحن: صمیمی، پرانرژی، قاطع و حرفه‌ای (نه خشک و رباتیک، نه بیش از حد عامیانه).
        هدف: نوشتن یک گزارش که هم "چرایی" بازار را بگوید و هم "استراتژی" بدهد.

        📊 **داده‌های امروز ({last_row['تاریخ']}):**

        1️⃣ **پول هوشمند (Smart Money):**
           - جریان امروز: {money_current:+,.1f} میلیارد تومان
           - تغییر نسبت به دیروز: {money_change:+,.1f} میلیارد تومان.
           
        2️⃣ **قدرت خریدار/فروشنده (بسیار مهم):**
           - عدد قدرت: {power_current:.2f}
           - **تفسیر سیستم:** {power_meaning}
           - (نکته برای تحلیلگر: در این داده‌ها، عدد منفی مثل -1.26 یعنی زور فروشنده بیشتر است. عدد مثبت یعنی زور خریدار. لطفاً دقیقاً بر این اساس تحلیل کن).
           
        3️⃣ **حجم و تابلو:**
           - ارزش معاملات: {vol_current:,.1f} همت ({vol_status_str}).
           - شاخص کل: {last_row['شاخص کل']:,.0f}
           - شاخص هم‌وزن: {last_row['شاخص هم‌وزن']:,.0f}

        📝 **ساختار گزارشی که باید بنویسی:**
        
        1. **تیتر بمب:** یک تیتر کوتاه و جذاب که دقیقاً حال و هوای امروز را برساند. (مثلاً اگر قدرت منفی است: "شبیخون فروشندگان!" یا اگر مثبت است: "بازگشت قدرتمند خریدار").
        
        2. **تصویر کلی (پاراگراف اول):** خیلی سریع و روان بگو امروز چه گذشت؟ آیا بین ورود پول و شاخص هماهنگی بود؟ (به زبان ساده تحلیل کن).

        3. **زیر پوست بازار (جزئیات):**
           - وضعیت 💰 (پول): آیا پول ترسیده و خارج شده یا پارک شده؟
           - وضعیت 💪 (قدرت): با توجه به عدد ({power_current})، بگو کی برنده میدان بود؟ (اگر منفی است، بگو فروشنده‌ها میدان‌داری کردند).
           - وضعیت 📊 (حجم): آیا حجم از روند حمایت می‌کند؟

        4. **پیش‌بینی و استراتژی فردا:**
           - بدون ابهام بگو: فردا چه کنیم؟ (خرید پله‌ای؟ نقد کردن؟ یا تماشا؟).
           - روند احتمالی فردا (صعودی/نزولی/نوسانی).

        ⚠️ **خط قرمزها:**
        - زیاده‌گویی ممنوع. متن باید "شسته و رفته" باشد.
        - اگر عدد قدرت منفی است، به هیچ وجه از واژه "رشد قدرت خریدار" استفاده نکن! بگو "قدرت دست فروشنده بود".
        - تحلیل باید حس زنده بودن داشته باشد.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ خطا در تحلیل هوش مصنوعی: {e}")
        return None

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

    if ma5.iloc[-1] > ma10.iloc[-1]: analysis_points.append("<b>روند کوتاه‌مدت:</b> صعودی ✅. قرار گرفتن میانگین ۵ روزه بالاتر از ۱۰ روزه، نشان‌دهنده قدرت در کوتاه‌مدت است.")
    else: analysis_points.append("<b>روند کوتاه‌مدت:</b> نزولی ❌. قرار گرفتن میانگین ۵ روزه زیر ۱۰ روزه، می‌تواند نشانه‌ای از ضعف یا شروع فاز اصلاحی کوتاه‌مدت باشد.")
    if ma10.iloc[-1] > ma30.iloc[-1]: analysis_points.append("<b>روند اصلی:</b> صعودی ✅. میانگین ۱۰ روزه بالاتر از ۳۰ روزه قرار دارد که نشان‌دهنده حاکمیت روند صعودی در میان‌مدت است.")
    else: analysis_points.append("<b>روند اصلی:</b> نزولی ❌. میانگین ۱۰ روزه زیر ۳۰ روزه است که نشان از تضعیف روند کلی و حاکمیت فشار فروش در میان‌مدت دارد.")
    if ma5.iloc[-2] >= ma10.iloc[-2] and ma5.iloc[-1] < ma10.iloc[-1]: analysis_points.append("⚠️ <b>هشدار تقاطع:</b> میانگین ۵ روزه امروز به زیر ۱۰ روزه عبور کرد که یک سیگنال منفی کوتاه‌مدت است.")
    if ma10.iloc[-2] >= ma30.iloc[-2] and ma10.iloc[-1] < ma30.iloc[-1]: analysis_points.append("🚨 <b>تقاطع مرگ (Death Cross):</b> میانگین ۱۰ روزه امروز به زیر ۳۰ روزه رفت که هشداری جدی برای تغییر روند به نزولی است.")
    if ma5.iloc[-2] <= ma10.iloc[-2] and ma5.iloc[-1] > ma10.iloc[-1]: analysis_points.append("💡 <b>نشانه مثبت:</b> میانگین ۵ روزه امروز به بالای ۱۰ روزه عبور کرد که یک سیگنال مثبت کوتاه‌مدت است.")
    if ma10.iloc[-2] <= ma30.iloc[-2] and ma10.iloc[-1] > ma30.iloc[-1]: analysis_points.append("🚀 <b>تقاطع طلایی (Golden Cross):</b> میانگین ۱۰ روزه امروز به بالای ۳۰ روزه رفت که نشانه‌ای بسیار مهم برای تقویت روند صعودی است.")
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
    fig.text(0.5, 0.89, reshape_text("(بر مبنای ارزش معاملات خرد سهام و ص. سهامی)"), ha='center', fontproperties=font_prop_regular, fontsize=16, color='#555555')

    outer_labels = {3000: '۳ همت', 5000: '۵ همت', 10000: '۱۰ همت', 15000: '۱۵ همت', 20000: '۲۰ همت'}
    label_radius = radius + 0.15
    for value, text in outer_labels.items():
        angle_rad = np.deg2rad(180 - (value / GAUGE_DISPLAY_MAX * 180))
        x = label_radius * np.cos(angle_rad)
        y = label_radius * np.sin(angle_rad) + 0.05
        ax.text(x, y, reshape_text(text), ha='center', va='center', fontproperties=font_prop_regular, fontsize=14, color='black')
    
    fig.text(0.5, 0.05, "Telegram: @Data_Bors", ha='center', fontproperties=font_prop_regular, fontsize=14, color='gray')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.2, 1.35)
    
    filename = f'Fear_Greed_Gauge-{file_str}.png'
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"شاخص نهایی با موفقیت در فایل '{filename}' ذخیره شد.")
    return filename


# --- مراحل اصلی اجرا ---
def main():
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("❌ متغیرهای محیطی تلگرام (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) تنظیم نشده‌اند.")
        return

    print("در حال دریافت داده‌ها...")
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
    if len(data) < 2: print("داده کافی برای تحلیل مقایسه‌ای وجود ندارد."); return

    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
    last_row, previous_row = df.iloc[-1], df.iloc[-2]
    last_value = last_row['ارزش معاملات']
    last_date = last_row['تاریخ']
    generated_filename = create_fear_greed_gauge_real_scale(last_value, now_str_file)

    if generated_filename:
        # تغییر در اینجا: اضافه شدن شرط برای مقادیر بالای ۲۰ هزار (طمع خیلی شدید)
        status_short = "وضعیت: " + ("<b>ترس شدید</b> 🥶" if last_value < 3000 else "<b>ترس</b> 😟" if last_value < 5000 else "<b>خنثی</b> 😐" if last_value < 10000 else "<b>طمع</b> 😊" if last_value < 15000 else "<b>طمع شدید</b> 🤩🔥" if last_value < 20000 else "<b>طمع خیلی شدید</b> 🤑🚀")
        
        photo_caption = "\n".join([f"<b>📊 شاخص ترس و طمع بازار سهام</b>", f"🗓️ تاریخ: {last_date}", f"<b>مقدار فعلی:</b> {last_value:,.1f} میلیارد تومان", status_short, "\n🆔 @Data_Bors"])
        send_photo_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, generated_filename, photo_caption)

        full_message_blocks = []
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
            
            ath_record_badge = ""
            ath_message = ""
            if len(df) > 1:
                previous_ath = df[key][:-1].max()
                if current_idx > previous_ath:
                    ath_record_badge = " (🚀 <b>رکورد جدید!</b>)"
                ath_message = f"  - سقف تاریخی: {int(max(current_idx, previous_ath)):,.0f}"
            else:
                ath_message = f"  - سقف تاریخی: {current_idx:,.0f}"

            yearly_subset = df.tail(252)
            yearly_low = yearly_subset[key].min()
            yearly_high = yearly_subset[key].max()
            
            dist_from_high = (current_idx - yearly_high) / yearly_high * 100 if yearly_high else 0
            dist_from_low = (current_idx - yearly_low) / yearly_low * 100 if yearly_low else 0

            yearly_high_message = f"📈<code>{int(yearly_high):,.0f}</code> (<b>{dist_from_high:+.1f}%</b>)"
            if current_idx >= yearly_high: yearly_high_message = f"📈<code>{current_idx:,.0f}</code> (<b>رکورد جدید سال!</b>)"
            
            yearly_range_message = f"  - بازه یکساله (📉<code>{int(yearly_low):,.0f}</code> (<b>{dist_from_low:+.1f}%</b>) | {yearly_high_message})"

            idx_parts = [
                f"⚪️ <b>شاخص {name}</b>" if name == 'کل' else f"⚖️ <b>شاخص {name}</b>",
                f"  - مقدار فعلی: <code>{current_idx:,.0f}</code>{ath_record_badge} <b>({idx_change:+,.0f} | {idx_percent:+.2f}%)</b> {'⬆️' if idx_change >= 0 else '⬇️'}",
                ath_message, yearly_range_message
            ]
            
            proximity_alert = generate_proximity_alert(current_idx, yearly_subset[key][:-1].max(), yearly_low, "سقف یکساله", "کف یکساله")
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

        ai_analysis = get_gemini_analysis(last_row, previous_row, df)
        if ai_analysis:
            ai_message = ai_analysis + "\n\n" + "\n".join([f"<i>این تحلیل توسط هوش مصنوعی (Google Gemini) تولید شده است.</i>", "🆔 @Data_Bors"])
            # اینجا تابع اصلاح شده استفاده می‌شود که متن طولانی را مدیریت می‌کند
            send_message_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ai_message)

    print(f"\n--- عملیات با موفقیت به پایان رسید. ---")

if __name__ == "__main__":
    main()
