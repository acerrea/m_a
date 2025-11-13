def clean_text_for_speech(html_text):
    """تگ‌های HTML را از متن حذف می‌کند تا برای خواندن توسط TTS آماده شود."""
    soup = bs_for_clean(html_text, "html.parser")
    # جایگزینی برخی علائم برای تلفظ بهتر
    text = soup.get_text()
    text = text.replace(":", "، ").replace(" H:", " H ").replace(" M:", " M ")
    return text

def convert_text_to_speech(text, filename="analysis_audio.mp3"):
    """متن را به یک فایل صوتی mp3 تبدیل می‌کند."""
    print("در حال تبدیل متن به صوت...")
    try:
        tts = gTTS(text=text, lang='fa', slow=False)
        tts.save(filename)
        print(f"✅ فایل صوتی با موفقیت در '{filename}' ذخیره شد.")
        return filename
    except Exception as e:
        print(f"❌ خطا در تبدیل متن به صوت: {e}")
        return None

def send_audio_to_telegram(token, chat_id, audio_path, caption=""):
    """فایل صوتی را به تلگرام ارسال می‌کند."""
    print("در حال ارسال فایل صوتی به تلگرام...")
    api_url = f"https://api.telegram.org/bot{token}/sendAudio"
    try:
        with open(audio_path, 'rb') as audio_file:
            response = requests.post(
                api_url,
                data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                files={'audio': audio_file},
                timeout=60 # زمان بیشتر برای آپلود فایل
            )
            response.raise_for_status()
            if response.json().get("ok"):
                print("✅ فایل صوتی با موفقیت به تلگرام ارسال شد.")
            else:
                print(f"❌ خطا در ارسال فایل صوتی: {response.json()}")
    except Exception as e:
        print(f"خطا در فرآیند ارسال فایل صوتی: {e}")
