# راه‌اندازی نهایی — همه‌چیز توی این zip هست

## ۱. Extract کن
`emotion_music_recommender` رو باز کن. اگه پروژه‌ی قدیمی از قبل داری، **کل این پوشه رو جایگزینش کن** (یا محتوای `src/` و `weights/` رو کامل overwrite کن) — چون این نسخه شامل همه‌ی اصلاحات این گفتگوست:

```
emotion_music_recommender/
  requirements.txt
  weights/                    <- ۴ فایل، آخرین نسخه (چهره ۶۷٪، متن ۵۷٪)
  src/
    streamlit_app.py          <- انگلیسی، وبکم، کالیبراسیون شخصی
    demo.py
    config.py, emotion_space.py, walkthrough.py
    fusion/, recommender/, text/, vision/, tests/
    train_text_emotion.py + .ipynb
    train_fer_cnn.py + .ipynb
    diagnose_env.py, test_face_backends.py, diagnose_text_model.py
```

## ۲. نصب پکیج‌ها

```powershell
cd emotion_music_recommender
pip install -r requirements.txt
```

(mediapipe عمداً توی این فایل نیست — دلیلش تعارض protobuf با tensorflow است؛ کد بدونش هم با Haar کامل کار می‌کنه. اگه بخوای دستی امتحانش کنی، توضیحش داخل خود `requirements.txt` هست.)

## ۳. تست کن

```powershell
cd src
python -m unittest discover -s tests -t .
```
باید `Ran 20 tests ... OK` بده — من همین الان دقیقاً همین فایل‌ها رو تست کردم، پاس شد.

## ۴. دموی زنده

```powershell
streamlit run streamlit_app.py
```

توی sidebar:
- یک جمله بنویس (انگلیسی بهتر جواب می‌ده)
- منبع عکس رو انتخاب کن: **Upload** یا **Webcam**
- اختیاری: بخش «🪞 Personal calibration» رو باز کن و برای هر حس یک عکس مرجع از خودت با وبکم بگیر
- بزن **▶ Run**

## چیزهایی که لازم نیست (فقط برای ری‌ترین کردن مدل‌ها)

اگه فقط می‌خوای دمو رو تست کنی، **این‌ها رو دانلود نکن** — وزن‌های آماده کافیه:
- `facial_expressions/` (دیتاست چهره، ~۵۰۰ مگابایت)
- CK+ دیتاست (برای بهبود fear/disgust)
- `dataset_text/` (GoEmotions)
- ISEAR دیتاست

اگه بعداً خواستی نوت‌بوک‌های آموزش رو دوباره اجرا کنی، به من بگو تا لینک دقیق هر کدوم رو دوباره بدم.
