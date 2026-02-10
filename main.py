import requests
import random
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import asyncio
import time
import math
import matplotlib.pyplot as plt
import io
import mplfinance as mpf
import pandas as pd
from datetime import datetime
import pandas_ta as talib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS']
import matplotlib.font_manager as fm


bot_token = '8379238010:AAFjcdWogdYjBaje6-1FPAxEKQHZ0ZXELBs'
ALLOWED_USER_IDS = {7223388540, 1149436347}

group_analysis_limits = {}  
group_analysis_counts = {}  
added_groups = set()

PUBLIC_CHANNEL_ID = -1002540547085
PRIVATE_CHANNEL_ID = -1002649275547

used_patterns = []
trade_photo_file_id = None
default_trade_photo_url = "https://i.imgur.com/JqYe5vn.png" 
open_trades = []
auto_trade_interval = None
auto_trade_task = None
waiting_for_private_coin = {} 
waiting_for_private_frame = {}

def reverse_arabic_text(text):
    """يعكس النص العربي ليتم عرضه بشكل صحيح في matplotlib."""
    return text[::-1]
import pandas as pd
import mplfinance as mpf
import talib
import io
import matplotlib.pyplot as plt

def create_professional_chart(candles, coin_name, support, resistance, current_price, detected_patterns, pattern_points_dict=None):
    try:
        # إعداد البيانات
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df.set_index('time')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # مؤشرات SMA
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)

        # ألوان مخصصة للشارت
        mc = mpf.make_marketcolors(
            up='lime',        # شموع صاعدة
            down='tomato',    # شموع نازلة
            wick='white',     # الفتائل
            edge='inherit',
            volume='in'
        )
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            facecolor='#1e1e2f',  # خلفية داكنة أنيقة
            figcolor='#1e1e2f',
            gridcolor='gray',
            y_on_right=False
        )

        # خطوط إضافية (Support, Resistance, SMA)
        apds = [
            mpf.make_addplot([resistance]*len(df), color='yellow', linestyle='-', linewidths=2, alpha=0.7),
            mpf.make_addplot([support]*len(df), color='red', linestyle='-', linewidths=2, alpha=0.7),
            mpf.make_addplot(df['SMA_50'], color='cyan', linestyle='--', linewidths=2, alpha=0.9),
            mpf.make_addplot(df['SMA_20'], color='orange', linestyle='--', linewidths=2, alpha=0.9)
        ]

        # رسم الشارت
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=s,
            addplot=apds,
            volume=False,
            returnfig=True,
            figsize=(12,6),
            title=f"ـﻟ ﻲﻨﻓ ﻞﻴﻠﺤﺗ {coin_name}"
        )

        ax = axlist[0]

        # إضافة علامة مائية
        ax.text(0.5, 0.5, 'XID TRADING', transform=ax.transAxes,
                fontsize=40, color='white', alpha=0.15,
                ha='center', va='center', rotation=30)

        # كتابة الأنماط المكتشفة
        patterns_text = f":ﺔﻜﺘﺸﻤﺘﻟﺍ ﻂﻤﻨﻟﺍ {', '.join(detected_patterns)}"
        ax.text(0.01, 0.95, patterns_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='#2e2e3f', alpha=0.8))

        # حفظ الصورة في buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        return buf

    except Exception as e:
        print(f"خطأ في إنشاء الشارت: {e}")
        return None
def get_random_coin():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=5000&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        valid_coins = [
            coin for coin in data
            if all(x not in coin['symbol'].lower() for x in ['btc', 'eth', 'bnb', 'usdt', 'usd'])
            and coin.get('current_price', 0) > 0
            and coin.get('total_volume', 0) > 50000 
        ]
        if valid_coins:
            return random.choice(valid_coins)
    return None

# دالة لجلب بيانات الشموع مع الحجم
def get_candle_data(coin_id, days, max_candles=200):
    try:
        # جلب بيانات السعر
        price_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        price_data = requests.get(price_url).json()

        # جلب بيانات الحجم
        volume_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={int(time.time())-days*86400}&to={int(time.time())}"
        volume_data = requests.get(volume_url).json()

        prices = price_data.get("prices", [])
        volumes = volume_data.get("total_volumes", [])

        # دمج البيانات
        candles = []
        for i in range(min(len(prices), len(volumes), max_candles)):
            candles.append({
                "time": prices[i][0],
                "open": prices[i][1] if i == 0 else prices[i-1][1],
                "high": max(prices[i][1], prices[i-1][1] if i > 0 else prices[i][1]),
                "low": min(prices[i][1], prices[i-1][1] if i > 0 else prices[i][1]),
                "close": prices[i][1],
                "volume": volumes[i][1] if i < len(volumes) else 0
            })

        return candles[-max_candles:]
    except Exception as e:
        print(f"خطأ في جلب بيانات الشموع: {e}")
        return []

# حساب الدعم والمقاومة باستخدام النقاط المحورية
def calculate_pivot_points(candles, lookback=50):
    if not candles or len(candles) < lookback:
        return 0, 0

    close_prices = [c['close'] for c in candles]
    high_prices = [c['high'] for c in candles]
    low_prices = [c['low'] for c in candles]

    last_high = max(high_prices[-lookback:])
    last_low = min(low_prices[-lookback:])
    last_close = close_prices[-1]

    # حساب مستويات فيبوناتشي
    pp = (last_high + last_low + last_close) / 3
    r1 = pp + (last_high - last_low) * 0.382
    r2 = pp + (last_high - last_low) * 0.618
    s1 = pp - (last_high - last_low) * 0.382
    s2 = pp - (last_low + last_high - 2 * last_close) * 0.618 

    support = min(s1, s2)
    resistance = max(r1, r2)

    return support, resistance

def identify_all_patterns(candles):
    open_prices = np.array([c['open'] if 'open' in c else c['close'] for c in candles])
    high_prices = np.array([c['high'] for c in candles])
    low_prices = np.array([c['low'] for c in candles])
    close_prices = np.array([c['close'] for c in candles])
    times_ms = np.array([c['time'] for c in candles])

    detected_patterns = {}
    pattern_points = {}

    all_talib_functions = [func for func in dir(talib) if func.startswith('CDL')]

    for pattern_name in all_talib_functions:
        pattern_function = getattr(talib, pattern_name)
        result = pattern_function(open_prices, high_prices, low_prices, close_prices)
        pattern_indices = np.where(result != 0)[0]
        if len(pattern_indices) > 0:
            detected_patterns[pattern_name] = ("شراء" if result[pattern_indices[-1]] > 0 else "بيع")

            points = []
            for index in pattern_indices:
                points.append((times_ms[index], close_prices[index])) # استخدام وقت الإغلاق كنقطة افتراضية
            pattern_points[pattern_name] = points

    return detected_patterns, pattern_points

# تحليل العملة بشكل متقدم مع عرض أهم نموذج أو اثنين
def analyze_coin(name, symbol, current_price, candles, interval, support, resistance):
    if not candles:
        return f"❌ لا توجد بيانات كافية لتحليل {symbol}. يرجى الانتظار قليلة او قد تكون العملة غير موجودة"

    # تحديد أهم النماذج الفنية
    top_patterns, _ = identify_top_patterns(candles)
    patterns_details = []
    for pattern_name, signal in top_patterns.items():
        # هنا ممكن نضيف تفاصيل بسيطة عن كل نموذج بناءً على اسمه
        description = ""
        if "ENGULFING" in pattern_name:
            description = "نموذج ابتلاعي يشير إلى انعكاس محتمل في الاتجاه."
        elif "MARUBOZU" in pattern_name:
            description = "شمعة ماروبوزو قوية تدل على زخم قوي في اتجاه واحد."
        elif "DOJI" in pattern_name:
            description = "شمعة دوجي تدل على تردد في السوق وإمكانية انعكاس."
        elif "HAMMER" in pattern_name:
            description = "نموذج المطرقة يشير إلى انعكاس صعودي محتمل بعد اتجاه هابط."
        elif "SHOOTINGSTAR" in pattern_name:
            description = "نموذج النجمة الساقطة يشير إلى انعكاس هبوطي محتمل بعد اتجاه صاعد."
        else:
            description = "تم اكتشاف نموذج." # وصف افتراضي

        pattern_name_clean = pattern_name.replace('CDL', '')
        patterns_details.append(f"**{pattern_name_clean} ({signal}):** {description}")

    patterns_text = "\n".join(patterns_details) if patterns_details else "لا يوجد أنماط محددة مهمة."

    # حساب الأهداف الإيجابية
    price_range = resistance - support
    positive_targets = [
        resistance + price_range * 0.382,
        resistance + price_range * 0.618,
        resistance + price_range * 1.0
    ]

    negative_targets = [
        support - price_range * 0.382,
        support - price_range * 0.618,
        support - price_range * 1.0
    ]

    stop_loss_value = calculate_stop_loss(candles, current_price,support)

    close_prices = [c['close'] for c in candles]
    sma_20 = talib.SMA(np.array(close_prices), timeperiod=20)[-1]
    sma_50 = talib.SMA(np.array(close_prices), timeperiod=50)[-1]
    rsi = talib.RSI(np.array(close_prices), timeperiod=14)[-1]

    # إنشاء الرسالة
    message = f"""
📊 *تحليل فني متقدم لـ {name} ({symbol.upper()})*

💵 *السعر الحالي:* ${current_price:.6f}
📈 *الاتجاه العام:* {'صاعد' if sma_20 > sma_50 else 'هابط' if sma_20 < sma_50 else 'جانبي'}

🛡️ *مستويات رئيسية:*
- الدعم القوي: ${support:.6f}
- المقاومة القوية: ${resistance:.6f}

🎯 *الأهداف الإيجابية (في حال الاختراق):*
1. ${positive_targets[0]:.6f} (ربح +{((positive_targets[0]-current_price)/current_price)*100:.2f}%)
2. ${positive_targets[1]:.6f} (ربح +{((positive_targets[1]-current_price)/current_price)*100:.2f}%)
3. ${positive_targets[2]:.6f} (ربح +{((positive_targets[2]-current_price)/current_price)*100:.2f}%)

📉 *الأهداف السلبية (في حال كسر الدعم أو النموذج الهابط):*
1. ${negative_targets[0]:.6f} (خسارة {((current_price-negative_targets[0])/current_price)*100:.2f}%)
2. ${negative_targets[1]:.6f} (خسارة {((current_price-negative_targets[1])/current_price)*100:.2f}%)
3. ${negative_targets[2]:.6f} (خسارة {((current_price-negative_targets[2])/current_price)*100:.2f}%)

⛔ *وقف الخسارة:* ${stop_loss_value:.6f} (خسارة {((current_price-stop_loss_value)/current_price)*100:.2f}%)

📌 *أهم الأنماط الفنية المحددة:*
{patterns_text}

📈 *المؤشرات الفنية:*
- المتوسط المتحرك 20 فترة: ${sma_20:.6f}
- المتوسط المتحرك 50 فترة: ${sma_50:.6f}
- مؤشر RSI (14): {rsi:.2f} {'(مشترى شديد)' if rsi < 30 else '(مشترى)' if rsi < 50 else '(بيع)' if rsi > 70 else '(بيع شديد)' if rsi > 80 else '(محايد)'}

⚠️ *تحذير المخاطرة:*
التداول محفوف بالمخاطر، لا تستثمر أكثر مما تستطيع تحمل خسارته.
"""
    return message

def calculate_stop_loss(candles, current_price, support_level):
    if not candles or len(candles) < 50 or support_level is None:
        return float('nan')
    high_prices = [c['high'] for c in candles]
    low_prices = [c['low'] for c in candles]
    close_prices = [c['close'] for c in candles]

    atr = talib.ATR(
        np.array(high_prices),
        np.array(low_prices),
        np.array(close_prices),
        timeperiod=50
    )[-1]

    # وضع وقف الخسارة أسفل مستوى الدعم بهامش يعتمد على جزء من ATR
    stop_loss = support_level - (atr * 0.75) # يمكنك تعديل هذا المعامل (0.75)
    return max(stop_loss, 0)
async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text_unicode = f"""
✨ **نـبـذة عـن هـذا الـبـوت** ✨

هذا الـبـوت يـقـدم تـحـلـيـلات فـنـيـة مـتـقـدمـة لـلـعـمـلات الـرقـمـيـة، مـدعـومـًا بـأحـدث الأدوات والـمـؤشـرات. يـهـدف إلـى تـوفـيـر رؤى قـيـمـة لـمـسـاعـدتـك فـي اتـخـاذ قـرارات تـداول مـسـتـنـيـرة.

⚠️ **تـنـويـه هـام وتـحـذيـر:** ⚠️
الـتـحـلـيـلات والـتـوصـيـات الـمـقـدمـة فـي هـذا الـبـوت هـي مـجـرد تـوقـعـات مـبـنـيـة عـلـى عـلـوم وحـسـابـات ومـؤشـرات فـنـيـة. صـعـود أو هـبـوط أي عـمـلـة يـقـع فـي عـلـم الـغـيـب وحـده، ونـحـن لا نـتـحـمـل أي مـسـؤولـيـة قـانـونـيـة أو مـالـيـة تـجاه أي قـرارات تـتـخـذونـهـا بـنـاءً عـلـى هـذه الـتـحـلـيـلات. يـرجـى إجـراء أبـحـاثـكـم الـخـاصـة والـتـحـقـق مـن مـصـادر مـتـعـددة قـبـل اتـخـاذ أي قـرار اسـتـثـمـاري. **كـمـا نـنـصـح بـشـدة بـالـبـحـث والـسـؤال عـن حـكـم الـشـريـعـة الإسـلامـيـة فـي الـعـمـلـة الـرقـمـيـة الـتـي تـنـوون تـداولـهـا، والـتـأكـد مـن حـرمـتـهـا أو حِـلِّـهـا وفـقـًا لـمـعـتـقـداتـكـم الـديـنـيـة قـبـل الـدخـول فـي أي صـفـقـات.**

🧑‍💼 **الـمـحـلـلـون والـمـدراء:**

👤👑 @pharaoh_GPA
👤👑 @mmderbi

🧑‍💻 **الـمـطـورون والـمـسـاهـمـون فـي بـنـاء الـبـوت وهـيـكـلـة الـتـحـلـيـل:**

**مبرمج | مطور:** 👨‍💻👑 @mmderbi
**سـاهـم فـي بـنـاء أسـاس الـبـوت وهـيـكـلـة الـتـحـلـيـل:** 🧑‍💻👑 @pharaoh_GPA


في حال رغبتك في برمجة بوت خاص يرجى التواصل مع @mmderbi

نـتـمـنـى لـكـم تـداولـًا مـوفـقـًا ومـسـؤولـًا!
"""
    await update.message.reply_text(about_text_unicode) # لا نستخدم parse_mode
async def send_telegram_message(message, chat_id, candles=None, coin_name=None,
                                    support=None, resistance=None, current_price=None, detected_patterns=None, pattern_points=None):
    try:
        bot = telegram.Bot(token=bot_token)

        # إنشاء الشارت إذا توفرت البيانات (بدون رسم الأنماط)
        chart_buffer = None
        if candles and coin_name and support is not None and resistance is not None and current_price is not None:
            chart_buffer = create_professional_chart(candles, coin_name, support, resistance, current_price, [], {})

        # إنشاء لوحة المفاتيح المضمنة مع رابط القناة

        if chart_buffer:
            await bot.send_photo(
                chat_id=chat_id,
                photo=chart_buffer,
                caption=message,
                parse_mode="Markdown",
             
            )
        else:
            await bot.send_photo(
                chat_id=chat_id,
                photo=default_trade_photo_url,
                caption=message,
                parse_mode="Markdown",
      
            )

    except Exception as e:
        print(f"فشل في إرسال الرسالة: {e}")
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="Markdown",
            )
        except Exception as e2:
            print(f"فشل في إرسال الرسالة بدون صورة: {e2}")

def identify_top_patterns(candles, top_n=2):
    open_prices = np.array([c['open'] if 'open' in c else c['close'] for c in candles])
    high_prices = np.array([c['high'] for c in candles])
    low_prices = np.array([c['low'] for c in candles])
    close_prices = np.array([c['close'] for c in candles])

    found_patterns = {}
    all_talib_functions = [func for func in dir(talib) if func.startswith('CDL')]

    pattern_translations = {
        'CDL_2CROWS': 'شمعتان سوداوان',
        'CDL_3BLACKCROWS': 'ثلاثة غربان سوداء',
        'CDL_3INSIDE': 'ثلاثة للداخل صاعدة/هابطة',
        'CDL_3LINESTRIKE': 'ضربة الثلاثة خطوط',
        'CDL_3OUTSIDE': 'ثلاثة للخارج صاعدة/هابطة',
        'CDL_3STARSINSOUTH': 'ثلاثة نجوم في الجنوب',
        'CDL_ADVANCEBLOCK': 'كتلة تقدم',
        'CDL_BELTHOLD': 'حزام الإمساك صاعد/هابط',
        'CDL_BREAKAWAY': 'انفصال',
        'CDL_CLOSINGMARUBOZU': 'ماروبوزو الإغلاق',
        'CDL_CONCEALBABYSWALL': 'ابتلاع الطفل المخفي',
        'CDL_COUNTERATTACK': 'هجوم مضاد',
        'CDL_DARKCLOUDCOVER': 'غطاء السحابة الداكنة',
        'CDL_DOJI': 'دوجي',
        'CDL_DOJISTAR': 'نجمة دوجي',
        'CDL_DRAGONFLYDOJI': 'دوجي اليعسوب',
        'CDL_ENGULFING': 'ابتلاعي صاعد/هابط',
        'CDL_EVENINGDOJISTAR': 'نجمة دوجي المساء',
        'CDL_EVENINGSTAR': 'نجمة المساء',
        'CDL_GAPSIDESIDEWHITE': 'فجوة جنبًا إلى جنب بيضاء',
        'CDL_GRAVESTONEDOJI': 'دوجي شاهد القبر',
        'CDL_HAMMER': 'مطرقة',
        'CDL_HANGINGMAN': 'رجل مشنوق',
        'CDL_HARAMI': 'هارامي صاعد/هابط',
        'CDL_HARAMICROSS': 'هارامي كروس صاعد/هابط',
        'CDL_HIGHWAVE': 'موجة عالية',
        'CDL_HIKKAKE': 'هيكاكيه صاعد/هابط',
        'CDL_HIKKAKEMOD': 'هيكاكيه معدل صاعد/هابط',
        'CDL_HOMINGPIGEON': 'حمام زاجل',
        'CDL_IDENTICAL3CROWS': 'ثلاثة غربان متطابقة',
        'CDL_INNECK': 'في العنق',
        'CDL_INVERTEDHAMMER': 'مطرقة مقلوبة',
        'CDL_KICKING': 'ركلة صاعدة/هابطة',
        'CDL_KICKINGBYLENGTH': 'ركلة بالطول صاعدة/هابطة',
        'CDL_LADDERBOTTOM': 'قاع السلم',
        'CDL_LONGLEGGEDDOJI': 'دوجي طويل الأرجل',
        'CDL_LONGLINE': 'خط طويل',
        'CDL_MARUBOZU': 'ماروبوزو',
        'CDL_MATCHINGLOW': 'قاع مطابق',
        'CDL_MATHOLD': 'حركة الصعود والهبوط',
        'CDL_MORNINGDOJISTAR': 'نجمة دوجي الصباح',
        'CDL_MORNINGSTAR': 'نجمة الصباح',
        'CDL_ONNECK': 'على العنق',
        'CDL_PIERCING': 'اختراق',
        'CDL_RICKSHAWMAN': 'رجل الريكشو',
        'CDL_RISEFALL3METHODS': 'صعود وهبوط بثلاث طرق',
        'CDL_SEPARATINGLINES': 'خطوط فاصلة',
        'CDL_SHOOTINGSTAR': 'نجمة ساقطة',
        'CDL_SHORTLINE': 'خط قصير',
        'CDL_SPINNINGTOP': 'قمة مغزولة',
        'CDL_STALLEDPATTERN': 'نموذج متوقف',
        'CDL_STICKSANDWICH': 'ساندويتش العصا',
        'CDL_TAKURI': 'تاكوري (خطاف)',
        'CDL_TASUKIGAP': 'فجوة تاسكي',
        'CDL_THRUSTING': 'اندفاع',
        'CDL_TRISTAR': 'تراي ستار',
        'CDL_UNIQUE3RIVER': 'نهر فريد بثلاثة',
        'CDL_UPSIDEGAP2CROWS': 'فجوة صاعدة بغربان سوداء'
    }

    # اكتشاف جميع النماذج
    for pattern_name_en in all_talib_functions:
        pattern_function = getattr(talib, pattern_name_en)
        result = pattern_function(open_prices, high_prices, low_prices, close_prices)
        pattern_indices = np.where(result != 0)[0]
        if len(pattern_indices) > 0:
            signal = "شراء" if result[pattern_indices[-1]] > 0 else "بيع"
            found_patterns[pattern_name_en] = signal

    # اختيار آخر أهم N نموذج تم اكتشافه وترجمتها
    top_n_patterns_en = list(found_patterns.keys())[-top_n:]
    top_patterns_translated = {}
    for pattern_name_en in top_n_patterns_en:
        signal = found_patterns[pattern_name_en]
        pattern_name_ar = pattern_translations.get(pattern_name_en, pattern_name_en.replace('CDL_', ''))
        top_patterns_translated[pattern_name_ar] = signal

    return top_patterns_translated, {} # مش هنرجع نقاط للرسم في الوقت الحالي
def is_allowed_user(user_id):
    return user_id in ALLOWED_USER_IDS

# التحقق من أن الرسالة قادمة من مجموعة مضافة
def is_added_group(chat_id):
    return chat_id in added_groups

# إضافة مجموعة جديدة (للمستخدمين المصرح لهم فقط)
async def add_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    if not is_allowed_user(user_id):
        await update.message.reply_text("عذراً، ليس لديك صلاحية لإضافة مجموعات.")
        return

    if update.message.chat.type == 'group' or update.message.chat.type == 'supergroup':
        if chat_id not in added_groups:
            added_groups.add(chat_id)
            # لا يتم تعيين حد افتراضي هنا، سيتم تعيينه بواسطة المستخدم
            group_analysis_counts[chat_id] = 0
            await update.message.reply_text("تمت إضافة المجموعة بنجاح. يمكنك الآن تعيين حد التحليلات لهذه المجموعة باستخدام الأمر /setlimmet.")
        else:
            await update.message.reply_text("هذه المجموعة مضافة بالفعل.")
    else:
        await update.message.reply_text("يمكن إضافة المجموعات فقط.")

# تعيين حد التحليلات للمجموعة الحالية (للمستخدمين المصرح لهم فقط)
async def set_group_limit_current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    if not is_allowed_user(user_id):
        await update.message.reply_text("عذراً، ليس لديك صلاحية لتعديل حدود المجموعات.")
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("استخدام الأمر: /setlimmet LIMIT (مثال: /setlimmet 10)")
        return

    try:
        limit = int(context.args[0])
        if chat_id in added_groups:
            group_analysis_limits[chat_id] = limit
            group_analysis_counts[chat_id] = min(group_analysis_counts.get(chat_id, 0), limit) # التأكد من عدم تجاوز الحد الحالي
            await update.message.reply_text(f"تم تعيين الحد الأقصى للتحليلات لهذه المجموعة إلى {limit}.")
        else:
            await update.message.reply_text("هذه المجموعة غير مضافة. قم بإضافتها أولاً باستخدام /addgroup.")
    except ValueError:
        await update.message.reply_text("يجب أن يكون الحد رقمًا صحيحًا.")

# التحقق من عدد التحليلات المتاحة للمجموعة الحالية (لأي مستخدم)
async def check_group_limit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id in added_groups:
        limit = group_analysis_limits.get(chat_id, "غير محدد")
        count = group_analysis_counts.get(chat_id, 0)
        if limit == "غير محدد":
            await update.message.reply_text(f"الحد الأقصى للتحليلات في هذه المجموعة: غير محدد.")
        else:
            remaining = limit - count
            await update.message.reply_text(f"الحد الأقصى للتحليلات في هذه المجموعة: {limit}\nالتحليلات المتاحة: {remaining}")
    else:
        await update.message.reply_text("هذه المجموعة غير مضافة بعد.")

# تحليل عملة محددة في الخاص (للمستخدمين المصرح لهم فقط)
async def private_analyze_coin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    if not is_allowed_user(user_id):
        await query.answer("عذراً، لست مصرحاً لك باستخدام هذا الزر.")
        return

    await query.message.reply_text("أرسل اسم العملة التي تريد تحليلها (مثال: BTC).")
    context.user_data['waiting_for_private_coin'] = True
    await query.answer()

async def handle_private_coin_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not is_allowed_user(user_id) or not context.user_data.get('waiting_for_private_coin'):
        return

    coin_name = update.message.text.lower()
    context.user_data['waiting_for_private_coin'] = False
    await update.message.reply_text(f"أرسل الإطار الزمني المطلوب للتحليل (مثال: 15m, 4h, 1d).")
    context.user_data['waiting_for_private_frame'] = coin_name

async def handle_private_frame_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not is_allowed_user(user_id) or not context.user_data.get('waiting_for_private_frame'):
        return

    frame = update.message.text.lower()
    coin_name = context.user_data.pop('waiting_for_private_frame')
    await analyze_specific_coin(update, context, coin_name, frame, private_call=True)

async def analyze_specific_coin(update: Update, context: ContextTypes.DEFAULT_TYPE, coin_name, interval_str, private_call=False):
    chat_id = update.message.chat_id if not private_call else update.message.from_user.id

    coin = get_coin_by_name(coin_name)
    if not coin:
        await update.message.reply_text("لم يتم العثور على العملة.")
        return

    coin_id, name, symbol, current_price = coin['id'], coin['name'], coin['symbol'].upper(), coin['current_price']
    days = 1 if interval_str == '15m' else 7 if interval_str == '4h' else 30 if interval_str == '1d' else None
    if days is None:
        await update.message.reply_text("فريم غير صالح.")
        return

    candles = get_candle_data(coin_id, days)
    if candles:
        support, resistance = calculate_pivot_points(candles)
        detected_patterns, pattern_points = identify_all_patterns(candles)
        analysis = analyze_coin(name, symbol, current_price, candles, interval_str, support, resistance)
        await send_telegram_message(
            analysis,
            chat_id,
            candles,
            name,
            support,
            resistance,
            current_price,
            detected_patterns,
            pattern_points
        )
    else:
        await update.message.reply_text("⏳ يرجى الانتظار قليلًا أو التأكد من كتابة اسم العملة بشكل صحيح. يمكنك المحاولة مرة أخرى لاحقًا.")

# معالجة أمر GPA في المجموعة
async def gpa_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    if not is_added_group(chat_id):
        return

    if chat_id not in group_analysis_limits:
        await update.message.reply_text("لم يتم تعيين حد للتحليلات لهذه المجموعة بعد. قم بتعيين الحد باستخدام /setlimmet LIMIT.")
        return

    if chat_id in group_analysis_counts and group_analysis_counts[chat_id] >= group_analysis_limits.get(chat_id, 0):
        await update.message.reply_text("تم تجاوز الحد الأقصى للتحليلات لهذه المجموعة.")
        return

    if len(context.args) != 2:
        await update.message.reply_text("استخدام الأمر: /gpa COINNAME FRAME (مثال: /gpa BTC 15m)")
        return

    coin_name, interval_str = context.args[0].lower(), context.args[1].lower()

    coin = get_coin_by_name(coin_name)
    if not coin:
        await update.message.reply_text("⏳ يرجى الانتظار قليلًا أو التأكد من كتابة اسم العملة بشكل صحيح. يمكنك المحاولة مرة أخرى بعد دقيقة.")
        return

    coin_id, name, symbol, current_price = coin['id'], coin['name'], coin['symbol'].upper(), coin['current_price']
    days = 1 if interval_str == '15m' else 7 if interval_str == '4h' else 30 if interval_str == '1d' else None
    if days is None:
        await update.message.reply_text("فريم غير صالح.")
        return

    candles = get_candle_data(coin_id, days)
    if candles:
        support, resistance = calculate_pivot_points(candles)
        detected_patterns, pattern_points = identify_all_patterns(candles)
        analysis = analyze_coin(name, symbol, current_price, candles, interval_str, support, resistance)
        await send_telegram_message(
            analysis,
            chat_id,
            candles,
            name,
            support,
            resistance,
            current_price,
            detected_patterns,
            pattern_points
        )
        group_analysis_counts[chat_id] = group_analysis_counts.get(chat_id, 0) + 1
    else:
        await update.message.reply_text("⏳ يرجى الانتظار قليلًا أو التأكد من كتابة اسم العملة بشكل صحيح. يمكنك المحاولة مرة أخرى لاحقًا.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not is_allowed_user(user_id):
        await update.message.reply_text("عذراً، لست مصرحاً لك باستخدام هذا البوت.")
        return

    buttons = [
        [InlineKeyboardButton("إضافة مجموعة", callback_data='add_new_group')],
        [InlineKeyboardButton("تعيين حد التحليلات لمجموعة", callback_data='set_limit_group')],
        [InlineKeyboardButton("تحليل عملة محددة", callback_data='private_analyze')],
        [InlineKeyboardButton("نشر تحليل عشوائي (عامة)", callback_data='publish_random_public')],
        [InlineKeyboardButton("نشر تحليل عشوائي (خاصة)", callback_data='publish_random_private')],
        [InlineKeyboardButton("إضافة صورة للصفقات", callback_data='add_trade_photo')],
        [InlineKeyboardButton("تحديد وقت الإرسال التلقائي (قريباً)", callback_data='set_auto_trade_time')]
    ]
    await update.message.reply_text("اختر إجراء:", reply_markup=InlineKeyboardMarkup(buttons))
async def publish_random_analysis(chat_id):
    coin = get_random_coin()
    if not coin:
        return "لم يتم العثور على عملة صالحة."

    coin_id, name, symbol, current_price = coin['id'], coin['name'], coin['symbol'].upper(), coin['current_price']
    candles = get_candle_data(coin_id, 7) # يمكنك تعديل الفترة الزمنية
    if not candles:
        return f"لا توجد بيانات كافية لـ {symbol}."

    support, resistance = calculate_pivot_points(candles)
    detected_patterns, pattern_points = identify_all_patterns(candles)
    analysis = analyze_coin(name, symbol, current_price, candles, '7d', support, resistance) # تعديل الفترة الزمنية في التحليل

    await send_telegram_message(
        analysis,
        chat_id,
        candles,
        name,
        support,
        resistance,
        current_price,
        detected_patterns,
        pattern_points
    )
    return None


async def publish_random_public(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    if not is_allowed_user(user_id):
        await query.answer("عذراً، لست مصرحاً لك باستخدام هذا الزر.")
        return
    await query.answer("جاري نشر تحليل عشوائي في القناة العامة...")
    result = await publish_random_analysis(PUBLIC_CHANNEL_ID)
    if result:
        await query.message.reply_text(result)

async def publish_random_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    if not is_allowed_user(user_id):
        await query.answer("عذراً، لست مصرحاً لك باستخدام هذا الزر.")
        return
    await query.answer("جاري نشر تحليل عشوائي في القناة الخاصة...")
    result = await publish_random_analysis(PRIVATE_CHANNEL_ID)
    if result:
        await query.message.reply_text(result)
    
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    if not is_allowed_user(user_id):
        await query.answer("عذراً، لست مصرحاً لك باستخدام هذا البوت.")
        return

    data = query.data

    if data == 'add_new_group':
        await query.message.reply_text("من فضلك أرسل أمر إضافة المجموعة الآن (/addgroup).", reply_markup=telegram.ForceReply(selective=True))
        context.user_data['waiting_for_add_group_command'] = True
        await query.answer()

    elif data == 'set_limit_group':
        await query.message.reply_text("من فضلك أرسل أمر تعيين الحد الآن (/setlimmet LIMIT).", reply_markup=telegram.ForceReply(selective=True))
        context.user_data['waiting_for_set_limit_command'] = True
        await query.answer()

    elif data == 'private_analyze':
        await private_analyze_coin_callback(update, context)
        await query.answer()

    elif data == 'publish_random_public':
        await publish_random_public(update, context)

    elif data == 'publish_random_private':
        await publish_random_private(update, context)

    elif data == 'add_trade_photo':
        await query.message.reply_text("من فضلك أرسل الصورة الآن ليتم حفظها للصفقات القادمة.")
        context.user_data['waiting_for_photo'] = True
        await query.answer()

    elif data == 'set_auto_trade_time':
        await query.message.reply_text("سيتم تفعيل هذه الميزة قريباً.")
        await query.answer()
# التعامل مع الرسائل النصية في لوحة التحكم
async def handle_control_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not is_allowed_user(user_id):
        return

    if context.user_data.get('waiting_for_add_group_command'):
        await add_group(update, context)
        context.user_data['waiting_for_add_group_command'] = False
        return

    if context.user_data.get('waiting_for_set_limit_command'):
        # سيتم التعامل مع أمر /setlimmet بواسطة الأمر نفسه
        return

    if context.user_data.get('waiting_for_photo'):
        await handle_photo(update, context)
        context.user_data['waiting_for_photo'] = False
        return

    if context.user_data.get('waiting_for_private_coin'):
        await handle_private_coin_input(update, context)
        return

    if context.user_data.get('waiting_for_private_frame'):
        await handle_private_frame_input(update, context)
        return

# التعامل مع رفع صورة
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global trade_photo_file_id
    if context.user_data.get('waiting_for_photo'):
        trade_photo_file_id = update.message.photo[-1].file_id
        await update.message.reply_text("تم حفظ الصورة بنجاح للصفقات القادمة.")
        context.user_data['waiting_for_photo'] = False

# دالة إرسال صفقات تلقائيًا (سيتم تعديلها لاحقاً إذا لزم الأمر)
async def auto_send_trades():
    global auto_trade_interval
    bot = telegram.Bot(token=bot_token)
    while True:
        try:
            # هذا الجزء يحتاج إلى تعديل ليراعي المجموعات وحدود التحليل
            pass
        except Exception as e:
            print(f"خطأ أثناء الإرسال التلقائي: {e}")
        if auto_trade_interval:
            await asyncio.sleep(auto_trade_interval)
        else:
            await asyncio.sleep(3600) # الانتظار لمدة ساعة إذا لم يتم تعيين الفاصل الزمني

# جلب عملة بالاسم
def get_coin_by_name(coin_name):
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=5000&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        for coin in response.json():
            if coin['id'].lower() == coin_name or coin['symbol'].lower() == coin_name:
                return coin
    return None


async def monitor_open_trades():
    bot = telegram.Bot(token=bot_token)
    while True:
        if open_trades:
            for trade in open_trades:
                try:
                    url = f"https://api.coingecko.com/api/v3/simple/price?ids={trade['coin_id']}&vs_currencies=usd"
                    response = requests.get(url)
                    if response.status_code == 200:
                        price = response.json()[trade['coin_id']]['usd']
                        for i, target in enumerate(trade['targets']):
                            if not trade['achieved'][i] and price >= target:
                                await bot.send_message(
                                    chat_id=trade['chat_id'],
                                    text=f"✅ تهانينا! تم تحقيق الهدف رقم {i+1} لعملة {trade['symbol']}.\nوصل السعر إلى ${price:.6f}",
                                    reply_to_message_id=trade['message_id'],
                                    parse_mode="Markdown"
                                )
                                trade['achieved'][i] = True
                except Exception as e:
                    print(f"مشكلة في متابعة الصفقة: {e}")
        await asyncio.sleep(60)

def main():
    application = Application.builder().token(bot_token).build()

    # أوامر لوحة التحكم الخاصة
    application.add_handler(CommandHandler("start", start, filters=filters.ChatType.PRIVATE)) # إضافة فلتر للمحادثات الخاصة فقط
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_control_input))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CommandHandler("addgroup", add_group))
    application.add_handler(CommandHandler("setlimit", set_group_limit_current))
    application.add_handler(CommandHandler("checklimet", check_group_limit))
    application.add_handler(CommandHandler("xid", gpa_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_private_coin_input))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_private_frame_input))

    allowed_updates = [Update.MESSAGE, Update.CALLBACK_QUERY] 
    asyncio.run(asyncio.gather(application.run_polling(allowed_updates=allowed_updates), monitor_open_trades()))

if __name__ == "__main__":
    main()
