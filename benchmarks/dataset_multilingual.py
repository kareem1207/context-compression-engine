"""
Small hand-written multilingual conversations (Spanish, Hindi, Chinese, English) with four planted facts each.

Every conversation is a shuffle of short filler dialogues about unrelated topics, with the facts stated in
passing as user messages. The questions are paraphrases in the same language, so they need an embedder that
understands the language (matching shared words is not enough). A fact counts as found when its key
appears in the context.

Used by benchmarks/retrieval_multilingual.py and tests/test_multilingual.py.
"""

from __future__ import annotations

import random

# Each language: filler dialogues (user, assistant) x 2, facts (statement, paraphrased question, key).
LANGS: dict[str, dict] = {
    "es": {
        "filler": [
            ("¿Cómo preparo una paella para cuatro personas?", "Sofríe pollo y verduras, añade el arroz y el caldo caliente, y cocina sin remover unos veinte minutos."),
            ("¿Y qué puedo servir de postre?", "Una crema catalana o fruta fresca con un poco de miel funcionan muy bien después de la paella."),
            ("Quiero viajar a Japón en primavera, ¿qué ciudades visito?", "Tokio y Kioto son imprescindibles, y en abril puedes ver los cerezos en flor en los parques."),
            ("¿Conviene comprar el pase de tren para turistas?", "Si vas a moverte entre varias ciudades, el pase semanal suele compensar frente a comprar billetes sueltos."),
            ("¿Cómo agrego un elemento a una lista en Python?", "Usa el método append: lista.append(valor) añade el valor al final de la lista."),
            ("¿Y cómo recorro la lista con un bucle?", "Escribe for elemento in lista: y dentro del bloque trabaja con cada elemento por separado."),
            ("Me cuesta dormir por las noches, ¿qué hago?", "Acuéstate a la misma hora, evita las pantallas una hora antes y mantén el dormitorio fresco y oscuro."),
            ("¿Ayuda tomar café por la tarde?", "Mejor evitarlo después de las tres, porque la cafeína puede tardar muchas horas en desaparecer del cuerpo."),
            ("¿Cuánto debería ahorrar cada mes?", "Una regla habitual es destinar el veinte por ciento de tus ingresos al ahorro antes de gastar en otras cosas."),
            ("¿Es mejor un fondo indexado o un depósito?", "El fondo indexado rinde más a largo plazo, mientras que el depósito ofrece seguridad para el dinero que necesitarás pronto."),
            ("¿Cómo se juega el fuera de juego en el fútbol?", "Un jugador está en fuera de juego si, al recibir el pase, tiene menos rivales que la pelota entre él y la portería."),
            ("¿Cuántos jugadores tiene cada equipo?", "Cada equipo juega con once jugadores en el campo, incluido el portero, y puede hacer cinco cambios."),
        ],
        "facts": [
            ("Por cierto, la boda de mi hermana es el próximo mes en Lisboa.", "¿En qué ciudad se casa mi hermana?", "lisboa"),
            ("Para que lo sepas, nuestra ventana de despliegue a producción es el martes a las tres de la madrugada.", "¿Qué día podemos publicar versiones nuevas?", "martes"),
            ("Una nota rápida: soy alérgico a los anacardos, así que no los pongas en ninguna receta.", "¿Qué ingrediente debo evitar por mi alergia?", "anacardo"),
            ("Recuerda que el nombre en clave de la nueva aplicación es Halcón.", "¿Cómo llamamos dentro de la empresa a la aplicación que viene?", "halcón"),
        ],
    },
    "hi": {
        "filler": [
            ("चार लोगों के लिए पुलाव कैसे बनाऊँ?", "चावल को धोकर भिगो दें, फिर सब्ज़ियों और मसालों के साथ घी में भूनें और दोगुना पानी डालकर पकाएँ।"),
            ("इसके साथ क्या परोसना अच्छा रहेगा?", "रायता और पापड़ पुलाव के साथ बहुत अच्छे लगते हैं, और ऊपर से थोड़ा सलाद भी रखा जा सकता है।"),
            ("मैं बसंत में जापान घूमने जाना चाहता हूँ, कौन से शहर देखूँ?", "टोक्यो और क्योटो ज़रूर देखें, और अप्रैल में पार्कों में चेरी के फूल खिले हुए मिलते हैं।"),
            ("क्या रेल पास खरीदना ठीक रहेगा?", "अगर आप कई शहरों में जाएँगे तो साप्ताहिक पास अलग-अलग टिकट खरीदने से सस्ता पड़ता है।"),
            ("पाइथन में सूची में नया तत्व कैसे जोड़ें?", "append विधि का उपयोग करें: सूची.append(मान) मान को सूची के अंत में जोड़ देता है।"),
            ("और सूची के हर तत्व पर लूप कैसे चलाऊँ?", "for तत्व in सूची: लिखें और ब्लॉक के अंदर हर तत्व पर अलग से काम करें।"),
            ("रात को नींद नहीं आती, क्या करूँ?", "रोज़ एक ही समय पर सोएँ, सोने से एक घंटा पहले स्क्रीन बंद कर दें और कमरा ठंडा व अँधेरा रखें।"),
            ("क्या शाम को चाय पीना ठीक है?", "तीन बजे के बाद चाय या कॉफ़ी से बचना बेहतर है क्योंकि कैफ़ीन का असर कई घंटे तक रहता है।"),
            ("हर महीने कितनी बचत करनी चाहिए?", "आम नियम यह है कि खर्च करने से पहले अपनी आय का बीस प्रतिशत बचत में डाल दें।"),
            ("म्यूचुअल फंड बेहतर है या एफडी?", "लंबी अवधि में म्यूचुअल फंड ज़्यादा रिटर्न देता है, जबकि जल्द चाहिए होने वाले पैसे के लिए एफडी सुरक्षित रहती है।"),
            ("क्रिकेट में एलबीडब्ल्यू कब दिया जाता है?", "जब गेंद बल्लेबाज़ के पैड पर लगे और अंपायर को लगे कि वह सीधे स्टंप से टकराती, तब बल्लेबाज़ को आउट दिया जाता है।"),
            ("एक टीम में कितने खिलाड़ी होते हैं?", "क्रिकेट में हर टीम के ग्यारह खिलाड़ी मैदान पर उतरते हैं, जिनमें एक विकेटकीपर भी शामिल होता है।"),
        ],
        "facts": [
            ("वैसे, मेरी बहन की शादी अगले महीने लिस्बन में है।", "मेरी बहन किस शहर में शादी कर रही है?", "लिस्बन"),
            ("आपको बता दूँ कि हमारी प्रोडक्शन डिप्लॉय विंडो मंगलवार सुबह तीन बजे है।", "हम नया रिलीज़ किस दिन जारी कर सकते हैं?", "मंगलवार"),
            ("एक छोटी बात: मुझे काजू से एलर्जी है, इसलिए किसी भी रेसिपी में काजू मत डालना।", "एलर्जी की वजह से मुझे किस चीज़ से परहेज़ करना चाहिए?", "काजू"),
            ("याद रखना कि नई ऐप का आंतरिक कोडनेम बाज़ है।", "कंपनी के अंदर हम आने वाली एप्लिकेशन को क्या कहते हैं?", "बाज़"),
        ],
    },
    "zh": {
        "filler": [
            ("四个人吃的西班牙海鲜饭应该怎么做？", "先把鸡肉和蔬菜炒香，加入大米和热高汤，然后不要搅拌，小火煮二十分钟左右。"),
            ("饭后可以配什么甜点？", "焦糖布丁或者新鲜水果配一点蜂蜜都很适合，吃完海鲜饭后口感很清爽。"),
            ("我想春天去日本旅行，应该去哪些城市？", "东京和京都一定要去，四月份可以在公园里看到盛开的樱花。"),
            ("需要买铁路周游券吗？", "如果要在好几个城市之间来回移动，一周的周游券通常比分开买车票更划算。"),
            ("在 Python 里怎么往列表中添加元素？", "使用 append 方法：列表.append(值) 会把值添加到列表的末尾。"),
            ("那怎么用循环遍历这个列表？", "写 for 元素 in 列表: 然后在代码块里分别处理每一个元素就可以了。"),
            ("晚上总是睡不着，怎么办？", "每天固定时间上床，睡前一小时不要看屏幕，并让卧室保持凉爽和黑暗。"),
            ("下午喝咖啡有影响吗？", "三点以后最好不要喝，因为咖啡因可能要很多个小时才能从身体里代谢掉。"),
            ("每个月应该存多少钱？", "常见的做法是在花钱之前先把收入的百分之二十存起来。"),
            ("指数基金和定期存款哪个更好？", "长期来看指数基金收益更高，而短期内要用的钱放在定期存款里更安全。"),
            ("足球里的越位是怎么判定的？", "接球的一刻，如果球员和球门之间的对方球员比球还少，就属于越位。"),
            ("一支球队场上有几个人？", "每支球队场上有十一名球员，包括守门员，比赛中最多可以换五个人。"),
        ],
        "facts": [
            ("顺便说一下，我姐姐的婚礼下个月在里斯本举行。", "我姐姐会在哪个城市结婚？", "里斯本"),
            ("跟你说一声，我们的生产环境发布窗口是每周二凌晨三点。", "我们哪一天可以上线新版本？", "周二"),
            ("简单说一句：我对腰果过敏，所以任何食谱里都不要放腰果。", "因为过敏，我需要避开哪种食材？", "腰果"),
            ("记住，新应用的内部代号叫猎鹰。", "公司内部怎么称呼即将发布的应用？", "猎鹰"),
        ],
    },
    "en": {
        "filler": [
            ("How do I cook paella for four people?", "Fry chicken and vegetables, add the rice and hot stock, and cook without stirring for about twenty minutes."),
            ("What could I serve for dessert?", "A crema catalana or fresh fruit with a little honey works well after a heavy rice dish."),
            ("I want to visit Japan in spring, which cities should I see?", "Tokyo and Kyoto are essential, and in April the parks are full of cherry blossom."),
            ("Is a rail pass worth buying?", "If you will move between several cities, the weekly pass usually costs less than single tickets."),
            ("How do I add an item to a list in Python?", "Use the append method: items.append(value) adds the value to the end of the list."),
            ("And how do I loop over the list?", "Write for item in items: and handle each element separately inside the block."),
            ("I cannot sleep at night, what should I do?", "Go to bed at the same time, avoid screens for an hour before, and keep the bedroom cool and dark."),
            ("Does afternoon coffee matter?", "Better to avoid it after three, because caffeine can stay in the body for many hours."),
            ("How much should I save each month?", "A common rule is to put twenty percent of your income aside before spending on anything else."),
            ("Is an index fund better than a deposit?", "An index fund returns more over the long run, while a deposit is safer for money you need soon."),
            ("How does the offside rule work in football?", "A player is offside if fewer opponents than the ball are between them and the goal when the pass is played."),
            ("How many players are on each team?", "Each team fields eleven players including the goalkeeper and may make five substitutions."),
        ],
        "facts": [
            ("By the way, my sister's wedding is next month in Lisbon.", "In which city will my sibling be getting married?", "lisbon"),
            ("Just so you know, our production deploy window is Tuesday at 3 AM.", "On what day are we allowed to push releases live?", "tuesday"),
            ("Quick note: I am allergic to cashews, so keep them out of any recipe.", "Is there an ingredient I must avoid because of an allergy?", "cashew"),
            ("Remember that our internal codename for the new app is Falcon.", "What do we call the upcoming application inside the company?", "falcon"),
        ],
    },
}


def build(lang: str, seed: int) -> dict:
    """{'messages': [...], 'questions': [{'q', 'keys'}...]} for one conversation in `lang`."""
    rng = random.Random(f"{lang}-{seed}")
    spec = LANGS[lang]
    dialogues = list(spec["filler"])
    rng.shuffle(dialogues)
    # keep exchange pairs together: filler is stored as consecutive (question, answer) pairs
    messages: list[dict] = []
    for q, a in dialogues:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    ack = {"es": "Entendido.", "hi": "ठीक है, समझ गया।", "zh": "好的，记住了。", "en": "Got it."}[lang]
    slots = sorted(rng.sample(range(2, len(dialogues)), len(spec["facts"])), reverse=True)
    order = list(range(len(spec["facts"])))
    rng.shuffle(order)
    for slot, fi in zip(slots, order):
        messages[slot * 2:slot * 2] = [
            {"role": "user", "content": spec["facts"][fi][0]},
            {"role": "assistant", "content": ack},
        ]
    questions = [{"q": q, "keys": [k]} for _, q, k in spec["facts"]]
    return {"messages": messages, "questions": questions}
