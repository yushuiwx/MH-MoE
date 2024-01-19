from .fewshot_task import HarnessBaseTask


class XNLIBase(HarnessBaseTask):
    DATASET_NAME = None
    QUESTION_WORD = None  # 'right'
    ENTAILMENT_LABEL = None  # 'Yes'
    NEUTRAL_LABEL = None  # 'Also'
    CONTRADICTION_LABEL = None  # 'No'

    def set_class_num(self):
        self.class_num = 3

    def set_dataname(self):
        self.dataname = f"xnli_{self.DATASET_NAME}"

    def preprocess_example(self, example):
        input_str = [''] * self.class_num
        ctx = example["text"]
        answer_str = [ctx.replace("[MASK]", item) for item in [self.ENTAILMENT_LABEL, self.NEUTRAL_LABEL, self.CONTRADICTION_LABEL]]
        label = example["label"]
        return input_str, answer_str, label

class XNLI_en(XNLIBase):  # English
    DATASET_NAME = "en"

    QUESTION_WORD = "right"
    ENTAILMENT_LABEL = "Yes"
    NEUTRAL_LABEL = "Also"
    CONTRADICTION_LABEL = "No"


class XNLI_de(XNLIBase):  # German
    DATASET_NAME = "de"

    QUESTION_WORD = "richtig"
    ENTAILMENT_LABEL = "Ja"
    NEUTRAL_LABEL = "Auch"
    CONTRADICTION_LABEL = "Nein"


class XNLI_ar(XNLIBase):  # Arabic
    DATASET_NAME = "ar"

    QUESTION_WORD = "صحيح"
    ENTAILMENT_LABEL = "نعم"
    NEUTRAL_LABEL = "لذا"
    CONTRADICTION_LABEL = "رقم"


class XNLI_bg(XNLIBase):  # Bulgarian
    DATASET_NAME = "bg"

    QUESTION_WORD = "правилно"
    ENTAILMENT_LABEL = "да"
    NEUTRAL_LABEL = "така"
    CONTRADICTION_LABEL = "не"


class XNLI_el(XNLIBase):  # Greek
    DATASET_NAME = "el"

    QUESTION_WORD = "σωστός"
    ENTAILMENT_LABEL = "Ναί"
    NEUTRAL_LABEL = "Έτσι"
    CONTRADICTION_LABEL = "όχι"


class XNLI_es(XNLIBase):  # Spanish
    DATASET_NAME = "es"

    QUESTION_WORD = "correcto"
    ENTAILMENT_LABEL = "Sí"
    NEUTRAL_LABEL = "Asi que"
    CONTRADICTION_LABEL = "No"


class XNLI_fr(XNLIBase):  # French
    DATASET_NAME = "fr"

    QUESTION_WORD = "correct"
    ENTAILMENT_LABEL = "Oui"
    NEUTRAL_LABEL = "Aussi"
    CONTRADICTION_LABEL = "Non"


class XNLI_hi(XNLIBase):  # Hindi
    DATASET_NAME = "hi"

    QUESTION_WORD = "सही"
    ENTAILMENT_LABEL = "हाँ"
    NEUTRAL_LABEL = "इसलिए"
    CONTRADICTION_LABEL = "नहीं"


class XNLI_ru(XNLIBase):  # Russian
    DATASET_NAME = "ru"

    QUESTION_WORD = "правильно"
    ENTAILMENT_LABEL = "Да"
    NEUTRAL_LABEL = "Так"
    CONTRADICTION_LABEL = "Нет"


class XNLI_sw(XNLIBase):  # Swahili
    DATASET_NAME = "sw"

    QUESTION_WORD = "sahihi"
    ENTAILMENT_LABEL = "Ndiyo"
    NEUTRAL_LABEL = "Hivyo"
    CONTRADICTION_LABEL = "Hapana"


class XNLI_th(XNLIBase):  # Thai
    DATASET_NAME = "th"

    QUESTION_WORD = "ถูกต้อง"
    ENTAILMENT_LABEL = "ใช่"
    NEUTRAL_LABEL = "ดังนั้น"
    CONTRADICTION_LABEL = "ไม่"


class XNLI_tr(XNLIBase):  # Turkish
    DATASET_NAME = "tr"

    QUESTION_WORD = "doğru"
    ENTAILMENT_LABEL = "Evet"
    NEUTRAL_LABEL = "Böylece"
    CONTRADICTION_LABEL = "Hayır"


class XNLI_ur(XNLIBase):  # Urdu
    DATASET_NAME = "ur"

    QUESTION_WORD = "صحیح"
    ENTAILMENT_LABEL = "جی ہاں"
    NEUTRAL_LABEL = "اس لئے"
    CONTRADICTION_LABEL = "نہیں"


class XNLI_vi(XNLIBase):  # Vietnamese
    DATASET_NAME = "vi"

    QUESTION_WORD = "đúng"
    ENTAILMENT_LABEL = "Vâng"
    NEUTRAL_LABEL = "Vì vậy"
    CONTRADICTION_LABEL = "Không"


class XNLI_zh(XNLIBase):  # Chinese
    DATASET_NAME = "zh"

    QUESTION_WORD = "正确"
    ENTAILMENT_LABEL = "是的"
    NEUTRAL_LABEL = "所以"
    CONTRADICTION_LABEL = "不是的"


LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

LANG_CLASSES = [
    XNLI_ar,
    XNLI_bg,
    XNLI_de,
    XNLI_el,
    XNLI_en,
    XNLI_es,
    XNLI_fr,
    XNLI_hi,
    XNLI_ru,
    XNLI_sw,
    XNLI_th,
    XNLI_tr,
    XNLI_ur,
    XNLI_vi,
    XNLI_zh,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xnli_{lang}"] = lang_class
    return tasks