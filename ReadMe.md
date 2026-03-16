# 🧠 Kontent Moderatsiyasi Uchun Matn Klassifikatsiyasi Loyihasi

## 📌 1. Loyiha haqida umumiy ma'lumot

Ushbu loyiha matnli kontentni avtomatik moderatsiya qilish uchun ishlab chiqilgan **Machine Learning tizimi** hisoblanadi.

Loyiha matnni tahlil qilib, uni turli **qoidabuzarlik kategoriyalariga** ajratadi.

Bunday tizimlar:

🌐 ijtimoiy tarmoqlar  
💬 forumlar  
📰 media platformalar  

uchun zararli kontentni avtomatik aniqlashda ishlatiladi.

---

## 🎯 2. Loyiha maqsadi

Loyihaning asosiy maqsadi:

- 🤖 matnli kontentni avtomatik klassifikatsiya qilish
- 🚫 zararli kontentni aniqlash
- 🧠 Machine Learning asosida moderatsiya tizimi yaratish
- 📊 turli modellarning natijalarini solishtirish
- ⚙️ preprocessing va feature engineering orqali modelni yaxshilash

---

## 🏷 3. Klassifikatsiya kategoriyalari

Dataset quyidagi **10 ta klassdan** iborat:

| Kategoriya | Tavsif |
|------|------|
| 📢 spam | reklama yoki takroriy kontent |
| 😡 harassment | haqoratli yoki tajovuzkor matn |
| ☠️ hate | nafrat nutqi |
| 🔪 violence | zo‘ravonlik bilan bog‘liq matn |
| 🔞 sexual | jinsiy kontent |
| 🩸 self_harm | o‘ziga zarar yetkazish bilan bog‘liq matn |
| 🚫 illegal_goods | noqonuniy mahsulotlar |
| 📄 copyright | mualliflik huquqi buzilishi |
| ❗ misinformation | noto‘g‘ri yoki yolg‘on ma'lumot |
| ✅ safe | oddiy xavfsiz kontent |

---

## 📊 4. Dataset haqida ma'lumot

Dataset taxminan:

```
35 000 ta matn
10 ta klass
```

Har bir klass taxminan:

```
3500 ta sample
```

Dataset formati:

| text | label |
|------|------|
| matn | kategoriya |

---

## 🔄 5. Loyiha pipeline tuzilishi

```
📥 Data Extraction
        ↓
🔗 Data Merge
        ↓
📂 Data Load
        ↓
📊 Baseline EDA
        ↓
🧹 Baseline Preprocessing
        ↓
🧠 Feature Extraction (TF-IDF)
        ↓
🤖 Baseline Model Training
        ↓
🔍 Advanced EDA
        ↓
⚙️ Improvement Preprocessing
        ↓
🧩 Feature Engineering
        ↓
🎯 Feature Selection
        ↓
🚀 Model Improvement
```

---

## 📁 6. Loyiha strukturasi

```
project
│
├── 📂 Data
│   ├── Raw_Data
│   ├── Processed_Data
│   └── Engineered_Data
│
├── 🤖 models
│
├── 📜 logs
│
├── 📓 notebooks
│   ├── extract_data.ipynb
│   ├── merge_data.ipynb
│   ├── load_data.ipynb
│   ├── baseline_eda.ipynb
│   ├── baseline_preprocess.ipynb
│   ├── feature_extraction.ipynb
│   ├── baseline_model_train.ipynb
│   ├── advanced_eda.ipynb
│   ├── improvement_preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   └── feature_selection.ipynb
│
├── ⚙️ src
├── 🧾 scripts
└── README.md
```

---

## 🧹 7. Preprocessing

Dataset ustida quyidagi preprocessing ishlari bajarildi:

- 🔡 matnni kichik harflarga o'tkazish
- 🌐 URL larni olib tashlash
- 🧾 HTML teglarni olib tashlash
- ❌ maxsus belgilarni olib tashlash
- 🛑 stopwords olib tashlash
- 🔁 duplicate matnlarni olib tashlash
- ✂️ juda qisqa matnlarni olib tashlash

---

## 🧠 8. Feature Extraction

Matnlar **TF-IDF (Term Frequency – Inverse Document Frequency)** orqali sonli vektorlarga aylantirildi.

Parametrlar:

```
max_features = 15000
min_df = 5
max_df = 0.9
ngram_range = (1,2)
```

---

## 🧩 9. Feature Engineering

Qo‘shimcha statistik featurelar yaratildi:

- 📏 text_length
- 🔤 word_count
- 🧠 unique_words
- 🔠 uppercase_ratio
- 🔢 digit_count

Bu featurelar **TF-IDF bilan birlashtirildi**.

---

## 🎯 10. Feature Selection

Feature selection **Chi-Square (χ²)** usuli orqali bajarildi.

Maqsad:

- 📉 keraksiz featurelarni olib tashlash
- 📦 dimensionni kamaytirish
- ⚡ model ishlashini yaxshilash

Tanlangan featurelar:

```
10000 ta eng muhim feature
```

---

## 🤖 11. Baseline modellar

Quyidagi Machine Learning modellar ishlatildi:

| Model | Tavsif |
|------|------|
| 📊 Naive Bayes | ehtimollik asosidagi model |
| 📈 Logistic Regression | chiziqli klassifikator |
| 🌳 Random Forest | ansambl daraxt modeli |
| 🧠 Linear SVM | maksimal margin klassifikatori |

---

## 📊 12. Model natijalari

| Model | Accuracy |
|------|------|
| Naive Bayes | 0.84 |
| Logistic Regression | 0.88 |
| Random Forest | 0.82 |
| 🏆 Linear SVM | **0.89** |

Eng yaxshi natijani **Linear SVM modeli** ko‘rsatdi.

---

## 🛠 13. Ishlatilgan texnologiyalar

Loyihada quyidagi texnologiyalar ishlatildi:

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn
- 📉 Matplotlib
- 📈 Seaborn
- 💾 Joblib

---

## 📌 14. Xulosa

Ushbu loyiha Machine Learning yordamida **avtomatik kontent moderatsiya tizimi** yaratish mumkinligini ko‘rsatadi.

To‘g‘ri preprocessing, feature engineering va kuchli klassifikatsiya modellar orqali zararli kontentni samarali aniqlash mumkin.

Bu loyiha real moderatsiya tizimlari uchun **mustahkam asos** bo‘lib xizmat qiladi.