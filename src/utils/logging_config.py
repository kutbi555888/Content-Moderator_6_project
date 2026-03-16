import logging
import os


def setup_logger():

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("ML_Project")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # project log
    project_handler = logging.FileHandler("logs/project.log", encoding="utf-8")
    project_handler.setLevel(logging.INFO)
    project_handler.setFormatter(formatter)

    # error log
    error_handler = logging.FileHandler("logs/errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.addHandler(project_handler)
    logger.addHandler(error_handler)

    return logger







# import logging
# import os


# def get_logger():

#     os.makedirs("logs", exist_ok=True)

#     logger = logging.getLogger("ml_project")

#     logger.setLevel(logging.INFO)

#     formatter = logging.Formatter(
#         "%(asctime)s | %(levelname)s | %(message)s"
#     )

#     file_handler = logging.FileHandler("logs/project.log", encoding="utf-8")

#     file_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)

#     return logger










# 3. logger ni chaqirish

# Istalgan notebook yoki scriptda:

# from src.utils.logging_config import setup_logger

# logger = setup_logger()
# 4. Pipeline boshlanish logi
# logger.info("🚀 Moderatsiya ML loyihasi boshlandi")

# log filega yoziladi:

# 2026-03-15 10:12:21 | INFO | 🚀 Moderatsiya ML loyihasi boshlandi
# 5. Data extraction bosqichi

# Scraper boshlanganda:

# logger.info("📥 Wikipedia dan ma'lumotlarni yig'ish boshlandi")

# tugaganda:

# logger.info("✅ Wikipedia ma'lumotlarini yig'ish tugadi")

# agar xato bo‘lsa:

# try:
#     # scraping code
# except Exception as e:
#     logger.error(f"❌ Data extraction jarayonida xatolik: {str(e)}")
# 6. Preprocessing bosqichi
# logger.info("🧹 Baseline preprocessing boshlandi")

# # preprocessing code

# logger.info("✅ Baseline preprocessing tugadi")
# 7. Feature engineering bosqichi
# logger.info("⚙️ Feature engineering boshlandi")

# # TF-IDF

# logger.info("✅ Feature engineering tugadi")