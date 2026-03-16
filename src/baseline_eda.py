import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger()


def class_distribution(df):

    logger.info("EDA class distribution boshlandi")

    counts = df["label"].value_counts()

    plt.figure(figsize=(10,6))

    sns.barplot(x=counts.index, y=counts.values)

    plt.xticks(rotation=45)

    plt.title("Class Distribution")

    plt.show()

    logger.info("EDA tugadi")