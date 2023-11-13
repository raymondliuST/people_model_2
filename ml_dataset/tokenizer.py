from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import json 
import numpy as np
from pyspark.sql.types import ArrayType, IntegerType, FloatType


def tokenize(path):
    """
        Tokenize dataframe to a vacab dict that includes indices for all tokens
    """
    # Create a Spark session
    spark = SparkSession.builder.appName("YourAppName").getOrCreate()

    df = spark.read.parquet(path).select("browserFamily", "deviceType", "os", "country").filter("deviceType != ''").na.drop()

    categorical_columns = ["browserFamily","deviceType", "os","country"]

    vocab_dict = {}
    index = 0
    for col in categorical_columns:
        for vocab in df.select(col).distinct().collect():
            vocab_dict[index] = vocab[col]
            index += 1

    
    with open('ml_dataset/tokenizer.json', 'w') as json_file:
        json.dump(vocab_dict, json_file, indent=4)

    return df, vocab_dict

def word_to_idx(vocab, vocab_dict):
    for idx, v in vocab_dict.items():
        if v == vocab:
            return int(idx)
        
    return -1

def word_to_onehot(vocab, vocab_dict):
    word_idx = word_to_idx(vocab, vocab_dict)

    onehot = np.zeros(len(vocab_dict))
    onehot[word_idx] = 1

    return onehot.tolist()

def index_dataframe(df, vocab_dict):
    """
        returns onehotted dataframe
    """

    word_to_idx_udf = udf(lambda x: word_to_idx(x, vocab_dict=vocab_dict), IntegerType())

    for col in df.columns:
        df = df.withColumn(col, word_to_idx_udf(col))

    df.write.mode("overwrite").parquet("pm-dataset-indexed.parquet")
    return df

def onehot_dataframe(df, vocab_dict):
    """
        returns onehotted dataframe
    """

    word_to_onehot_udf = udf(lambda x: word_to_onehot(x, vocab_dict=vocab_dict), ArrayType(FloatType()))

    for col in df.columns:
        df = df.withColumn(col, word_to_onehot_udf(col))

    df.write.mode("overwrite").parquet("pm-dataset-onehot.parquet")
    return df



if __name__ == "__main__":
    df, vocab_dict = tokenize("ml_dataset/pm-dataset.parquet")

    df_onehot = onehot_dataframe(df, vocab_dict)
    df_index = index_dataframe(df, vocab_dict)
