from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import json 
import numpy as np
from pyspark.sql.types import ArrayType, IntegerType, FloatType

spark = SparkSession.builder.appName("YourAppName").getOrCreate()

df = spark.read.parquet("ml_dataset/pm-dataset.parquet").select("browserFamily", "deviceType", "os", "country").filter("deviceType != ''").na.drop()

categorical_columns = ["browserFamily","deviceType", "os","country"]

for c in categorical_columns:
    vocab_freq = df.groupby(c).count().rdd.map(lambda x: (x[0], x[1])).collectAsMap()
    if c == "country":
        include_list = [vocab for vocab, size in vocab_freq.items() if size > 600]
    elif c == "browserFamily":
        include_list = [vocab for vocab, size in vocab_freq.items() if size > 500]
    else:
        include_list = [vocab for vocab, size in vocab_freq.items() if size > 100]
    
    df = df.filter(col(c).isin(include_list))

df.write.mode("overwrite").parquet("ml_dataset/pm-dataset-cleaned.parquet")

import pdb
pdb.set_trace()