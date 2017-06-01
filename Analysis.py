import pyspark
from pyspark.sql import SQLContext

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

# 讀 BlocPower_T.csv
df2 = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/BlocPower_T.csv')
# print(df2.take(5))

# 讀 HDD-Features.csv
dfHDD = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/HDD-Features.csv')
# print(dfHDD.take(5))

# 讀 CDD-HDD-Features.csv
dfCH = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/CDD-HDD-Features.csv')
# print(dfCH.take(5))

import numpy as np
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

