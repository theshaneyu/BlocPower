import pyspark
from pyspark.sql import SQLContext
from BlocPower import ScatterMatrix

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

# df2.show(5)

# 資料清理的函式
def energy(v): # reformat the values to get an actual number (e.g., 117,870 kWh to 117870)
    # v = v.encode('ascii', 'ignore').split(' ')[0].replace(',','')
    v = str(v).split(' ')[0].replace(',','')
    return np.nan if(v=='' or 'None') else float(v)
def age(v): # computes the age of a buildings, given the year of construction
    # v = v.encode('ascii','ignore')
    v = str(v)
    return 2016.0-float(v) if(len(v)==4 and v!='None') else np.nan
def stories(v):
    return float(v)
def sqFeet(v): # get rid of commas 
    # v = v.encode('ascii','ignore').replace(',','')
    v = str(v).replace(',','')
    return np.nan if(v=='' or 'None') else float(v) 
def plei(v): # in the PLEI columns, missing values can be interpeted as 0 plugged equipment
    try:
        vv = float(v)
    except:
        vv = 0.0
    return vv 
# Define udf's to apply the defined function to the Spark DataFrame
udfEnergy = udf(energy, DoubleType())
udfAge = udf(age, DoubleType())
udfStories = udf(stories, DoubleType())
udfSqFeet = udf(sqFeet, DoubleType())
udfPlei = udf(plei, DoubleType())

dfN = df2.withColumn("UTSUM_Electricity_Usage", udfEnergy("UTSUM_Electricity_Usage")) \
         .withColumn("INFO_Year of Construction", udfAge("INFO_Year of Construction")) \
         .withColumn("INFO_Number of Stories", udfStories("INFO_Number of Stories")) \
         .withColumn("INFO_Total Square Feet", udfSqFeet("INFO_Total Square Feet")) \
         .withColumn("PLEI_1_Quantity", udfPlei("PLEI_1_Quantity")) \
         .withColumn("PLEI_3_Quantity", udfPlei("PLEI_3_Quantity")).cache()
dfN = dfN.withColumnRenamed("UTSUM_Electricity_Usage","energy") \
           .withColumnRenamed("INFO_Year of Construction","age") \
           .withColumnRenamed("INFO_Number of Stories","number_stories") \
           .withColumnRenamed("INFO_Total Square Feet","square_feet") \
           .withColumnRenamed("PLEI_1_Quantity","plei_1") \
           .withColumnRenamed("PLEI_3_Quantity","plei_3")


# compute average of non-missing energy and age
energy_mean = np.nanmean(np.asarray(dfN.select("energy").rdd.map(lambda r: r[0]).collect()))
age_mean = np.nanmean(np.asarray(dfN.select("age").rdd.map(lambda r: r[0]).collect()))
# fill missing values with the computed average
dfN = dfN.na.fill({"energy": energy_mean, "age": age_mean})

# define Spark DataFrame to be written to our object store
dfOut = dfN.select('energy', 'age', 'number_stories','square_feet','plei_1','plei_3')


# use the .toPandas() function to map Spark DataFrames to pandas DataFrames
dfNp = dfN.toPandas()
dfHDDp = dfHDD.toPandas()

ScatterMatrixObj = ScatterMatrix()
ScatterMatrixObj.draw(dfNp, dfHDDp)