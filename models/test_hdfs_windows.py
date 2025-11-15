from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("hdfs-write-read-test")
         .master("local[*]")
         .config("spark.hadoop.fs.defaultFS","hdfs://localhost:9000")
         .getOrCreate())

df = spark.createDataFrame([(1,'a'),(2,'b')], ['id','v'])
df.write.mode('overwrite').parquet('hdfs://localhost:9000/user/mannhi/test_parquet')

print("Files on HDFS:")
print([f for f in spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration()).listStatus(
    spark._jvm.org.apache.hadoop.fs.Path('/user/mannhi'))])

spark.read.parquet('hdfs://localhost:9000/user/mannhi/test_parquet').show()
spark.stop()