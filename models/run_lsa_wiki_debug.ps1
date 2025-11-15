# --- ENV cơ bản ---
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

$env:JAVA_HOME            = 'C:\Program Files\Java\jdk-17'
$env:HADOOP_HOME          = 'C:\hadoop'
$env:HADOOP_COMMON_HOME   = $env:HADOOP_HOME
$env:HADOOP_CONF_DIR      = 'C:\hadoop_conf'
$env:HADOOP_USER_NAME     = 'mannhi'
$env:SPARK_DIST_CLASSPATH = 'C:\hadoop_jars\*'          # nếu bạn đã gom JARs Hadoop

$env:SPARK_HOME            = Join-Path $env:CONDA_PREFIX 'Lib\site-packages\pyspark'
$env:PYSPARK_PYTHON        = (Join-Path $env:CONDA_PREFIX 'python.exe')
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:JAVA_TOOL_OPTIONS     = '-Djava.net.preferIPv4Stack=true'  # tùy chọn, thường an toàn

$env:Path = @(
  (Join-Path $env:SPARK_HOME 'bin'),
  (Join-Path $env:JAVA_HOME   'bin'),
  (Join-Path $env:HADOOP_HOME 'bin'),
  $env:Path
) -join ';'

$INPUT = 'hdfs:///user/ds/wikidump.xml'
$OUT   = 'hdfs:///user/ds/lsa_out_debug'

# Kiểm tra spark-submit đúng của pyspark-pip
& (Join-Path $env:SPARK_HOME 'bin\spark-submit.cmd') --version

# Chạy: KHÔNG ép host/port; để Spark tự chọn
& (Join-Path $env:SPARK_HOME 'bin\spark-submit.cmd') `
  --master local[4] `
  --driver-memory 12g `
  --conf spark.ui.showConsoleProgress=true `
  --conf spark.python.worker.reuse=true  `
  --conf spark.python.worker.faulthandler.enabled=true `
  --conf spark.sql.execution.pyspark.udf.faulthandler.enabled=true `
  --conf spark.default.parallelism=16 `
  --conf spark.sql.shuffle.partitions=32 `
  --conf spark.python.worker.connect.timeout=300 `
  .\lsa_wikipedia_v3.py `
  --input $INPUT `
  --out   $OUT `
  --sample 1.0 `
  --cv_fit_sample 0.3 `
  --vocab_size 15000 `
  --min_df 40 `
  --k 80 `
  --partitions 32 `
  --limit_pages 200000