$env:HADOOP_HOME = 'C:\hadoop'
$env:HADOOP_COMMON_HOME = 'C:\hadoop'
$env:HADOOP_CONF_DIR = 'C:\hadoop_conf'         # nơi bạn copy core-site.xml etc.
$env:PATH = $env:PATH + ';C:\hadoop\bin'
$env:HADOOP_USER_NAME = $env:USERNAME          # ép auth simple
$env:SPARK_DIST_CLASSPATH = 'C:\hadoop_jars\*'  # hoặc danh sách đầy đủ nếu cần
# (tùy chọn) ép java sử dụng simple auth
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
$env:Path = $env:JAVA_HOME + '\bin;' + $env:Path
# tạm thời cho session này
$env:HADOOP_USER_NAME = 'mannhi'
# .\run_pyspark_windows.ps1

python test_hdfs_windows.py