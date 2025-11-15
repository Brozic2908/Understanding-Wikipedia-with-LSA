# LSA on Wikipedia with PySpark (PySpark + spark-xml + HDFS)

## 1) Put Wikipedia dump to HDFS

Use the **uncompressed** XML for best performance with `spark-xml`:

```bash
# In WSL Ubuntu
curl -L https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 \
| bzip2 -cd > enwiki-latest-pages-articles-multistream.xml

hdfs dfs -mkdir -p /user/ds
hdfs dfs -put enwiki-latest-pages-articles-multistream.xml /user/ds/wikidump.xml
```

## 2) Run the PySpark job

**Windows native PowerShell (single machine / local mode):**
```powershell
# Make sure your env is the same Python for driver & worker (e.g. conda py310)
$env:PYSPARK_PYTHON = "C:\Users\LENOVO\miniconda3\envs\pyspark310\python.exe"
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:HADOOP_HOME = "C:\hadoop"  # winutils.exe in C:\hadoop\bin

spark-submit `
  --master local[*] `
  --conf spark.hadoop.fs.defaultFS=hdfs://localhost:9000 `
  lsa_wikipedia.py `
  --input hdfs:///user/ds/wikidump.xml `
  --out hdfs:///user/ds/lsa_out `
  --k 100 --min_df 5 --vocab_size 50000 --partitions 200
```

**WSL/Ubuntu (talks to HDFS in WSL):**
```bash
export PYSPARK_PYTHON=/home/$USER/miniconda3/envs/pyspark310/bin/python
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON

spark-submit \
  --master local[*] \
  lsa_wikipedia.py \
  --input hdfs:///user/ds/wikidump.xml \
  --out hdfs:///user/ds/lsa_out \
  --k 100 --min_df 5 --vocab_size 50000 --partitions 200
```

> The script auto-loads `spark-xml` via `spark.jars.packages` with artifact `com.databricks:spark-xml_2.12:0.18.0`.
> If your Spark/Scala version differs, adjust the artifact accordingly.

## 3) Outputs (in HDFS)

- `/topics` — JSON rows with top terms per topic (k rows)
- `/doc_embeddings` — JSON rows with `title` and `embedding` (U Σ)
- `/term_embeddings` — JSON rows with `term` and `embedding` (V Σ)
- `/term_neighbors_samples` — small demo JSON of nearest terms for a handful of seeds

## 4) Tips

- Start with a small sample to verify the pipeline: `--sample 0.01`
- If you see `Python worker exited` or `version mismatch`, ensure both env vars point to the **same** Python:
  - `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON`
- If `spark-xml` cannot read the `.bz2`, uncompress before uploading to HDFS (recommended).
- Memory: for big runs, consider `--conf spark.driver.memory=8g --conf spark.executor.memory=8g`.
- To change tokenization/stopwords, edit the tokenizer or pass a custom stopwords list into `StopWordsRemover`.
