#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia LSA with PySpark — pure-Python (no NumPy)

- KHÔNG dùng Python UDF; cleaning = Spark SQL regexp_replace
- Fit CountVectorizer trên sample (transform trên full)
- KHÔNG persist/cache để tránh InMemoryRelation chiếm RAM
- SVD dùng RowMatrix (mllib) như trong sách
- Ghi: topics / doc_embeddings (U*Σ) / term_embeddings (V*Σ)
- Mọi tính toán sau SVD đều dùng list Python, không NumPy

Tested with: Spark 3.4/3.5, Scala 2.12, Python 3.10, Java 11/17
"""

import argparse
from math import sqrt

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors as OldVectors
from pyspark.ml.linalg import Vector as MLVector, SparseVector as MLSparseVector, DenseVector as MLDenseVector


def to_old_vector(v: MLVector):
    """Convert MLlib new Vector to old Vector (for RowMatrix)."""
    if isinstance(v, MLSparseVector):
        return OldVectors.sparse(v.size, list(zip(v.indices.tolist(), v.values.tolist())))
    elif isinstance(v, MLDenseVector):
        return OldVectors.dense(v.toArray().tolist())
    else:
        try:
            return OldVectors.dense(list(v.toArray()))
        except Exception as e:
            raise TypeError(f"Unknown vector type: {type(v)}") from e


def get_V_ij(V_dense, i, j):
    """
    Truy cập phần tử (i,j) của DenseMatrix V (cột-major).
    V.numRows = m (số terms), V.numCols = k (concepts).
    """
    m = V_dense.numRows
    n = V_dense.numCols
    if not V_dense.isTransposed:
        return float(V_dense.values[j * m + i])
    else:
        # nếu transposed, values thuộc V^T (n x m), column-major
        return float(V_dense.values[i * n + j])


def main():
    p = argparse.ArgumentParser(
        description="Run LSA (SVD) on Wikipedia with PySpark — pure-Python")
    p.add_argument("--input", required=True,
                   help="Path to *uncompressed* Wikipedia XML (file:///... or hdfs:///...)")
    p.add_argument("--out", required=True,
                   help="Output dir (file:///... or hdfs:///...)")
    p.add_argument("--k", type=int, default=100,
                   help="SVD rank (number of concepts)")
    p.add_argument("--min_df", type=int, default=30,
                   help="Min doc frequency for CountVectorizer")
    p.add_argument("--vocab_size", type=int,
                   default=80_000, help="Max vocab size")
    p.add_argument("--sample", type=float, default=1.0,
                   help="Sample fraction of pages for the *pipeline* (0<p<=1)")
    p.add_argument("--cv_fit_sample", type=float, default=0.10,
                   help="Fraction (0–1) for fitting CountVectorizer")
    p.add_argument("--partitions", type=int, default=64,
                   help="Target partitions for heavy stages")
    p.add_argument("--packages", default="com.databricks:spark-xml_2.12:0.18.0",
                   help="spark-xml package coordinate")
    p.add_argument("--limit_pages", type=int, default=0,
                   help="Hard cap số page đọc từ XML (đặt sớm, trước shuffle)")
    args, _ = p.parse_known_args()

    spark = (
        SparkSession.builder
        .appName("Wikipedia-LSA-PySpark-PurePython")
        .config("spark.jars.packages", args.packages)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
        .config("spark.sql.files.openCostInBytes", str(8 * 1024 * 1024))
        .config("spark.sql.shuffle.partitions", str(max(64, args.partitions)))
        # ổn định python worker + traceback tốt hơn
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        # tránh in-memory column store
        .config("spark.sql.inMemoryColumnarStorage.compressed", "false")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    # ---- 1) Load Wikipedia XML với schema tường minh ----
    page_schema = T.StructType([
        T.StructField("title", T.StringType(), True),
        T.StructField("ns", T.IntegerType(), True),
        T.StructField("redirect", T.StructType([]), True),
        T.StructField("revision", T.StructType([
            T.StructField("text", T.StructType([
                T.StructField("_VALUE", T.StringType(), True)
            ]), True)
        ]), True)
    ])

    print(f"Reading XML from: {args.input}")
    df_xml = (
        spark.read.format("xml")
        .option("rowTag", "page")
        .option("inferSchema", "false")
        .schema(page_schema)
        .load(args.input)
        .select(
            F.col("title").alias("title"),
            F.col("ns").cast("int").alias("ns"),
            F.col("redirect").alias("redirect"),
            F.col("revision.text._VALUE").alias("raw_text"),
        )
    )

    # đặt LIMIT trước mọi repartition/shuffle
    if args.limit_pages and args.limit_pages > 0:
        df_xml = df_xml.limit(int(args.limit_pages))

    if 0 < args.sample < 1.0:
        df_xml = df_xml.sample(False, args.sample, seed=13)

    # ---- 2) Lọc giống sách: ns==0, bỏ redirect, bỏ disambiguation, text != null ----
    df_articles = (
        df_xml
        .filter(F.col("ns") == 0)
        .filter(F.col("redirect").isNull())
        .filter(~F.lower(F.col("title")).contains("(disambiguation)"))
        .filter(F.col("raw_text").isNotNull())
        .repartition(args.partitions)
        .select("title", "raw_text")
    )

    # ---- 3) Clean wiki markup -> plain text (regexp_replace chain) ----
    txt = F.col("raw_text")
    txt = F.regexp_replace(txt, r"(?is)<ref[^>]*?>.*?</ref>", " ")
    txt = F.regexp_replace(txt, r"(?s)\{\{.*?\}\}", " ")
    txt = F.regexp_replace(txt, r"(?s)<[^>]+>", " ")
    txt = F.regexp_replace(txt, r"(?i)\[\[(?:File|Image):[^\]]*\]\]", " ")
    txt = F.regexp_replace(txt, r"https?://\S+", " ")
    txt = F.regexp_replace(txt, r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"$2")
    txt = F.regexp_replace(txt, r"\[\[([^\]]+)\]\]", r"$1")
    txt = F.regexp_replace(txt, r"[_\[\]]", " ")
    txt = F.regexp_replace(txt, r"\s+", " ")

    df_text = (
        df_articles
        .withColumn("text", F.trim(txt))
        .select("title", "text")
        .filter(F.length("text") > 0)
    )

    # ---- 4) Tokenize + stopwords ----
    tokenizer = RegexTokenizer(
        inputCol="text", outputCol="tokens",
        pattern="[^A-Za-z]+", minTokenLength=2, toLowercase=True
    )
    df_tok = tokenizer.transform(df_text)

    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    df_filtered = remover.transform(df_tok).select("title", "filtered")

    # ---- 5) CountVectorizer: fit sample, transform full ----
    df_for_vocab = df_filtered
    if 0 < args.cv_fit_sample < 1.0:
        df_for_vocab = df_filtered.sample(False, args.cv_fit_sample, seed=13)

    cv = CountVectorizer(
        inputCol="filtered", outputCol="tf",
        vocabSize=int(args.vocab_size), minDF=int(args.min_df), binary=False
    )
    print(f"Fitting CountVectorizer on sample fraction = {args.cv_fit_sample}")
    cv_model = cv.fit(df_for_vocab)
    vocab = cv_model.vocabulary

    df_tf = cv_model.transform(df_filtered).select("title", "tf")
    print("TF built.")

    # ---- 6) TF-IDF ----
    idf = IDF(inputCol="tf", outputCol="features")
    idf_model = idf.fit(df_tf)
    df_tfidf = idf_model.transform(df_tf).select("title", "features")
    print("TF-IDF built.")

    # ---- 7) RowMatrix + SVD ----
    rdd_old = df_tfidf.rdd.map(lambda r: to_old_vector(r["features"]))
    mat = RowMatrix(rdd_old)

    k = int(args.k)
    print(f"Computing SVD, k={k} ...")
    svd = mat.computeSVD(k, computeU=True)
    U = svd.U                 # RowMatrix (nDocs x k)
    s_vec = svd.s             # DenseVector(k)
    V = svd.V                 # DenseMatrix(nTerms x k)
    s_list = [float(x) for x in s_vec]  # DenseVector is iterable

    # ---- 8) Topics: top terms per concept (không NumPy) ----
    topn = 20
    m_terms = V.numRows
    k_cols = V.numCols

    topics_rows = []
    for j in range(min(k, k_cols)):
        # lấy cột j của V
        col = [get_V_ij(V, i, j) for i in range(m_terms)]
        # top theo |weight|
        idx_sorted = sorted(range(m_terms), key=lambda i: -abs(col[i]))[:topn]
        topics_rows.append((
            int(j),
            float(s_list[j]),
            [vocab[i] for i in idx_sorted],
            [float(col[i]) for i in idx_sorted],
        ))

    topics_schema = T.StructType([
        T.StructField("topic", T.IntegerType(), False),
        T.StructField("singular_value", T.DoubleType(), False),
        T.StructField("terms", T.ArrayType(T.StringType()), False),
        T.StructField("weights", T.ArrayType(T.DoubleType()), False),
    ])
    topics_df = spark.createDataFrame(topics_rows, schema=topics_schema)

    out_base = args.out.rstrip("/")
    topics_path = f"{out_base}/topics"
    topics_df.write.mode("overwrite").json(topics_path)
    print(f"Wrote topics -> {topics_path}")

    # ---- 9) Document embeddings: U * Σ (thuần Python list) ----
    def scale_vec(vec_old):
        arr = vec_old.toArray()  # mllib DenseVector -> array('d')
        klen = len(s_list)
        return [float(arr[i]) * s_list[i] for i in range(klen)]

    US = U.rows.map(scale_vec)

    # zip index để join title
    idx_emb_schema = T.StructType([
        T.StructField("idx", T.LongType(), False),
        T.StructField("embedding", T.ArrayType(T.DoubleType()), False),
    ])
    US_z = US.zipWithIndex().map(lambda x: (int(x[1]), x[0]))
    # giảm áp lực mỗi task khi toDF
    US_df = spark.createDataFrame(US_z.repartition(32), schema=idx_emb_schema)

    df_titles_idx = (
        df_tfidf.select("title").rdd.zipWithIndex()
        .map(lambda x: (int(x[1]), x[0]["title"]))
        .toDF(["idx", "title"])
    )
    doc_embed_df = df_titles_idx.join(
        US_df, "idx").select("title", "embedding")
    doc_embed_path = f"{out_base}/doc_embeddings"
    doc_embed_df.write.mode("overwrite").json(doc_embed_path)
    print(f"Wrote doc embeddings -> {doc_embed_path}")

    # ---- 10) Term embeddings: V * Σ (không NumPy) ----
    term_rows = []
    for i in range(m_terms):
        emb = [get_V_ij(V, i, j) * s_list[j] for j in range(k_cols)]
        term_rows.append((vocab[i], [float(x) for x in emb]))

    term_schema = T.StructType([
        T.StructField("term", T.StringType(), False),
        T.StructField("embedding", T.ArrayType(T.DoubleType()), False),
    ])
    term_embed_df = spark.createDataFrame(term_rows, schema=term_schema)
    term_embed_path = f"{out_base}/term_embeddings"
    term_embed_df.write.mode("overwrite").json(term_embed_path)
    print(f"Wrote term embeddings -> {term_embed_path}")

    # ---- 11) (Optional) vài neighbors để sanity-check (thuần Python) ----
    sample_terms = ["science", "robot", "economics", "football", "quantum"]
    term_lookup = {t: i for i, t in enumerate(vocab)}
    chosen = [t for t in sample_terms if t in term_lookup]

    if chosen:
        # tiền xử lý: norm của từng vector term (tránh lặp)
        def l2_norm(vec):
            s = 0.0
            for v in vec:
                s += v * v
            return sqrt(s) if s > 0 else 1.0

        # xây nhanh một bộ vec (giới hạn 5000 term đầu)
        max_check = min(5000, len(vocab))
        precomputed = []
        for i in range(max_check):
            v_i = [get_V_ij(V, i, j) * s_list[j] for j in range(k_cols)]
            n_i = l2_norm(v_i)
            precomputed.append((vocab[i], v_i, n_i))

        def cos_sim(a, b, nb):
            # nb: norm(b); norm(a) tính sẵn bên ngoài
            na = l2_norm(a)
            dot = 0.0
            for x, y in zip(a, b):
                dot += x * y
            return dot / (na * nb + 1e-12)

        sim_rows = []
        for t in chosen:
            ti = term_lookup[t]
            tgt = [get_V_ij(V, ti, j) * s_list[j] for j in range(k_cols)]
            nt = l2_norm(tgt)
            scores = []
            for label, vec, nv in precomputed:
                scores.append((label, float(cos_sim(tgt, vec, nv))))
            scores.sort(key=lambda x: -x[1])
            sim_rows.append((t, scores[:15]))

        sim_schema = T.StructType([
            T.StructField("term", T.StringType(), False),
            T.StructField("neighbors", T.ArrayType(T.StructType([
                T.StructField("_1", T.StringType(), False),
                T.StructField("_2", T.DoubleType(), False),
            ])), True)
        ])
        sim_df = spark.createDataFrame(sim_rows, schema=sim_schema)
        sim_out = f"{out_base}/term_neighbors_samples"
        sim_df.write.mode("overwrite").json(sim_out)
        print(f"Wrote term neighbor samples -> {sim_out}")

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
