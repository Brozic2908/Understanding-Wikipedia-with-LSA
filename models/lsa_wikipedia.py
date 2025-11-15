#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSA on Wikipedia with PySpark — "like the book" edition

This version mirrors the Chapter 6 workflow in *Advanced Analytics with Spark*:
- Read the Wikipedia XML dump (spark-xml)
- **Filter to mainspace articles (ns == 0)**
- **Drop redirects**
- **Drop disambiguation pages**
- Strip wiki markup to plain text
- Tokenize, remove stopwords (optional: lemmatize with NLTK WordNet)
- Build TF–IDF document vectors
- Compute truncated SVD (RowMatrix.computeSVD) for LSA
- Save artifacts: topics (top terms per concept), doc embeddings (U*S), term embeddings (V*S)

Tested with: Spark 3.4/3.5, Scala 2.12, Python 3.10, Java 11/17
"""

import os
import re
import sys
import json
import math
import argparse
from typing import List, Tuple

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors as OldVectors
from pyspark.ml.linalg import Vector as MLVector, SparseVector as MLSparseVector, DenseVector as MLDenseVector

# -----------------------
# Utilities
# -----------------------

def to_old_vector(v: MLVector):
    if isinstance(v, MLSparseVector):
        return OldVectors.sparse(v.size, list(zip(v.indices.tolist(), v.values.tolist())))
    elif isinstance(v, MLDenseVector):
        return OldVectors.dense(v.toArray().tolist())
    else:
        raise TypeError(f"Unknown vector type: {type(v)}")

# -----------------------
# Wiki cleaning & filters
# -----------------------

# Heuristic cleaner: remove common MediaWiki markup
REF_RE = re.compile(r"<ref.*?>.*?</ref>", flags=re.DOTALL | re.IGNORECASE)
TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", flags=re.DOTALL)      # {{ ... }}
TAG_RE = re.compile(r"<[^>]+>")                                # <tag>
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", flags=re.IGNORECASE)
EXTERNAL_URL_RE = re.compile(r"https?://\S+")
BRACKETS_RE = re.compile(r"\[\[|\]\]")
MULTISPACE_RE = re.compile(r"\s+")

# Collapse a wiki link [[target|alias]] -> alias; keep simple [[target]] -> target
def _collapse_links(txt: str) -> str:
    def repl(m):
        s = m.group(0)
        if "|" in s:
            # [[xx|alias]] -> alias (strip [[ ]])
            try:
                inner = s[2:-2]
                return inner.split("|", 1)[1]
            except Exception:
                return " "
        else:
            return s[2:-2]
    return re.sub(r"\[\[[^\]]+\]\]", repl, txt)

def clean_wiki_markup(text: str) -> str:
    if not text:
        return ""
    text = REF_RE.sub(" ", text)
    text = TEMPLATE_RE.sub(" ", text)
    text = FILE_LINK_RE.sub(" ", text)
    text = _collapse_links(text)
    text = TAG_RE.sub(" ", text)
    text = EXTERNAL_URL_RE.sub(" ", text)
    text = BRACKETS_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()

clean_udf = F.udf(clean_wiki_markup, T.StringType())

# Optional: very light lemmatizer using WordNet if available
def _maybe_setup_nltk(lemmatize: bool):
    if not lemmatize:
        return None
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        # Try to ensure wordnet is available
        try:
            from nltk.corpus import wordnet as _wn  # noqa
        except LookupError:
            nltk.download("wordnet")
        return WordNetLemmatizer()
    except Exception:
        return None

# UDF lemmatizer (noop if NLTK is not present)
def make_lemma_udf(enable: bool):
    if not enable:
        return None
    import nltk  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    lemmatizer = WordNetLemmatizer()
    @F.udf(returnType=T.ArrayType(T.StringType()))
    def lemma_udf(tokens: List[str]) -> List[str]:
        try:
            return [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            return tokens
    return lemma_udf

# -----------------------
# Main
# -----------------------

def main():
    p = argparse.ArgumentParser(description="Run LSA (SVD) on Wikipedia with PySpark")
    p.add_argument("--input", required=True, help="HDFS path to *uncompressed* Wikipedia XML (e.g., hdfs:///user/ds/wikidump.xml)")
    p.add_argument("--out", required=True, help="HDFS output dir (e.g., hdfs:///user/ds/lsa_out)")
    p.add_argument("--k", type=int, default=400, help="SVD rank (number of concepts)")
    p.add_argument("--min_df", type=int, default=5, help="Min doc frequency for CountVectorizer")
    p.add_argument("--vocab_size", type=int, default=200_000, help="Max vocab size")
    p.add_argument("--sample", type=float, default=1.0, help="Sample fraction of pages (0<p<=1)")
    p.add_argument("--partitions", type=int, default=200, help="Repartition for heavy stages")
    p.add_argument("--lemmatize", action="store_true", help="Enable simple WordNet lemmatization (requires NLTK)")
    p.add_argument("--packages", default="com.databricks:spark-xml_2.12:0.18.0", help="spark-xml package coordinate")
    args, _ = p.parse_known_args()

    spark = (
        SparkSession.builder
        .appName("Wikipedia-LSA-PySpark")
        .config("spark.jars.packages", args.packages)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(max(200, args.partitions)))
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    # 1) Load Wikipedia XML (requires uncompressed XML for best performance)
    print(f"Reading XML from: {args.input}")
    # We read raw fields we need to filter like in the book:
    # <page>
    #   <title>...</title>
    #   <ns>0</ns>            # namespace id
    #   <redirect .../>       # present if redirect
    #   <revision><text>...</text></revision>
    # </page>
    df_xml = (
        spark.read.format("xml")
        .option("rowTag", "page")
        .load(args.input)
        .select(
            F.col("title").alias("title"),
            F.col("ns").cast("int").alias("ns"),
            F.col("redirect").alias("redirect"),
            F.col("revision.text._VALUE").alias("raw_text"),
        )
    )

    if args.sample and args.sample < 1.0:
        df_xml = df_xml.sample(False, args.sample, seed=13)

    # --- style filters ---
    # 1) Namespace == 0 (articles)
    # 2) NOT redirect (redirect field is null when absent)
    # 3) Title is not a disambiguation page
    #    (book checks title contains "(disambiguation)")
    # 4) Non-empty text
    df_articles = (
        df_xml
        .filter(F.col("ns") == 0)
        .filter(F.col("redirect").isNull())
        .filter(~F.lower(F.col("title")).contains("(disambiguation)"))
        .filter(F.col("raw_text").isNotNull())
    )

    # 2. Clean wiki markup -> plain text
    df_text = (
        df_articles
        .withColumn("text", clean_udf(F.col("raw_text")))
        .select("title", "text")
        .filter(F.length("text") > 0)
        .repartition(args.partitions)
        .persist()
    )

    # Tokenize & stopwords (letters only, >= 2 chars; lowercased by tokenizer)
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="[^A-Za-z]+",
        minTokenLength=2,
        toLowercase=True,
    )
    df_tok = tokenizer.transform(df_text)

    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    df_filtered = remover.transform(df_tok).select("title", "filtered")

    # Optional lemmatization (very light vs Stanford CoreNLP in the book)
    if args.lemmatize:
        lemma_udf = make_lemma_udf(True)
        df_filtered = df_filtered.withColumn("filtered", lemma_udf(F.col("filtered")))

    # Term frequency -> CountVectorizer
    cv = CountVectorizer(
        inputCol="filtered",
        outputCol="tf",
        vocabSize=args.vocab_size,
        minDF=args.min_df,
        binary=False,
    )
    cv_model = cv.fit(df_filtered)
    vocab = cv_model.vocabulary
    df_tf = cv_model.transform(df_filtered).select("title", "tf")

    # TF-IDF
    idf = IDF(inputCol="tf", outputCol="features")
    idf_model = idf.fit(df_tf)
    df_tfidf = idf_model.transform(df_tf).select("title", "features").persist()

    # Convert to RowMatrix rows
    rdd_old = df_tfidf.rdd.map(lambda r: to_old_vector(r["features"]))
    mat = RowMatrix(rdd_old)

    # SVD
    k = int(args.k)
    print(f"Computing SVD, k={k} ...")
    svd = mat.computeSVD(k, computeU=True)
    U = svd.U                  # RowMatrix (nDocs x k)
    s = svd.s                  # DenseVector(k)
    V = svd.V                  # DenseMatrix(nTerms x k)

    import numpy as np
    V_np = np.array(V.toArray())           # (nTerms, k)
    s_np = np.array(s.toArray())           # (k,)

    # --- Export topics: top terms by |weight| per concept ---
    topn = 20
    rows = []
    for j in range(min(k, V_np.shape[1])):
        col = V_np[:, j]
        idx = np.argsort(-np.abs(col))[:topn]
        rows.append((int(j), float(s_np[j]), [vocab[i] for i in idx], [float(col[i]) for i in idx]))

    topics_schema = T.StructType([
        T.StructField("topic", T.IntegerType(), False),
        T.StructField("singular_value", T.DoubleType(), False),
        T.StructField("terms", T.ArrayType(T.StringType()), False),
        T.StructField("weights", T.ArrayType(T.DoubleType()), False),
    ])
    topics_df = spark.createDataFrame(rows, schema=topics_schema)

    out_base = args.out.rstrip("/")
    topics_path = f"{out_base}/topics"
    topics_df.write.mode("overwrite").json(topics_path)
    print(f"Wrote topics -> {topics_path}")

    # --- Document embeddings: U * S ---
    US = U.rows.map(lambda v: OldVectors.dense((np.array(v.toArray()) * s_np).tolist()))
    # attach titles via zipWithIndex
    df_titles = df_tfidf.select("title").rdd.zipWithIndex().map(lambda x: (x[1], x[0]["title"])).toDF(["idx", "title"])
    US_df = US.zipWithIndex().map(lambda x: (x[1], x[0].toArray().tolist())).toDF(["idx", "embedding"])
    doc_embed_df = df_titles.join(US_df, "idx").select("title", "embedding")
    doc_embed_path = f"{out_base}/doc_embeddings"
    doc_embed_df.write.mode("overwrite").json(doc_embed_path)
    print(f"Wrote doc embeddings -> {doc_embed_path}")

    # --- Term embeddings: V * S ---
    VS = (V_np * np.diag(s_np))   # (nTerms, k)
    term_rows = [(vocab[i], VS[i, :].tolist()) for i in range(len(vocab))]
    term_schema = T.StructType([
        T.StructField("term", T.StringType(), False),
        T.StructField("embedding", T.ArrayType(T.DoubleType()), False),
    ])
    term_embed_df = spark.createDataFrame(term_rows, schema=term_schema)
    term_embed_path = f"{out_base}/term_embeddings"
    term_embed_df.write.mode("overwrite").json(term_embed_path)
    print(f"Wrote term embeddings -> {term_embed_path}")

    # --- Small sanity sample: similar terms by cosine in VS ---
    def top_cosine_sim(target_vec, all_vecs, all_labels, topm=15):
        tv = np.array(target_vec)
        A = np.vstack(all_vecs)
        num = A @ tv
        denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(tv) + 1e-12)
        sims = num / denom
        order = np.argsort(-sims)[:topm]
        return [(all_labels[i], float(sims[i])) for i in order]

    sample_terms = ["science", "robot", "economics", "football", "quantum"]
    term_lookup = {t: i for i, t in enumerate(vocab)}
    chosen = [t for t in sample_terms if t in term_lookup]
    max_check = min(5000, len(vocab))
    labels = vocab[:max_check]
    vecs = [VS[i, :] for i in range(max_check)]
    samples = []
    for t in chosen:
        neigh = top_cosine_sim(VS[term_lookup[t], :], vecs, labels, topm=15)
        samples.append((t, [(w, s) for (w, s) in neigh]))

    if samples:
        sim_schema = T.StructType([
            T.StructField("term", T.StringType(), False),
            T.StructField("neighbors", T.ArrayType(T.StructType([
                T.StructField("_1", T.StringType(), False),
                T.StructField("_2", T.DoubleType(), False),
            ])), True)
        ])
        sim_df = spark.createDataFrame(samples, schema=sim_schema)
        sim_out = f"{out_base}/term_neighbors_samples"
        sim_df.write.mode("overwrite").json(sim_out)
        print(f"Wrote term neighbor samples -> {sim_out}")

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
