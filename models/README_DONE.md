# sau khi hoàn tất run_lsa_wiki_debug thì  chạy cái này để lấy file kết quả từ hdfs:
# topics
hdfs dfs -getmerge /user/ds/lsa_out_debug/topics/part-*.json \
  /mnt/c/Users/LENOVO/Downloads/big_data/topics.jsonl

# term embeddings
hdfs dfs -getmerge /user/ds/lsa_out_debug/term_embeddings/part-*.json \
  /mnt/c/Users/LENOVO/Downloads/big_data/term_embeddings.jsonl

# term neighbors (ít file, merge tương tự)
hdfs dfs -getmerge /user/ds/lsa_out_debug/term_neighbors_samples/part-*.json \
  /mnt/c/Users/LENOVO/Downloads/big_data/term_neighbors_samples.jsonl