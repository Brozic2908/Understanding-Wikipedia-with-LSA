# Wikipedia LSA: Tra cứu và Phân tích Ngữ nghĩa Ẩn

Dự án này sử dụng Apache Spark (PySpark) để thực hiện Phân tích Ngữ nghĩa Ẩn (LSA) trên dữ liệu Wikipedia. Sau đó, một ứng dụng Desktop (PyQt6) được dùng để tra cứu ngữ nghĩa (semantic search) và trực quan hóa mô hình.

## ✨ Tính năng chính

- **Tra cứu Ngữ nghĩa:** Tìm kiếm bài viết Wikipedia theo khái niệm (sử dụng cosine similarity) thay vì từ khóa.
- **Trực quan hóa Mô hình:**
  - Phân tích Silhouette score để tìm số cụm (K) tối ưu.
  - Hiển thị phân cụm K-Means của các thuật ngữ qua PCA 2D.
- **Phân tích Topic:** Trực quan hóa "sức mạnh" của 80 topic (Singular Values) từ SVD.
- **Pipeline Big Data:** Script PySpark (`lsa_wikipedia_v3.py`) để xử lý XML, xây dựng TF-IDF và tính toán SVD.

## 🛠️ Công nghệ sử dụng

- **Backend:** Apache Spark (PySpark), Spark MLlib, HDFS
- **Frontend (App):** PyQt6, Matplotlib
- **Data Science:** NumPy, scikit-learn (KMeans, PCA)

## 🚀 Hướng dẫn Chạy

Dự án gồm 2 phần: **Backend (Spark)** để xử lý dữ liệu và **Frontend (PyQt App)** để tương tác.

### 1. Chạy Pipeline Big Data (Backend)

1.  **Yêu cầu:** Cần có một cụm Spark, HDFS đã được thiết lập và tệp `wikidump.xml` (dữ liệu Wikipedia) đã được tải lên HDFS.
2.  **Cấu hình:** Chỉnh sửa tệp `run_lsa_wiki_debug.ps1` để trỏ đúng đến các biến môi trường (`$env:JAVA_HOME`, `$env:HADOOP_HOME`,...) và đường dẫn file input/output trên HDFS.
3.  **Thực thi:** Chạy script PowerShell để bắt đầu job Spark.
    ```powershell
    .\run_lsa_wiki_debug.ps1
    ```
    Quá trình này sẽ chạy `lsa_wikipedia_v3.py` trên cụm Spark.

### 2. Lấy Dữ liệu từ HDFS

Sau khi job Spark hoàn tất, các kết quả sẽ nằm trên HDFS.

1.  Tạo thư mục `db` trong thư mục gốc của dự án này.
2.  Sử dụng `hdfs dfs -getmerge` để tải và gộp các tệp kết quả vào thư mục `db`:
    ```bash
    # (Thay đổi đường dẫn HDFS nếu cần)
    hdfs dfs -getmerge /user/ds/lsa_out_debug/topics/part-*.json db/topics.json
    hdfs dfs -getmerge /user/ds/lsa_out_debug/term_embeddings/part-*.json db/term_embeddings.json
    hdfs dfs -getmerge /user/ds/lsa_out_debug/doc_embeddings/part-*.json db/doc_embeddings.json
    ```

### 3. Chạy Ứng dụng Desktop (Frontend)

Khi đã có 3 tệp JSON trong thư mục `db/`, bạn có thể chạy ứng dụng giao diện.

1.  **Cài đặt thư viện:**
    ```bash
    pip install PyQt6 numpy scikit-learn matplotlib
    ```
2.  **Chạy ứng dụng:**
    ```bash
    python simulation/main.py
    ```

## 📁 Cấu trúc Thư mục

```bash
root/
├── .venv/ # Môi trường ảo Python
├── db/ # Nơi chứa dữ liệu (JSON) lấy từ HDFS
│ ├── doc_embeddings.json
│ ├── term_embeddings.json
│ └── topics.json
├── models/ # Chứa scripts pipeline Big Data
│ ├── lsa_wikipedia_v3.py # Script PySpark chính
│ ├── run_lsa_wiki_debug.ps1 # Script để chạy job Spark
│ └── ...
└── simulation/ # Mã nguồn ứng dụng PyQt6
├── screen/ # Các màn hình của ứng dụng
│ ├── PlotScreen.py
│ └── SearchScreen.py
├── main.py # File chạy ứng dụng chính
└── ...
```
