# 📊 BTL-UI — Ứng dụng trực quan dữ liệu bằng PyQt6 & Matplotlib

Ứng dụng giao diện người dùng (UI) viết bằng **PyQt6** để trực quan hóa dữ liệu và chạy các thuật toán học máy cơ bản như **K-Means**, **Silhouette Score**, v.v.

---

## 🧩 1. Yêu cầu hệ thống

- Python **3.12+** (Khuyến nghị: **Python 3.12.6** để tương thích tốt nhất)
- pip / uv (nếu dùng `uv`, tốc độ cài đặt sẽ nhanh hơn pip)
- Git (nếu clone repo)

---

## ⚙️ 2. Tạo môi trường ảo (Virtual Environment)

Giúp cô lập các thư viện của dự án, tránh xung đột với hệ thống.

### Trên Windows (CMD hoặc PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### Trên Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 📦 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## 📜 4. Chạy ứng dụng chính

```bash
python main.py
```