# KB Server — 单文件本地知识库
# 构建: docker build -t kb-server .
FROM python:3.10-slim

# 系统依赖：PaddleOCR / PyMuPDF 运行所需
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用本体（单文件）
COPY kb_server.py /app/kb_server.py

# 数据目录（文档 + SQLite）
ENV KB_DOCS_DIR=/kb_persist/docs
ENV KB_DB_PATH=/kb_persist/kb.db
VOLUME ["/kb_persist"]

EXPOSE 8080

CMD ["python", "kb_server.py"]
