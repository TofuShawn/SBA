FROM python:3.12-slim

WORKDIR /app

# CPU-only torch to keep the image small; neural models are tiny.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir nicegui numpy

COPY SBA.py alphazero.py run.bat ./

EXPOSE 8080

CMD ["python", "SBA.py", "--host", "0.0.0.0", "--port", "8080"]
