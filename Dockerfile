FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install bitsandbytes accelerate \
    && pip install -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]