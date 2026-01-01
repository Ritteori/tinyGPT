FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir \
    torch \
    fastapi \
    uvicorn \
    pydantic \
    pandas \
    numpy

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]