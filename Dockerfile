FROM python:3.12-slim

# Updating packages and installing cron
RUN apt-get update -y
RUN apt install wget -y

WORKDIR /notifications

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u" ,"./main.py"]