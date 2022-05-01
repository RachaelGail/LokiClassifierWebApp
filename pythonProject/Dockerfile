FROM python:3.9

# COPY . /app
# WORKDIR /app

COPY requirements.txt /requirements.txt

RUN python3.9 -m pip install -r /requirements.txt

RUN python -m nltk.downloader punkt

# ENTRYPOINT ["python"]
# CMD ["app.py"]