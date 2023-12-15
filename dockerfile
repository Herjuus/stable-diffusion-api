FROM python:3.10.0a7-alpine3.13

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

ENTRYPOINT [ "python3" ]

CMD [ "api.py" ]