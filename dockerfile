FROM python:3.13.0a2-alpine3.19


WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

ENTRYPOINT [ "python3" ]

CMD [ "api.py" ]