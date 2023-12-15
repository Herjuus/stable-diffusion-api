FROM python

RUN apt-get update && apt-get install -y git

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

ENTRYPOINT [ "python3" ]

CMD [ "api.py" ]