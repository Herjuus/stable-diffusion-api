FROM python:3.11

RUN apt-get update && apt-get install -y git && apt-get install libgl1

RUN pip install --upgrade pip

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT [ "python3" ]

CMD [ "api.py" ]