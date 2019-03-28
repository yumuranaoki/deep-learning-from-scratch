FROM python:3.6
RUN mkdir /develop
WORKDIR /develop
COPY requirements.txt /develop
RUN pip install -r requirements.txt