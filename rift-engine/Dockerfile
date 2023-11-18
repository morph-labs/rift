FROM python:3.10.13-bookworm
WORKDIR /rift-engine
COPY . /rift-engine
RUN pip install -e .
CMD python -m rift.server.core --host 0.0.0.0 --port 7797