# Dockerfile
FROM apache/airflow:2.8.4-python3.10 
COPY --from=parent requirements.txt ./tmp/requirements.txt
RUN pip install -U pip
RUN pip install -r ./tmp/requirements.txt