FROM python:3.11-slim
WORKDIR /app/agent
COPY . /app/agent
RUN chmod +x /app/agent/run.sh && mkdir -p /app/agent/output /output
ENV PYTHONUNBUFFERED=1 DATA_ROOT=/input OUTPUT_DIR=/output
ENTRYPOINT ["/app/agent/run.sh"]
