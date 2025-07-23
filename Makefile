# Makefile for Churn & CLV MLOps Project

.PHONY: help train drift-report start-server test-predict clean lint docker-build docker-run docker-test flow-train

# Help message
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  train           Train the model (scripts/train.py)"
	@echo "  drift-report    Generate Evidently drift report"
	@echo "  start-server    Run the Flask API (locally)"
	@echo "  test-predict    Send test input to API via curl"
	@echo "  docker-build    Build Docker image for the API"
	@echo "  docker-run      Run the API Docker container"
	@echo "  docker-test     Run container and test prediction"
	@echo "  flow-train      Run model training flow via Prefect"
	@echo "  clean           Remove Python cache and temp files"
	@echo "  lint            Format code using Ruff"
	@echo ""

# Run training script (you can replace with Prefect-based one if needed)
train:
	python scripts/train.py

# Run Prefect training flow (main training pipeline)
flow-train:
	PYTHONPATH=. python orchestration/churn_training_flow.py

# Generate data drift report with Evidently
drift-report:
	python monitoring/drift_report.py

# Run the Flask API locally
start-server:
	python web_service/predict_flask.py

# Send a test request to the running model server
test-predict:
	curl -X POST http://localhost:9696/predict \
		-H "Content-Type: application/json" \
		-d @sample_customer.json

# Build the Docker image for the Flask API
docker-build:
	docker build -t churn-api .

# Run the Flask API in a Docker container
docker-run:
	docker run -it --rm -p 9696:9696 churn-api

# Run container and test API (kill container after test)
docker-test:
	docker run -d --rm -p 9696:9696 --name churn-api-test churn-api && \
	sleep 3 && \
	curl -X POST http://localhost:9696/predict \
		-H "Content-Type: application/json" \
		-d @sample_customer.json && \
	docker stop churn-api-test

# Clean up cache and temporary files
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage

# Format code using Ruff
lint:
	ruff . --fix

test-unit:
	PYTHONPATH=. pytest tests/