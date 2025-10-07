.PHONY: setup train test run run-http clean build-image run-image lint

setup:
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install pytest

train:
	python -m src.ml --train data/demo_train.tsv --out models/sentiment.joblib

test:
	pytest -q

run:
	python -m src.service "I love pizza!"

run-http:
	ENABLE_HTTP=1 python -m src.service

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f models/*.joblib

lint:
	python -m compileall src/ tests/

build-image:
	docker build -t sentiment-simple-ml:latest .

run-image:
	docker run -p 8000:8000 -e ENABLE_HTTP=1 sentiment-simple-ml:latest
