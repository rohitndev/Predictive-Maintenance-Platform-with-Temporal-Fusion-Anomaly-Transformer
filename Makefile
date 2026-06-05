# Predictive Maintenance Platform — developer shortcuts.
# On Windows, run the underlying commands directly or use `make` via Git Bash.

.PHONY: install install-cloud train demo serve test lint clean

install:
	python -m venv venv
	venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
	venv/Scripts/pip install -r requirements.txt

install-cloud:
	venv/Scripts/pip install -r requirements-cloud.txt

train:
	python -m scripts.train_pipeline

demo:
	python -m scripts.run_demo

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	ruff check src api data scripts tests

clean:
	rm -rf artifacts/models/* artifacts/output/* .pytest_cache .ruff_cache
