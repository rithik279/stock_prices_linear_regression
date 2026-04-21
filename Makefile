.PHONY: help quickstart train report install clean test

help:
	@echo "Stock Price Linear Regression - Available commands:"
	@echo ""
	@echo "  make quickstart  - Run quickstart script (generates prediction plot in <60s)"
	@echo "  make train       - Train all models using configs/"
	@echo "  make report      - Generate full diagnostic report with all plots"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run basic tests"
	@echo "  make clean       - Remove output files"

quickstart:
	@echo "Running quickstart..."
	@mkdir -p output
	python quickstart.py

train:
	@echo "Training models..."
	@mkdir -p output
	python scripts/train.py

report:
	@echo "Generating diagnostic report..."
	@mkdir -p output/figures output/reports
	python scripts/report.py

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

clean:
	@echo "Cleaning output files..."
	rm -rf output/
	rm -rf __pycache__ logic/__pycache__ scripts/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

test:
	@echo "Running tests..."
	python -c "from logic.data_import import import_fcn; df = import_fcn(); print(f'Data import OK: {len(df)} rows')"
