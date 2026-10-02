# ── Biến ─────────────────────────────────────────────────────────────────────
PYTHON   ?= python
PYTEST   ?= pytest
RUFF     ?= ruff
ENTITY   ?= books
MODE     ?= full
BATCH    ?= 500
CHECKPOINT_DIR ?= checkpoints

.PHONY: install lint test fetch-full fetch-incremental fetch-others eda spark-books spark-authors spark-dimensions spark-all help

help:
	@echo "Targets:"
	@echo "  install            pip install -e .[ingestion,dev]"
	@echo "  lint               ruff check src/ tests/"
	@echo "  test               pytest"
	@echo "  fetch-full         make fetch-full ENTITY=books"
	@echo "  fetch-incremental  make fetch-incremental ENTITY=books"
	@echo "  eda                chạy notebooks/eda_books_spark.py"

install:
	$(PYTHON) -m pip install -e ".[ingestion,dev]"

lint:
	$(RUFF) check src/ tests/

test:
	$(PYTEST) -v

fetch-full:
	PYTHONPATH=src $(PYTHON) -m ingestion fetch \
		--entity $(ENTITY) --mode full \
		--batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)

fetch-incremental:
	PYTHONPATH=src $(PYTHON) -m ingestion fetch \
		--entity $(ENTITY) --mode incremental \
		--batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)

fetch-others:
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity authors     --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity series      --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity tags        --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity publishers  --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity countries   --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity languages   --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)
	PYTHONPATH=src $(PYTHON) -m ingestion fetch --entity platforms   --mode full --batch-size $(BATCH) --checkpoint-dir $(CHECKPOINT_DIR)

eda:
	PYTHONPATH=src $(PYTHON) notebooks/eda_books_spark.py

spark-books:
	PYTHONPATH=src $(PYTHON) -m transform.bronze_to_silver.run_books

spark-authors:
	PYTHONPATH=src $(PYTHON) -m transform.bronze_to_silver.run_authors

spark-dimensions:
	PYTHONPATH=src $(PYTHON) -m transform.bronze_to_silver.run_dimensions

spark-all: spark-books spark-authors spark-dimensions

DBT_DIR = src/transform/silver_to_gold

dbt-run:
	cd $(DBT_DIR) && dbt run --profiles-dir .

dbt-test:
	cd $(DBT_DIR) && dbt test --profiles-dir .

