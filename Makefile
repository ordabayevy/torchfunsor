.PHONY: install lint license format test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall torchfunsor

lint: FORCE
	ruff check .
	ruff format --check .

docs: FORCE
	cd docs && make html

license: FORCE
	python scripts/update_headers.py

format: license FORCE
	ruff check --fix .
	ruff format .

typecheck: FORCE
	mypy funsor tests

FORCE:
