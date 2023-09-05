
fmt:
	poetry run isort --profile black .
	poetry run black .

lint:
	poetry run isort --profile black --check .
	poetry run black --check .
	poetry run flake8 --show-source --statistics
	poetry run mypy .

test:
	poetry run pytest -rf