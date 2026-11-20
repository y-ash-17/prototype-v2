.PHONY: up down migrate seed eval test

up:
	docker compose up -d --build

down:
	docker compose down -v

migrate:
	alembic upgrade head

seed:
	python -m scripts.seed_and_ingest

eval:
	python -m evals.run_eval

test:
	pytest -v