.PHONY: dev-backend dev-frontend test setup

setup:
	./start.sh --help

dev-backend:
	source .venv/bin/activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

test:
	source .venv/bin/activate && python -m pytest tests/ -v

clean:
	rm -rf .venv node_modules frontend/node_modules frontend/dist sessions/ __pycache__
