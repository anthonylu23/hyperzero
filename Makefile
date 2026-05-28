.PHONY: demo-api demo-web test-demo build-web

demo-api:
	python3 -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000

demo-web:
	npm run dev --prefix apps/web -- --port 5173

test-demo:
	python3 -m pytest tests/test_demo_api.py -q

build-web:
	npm run build --prefix apps/web
