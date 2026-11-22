.PHONY: setup download prepare train report explain app all clean

PY := venv/bin/python

setup:
	python3 -m venv venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) -m src.cli download --tickers AAPL MSFT GOOGL AMZN META TSLA NVDA SPY

prepare:
	$(PY) -m src.cli prepare --tickers AAPL MSFT GOOGL AMZN META TSLA NVDA SPY --market

train:
	$(PY) -m src.cli train --ticker AAPL --model xgboost --with-market --tune

report:
	$(PY) -m src.cli report --ticker AAPL --with-market

explain:
	$(PY) -m src.cli explain --ticker AAPL --model xgboost --with-market

app:
	$(PY) -m streamlit run app/streamlit_app.py

all:
	$(PY) -m src.cli all --tickers AAPL MSFT TSLA SPY --models xgboost --market

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
