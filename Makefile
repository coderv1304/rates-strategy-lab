# macOS / Linux / WSL shortcuts. Windows users: run the python commands directly.
.PHONY: data demo backtest test lint note

data:
	python -m rateslab.pipeline --source fred --only data

demo:
	python -m rateslab.pipeline --source synthetic

backtest:
	python -m rateslab.pipeline --source fred

test:
	pytest -q

lint:
	ruff check .

note:
	python -m rateslab.weekly_note
