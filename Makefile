# labelcheck — development and deployment entry points.
#
# Everything runs out of a virtualenv in the source tree; the application is never
# installed as a package (see the note in pyproject.toml). Override VENV to point at
# another interpreter, e.g. `make test VENV=/opt/labelcheck/venv`.

VENV    ?= .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
RUFF    := $(VENV)/bin/ruff
PYTEST  := $(VENV)/bin/pytest
PORT    ?= 8712

# Development tools are pinned here rather than in requirements.txt, because they are not
# installed on the deploy box: it gets the runtime closure and nothing else.
DEV_TOOLS := ruff==0.16.1 pytest==9.1.1

.PHONY: help install install-dev dev test test-all lint fmt bench build-fe

help:
	@echo "install     runtime dependencies, hash-verified, exactly the pinned closure"
	@echo "install-dev the two development tools (ruff, pytest) on top of install"
	@echo "dev         run the app locally with reload on http://127.0.0.1:$(PORT)"
	@echo "test        fast suite (no OCR, under 15 s)"
	@echo "test-all    fast suite plus the golden set (runs OCR, minutes)"
	@echo "lint        ruff check + format check"
	@echo "fmt         reformat in place"
	@echo "bench       measure this machine and print recommended knobs"
	@echo "build-fe    build the frontend bundle into static/dist/"

# --no-deps is required: it is what keeps opencv-python (a hard rapidocr dependency, GUI
# stack included) off the box in favour of the headless build. See requirements.txt.
install:
	$(PIP) install --no-deps --require-hashes -r requirements.txt

install-dev: install
	$(PIP) install $(DEV_TOOLS)

dev:
	$(UVICORN) app.api:app --reload --host 127.0.0.1 --port $(PORT)

test:
	$(PYTEST) -q

test-all:
	$(PYTEST) -q -m "slow or not slow"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

fmt:
	$(RUFF) format .
	$(RUFF) check --fix .

bench:
	$(PY) -m app.bench

build-fe:
	npm ci
	npm run build
