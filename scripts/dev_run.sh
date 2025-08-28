#!/usr/bin/env bash
export $(grep -v '^#' .env | xargs)
uvicorn servidor:app --host 0.0.0.0 --port 8000 --reload
