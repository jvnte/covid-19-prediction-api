# Makefile for setting up the environment
# and starting both applications

setup:
	pip install -r requirements.txt

run:
	uvicorn api:api --reload &
	streamlit run dashboard.py