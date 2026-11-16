"""Launch the ethylene instrument app: `python app.py` then open http://127.0.0.1:8000."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("ethylene.api:app", host="127.0.0.1", port=8000)
