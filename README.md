# CaféReserve AI - Streamlit Frontend (Windows)

Files: `app.py` (UI), `ml_core.py` (your ML code in functions), `cafereserve_simple.py` (your original script, unchanged).

1. Open this folder in VS Code, then open a Command Prompt terminal.
2. `python -m venv .venv`
3. `.venv\Scripts\activate`
4. `python -m pip install --upgrade pip`
5. `pip install -r requirements.txt`
6. `python -m streamlit run app.py`   (use this if `streamlit` is not recognized)

The app opens at http://localhost:8501. Stop it with Ctrl+C.
Models train once when the app starts and are cached; the data is saved to `data/cafe_reservations.csv`.
