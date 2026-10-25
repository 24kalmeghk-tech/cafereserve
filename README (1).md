# Cafe Insight

**Smart Reservations. Happier Customers. Better Decisions.**

A Streamlit machine-learning mini-project for exploring café reservation patterns, estimating party size, predicting no-show risk, segmenting customers, and reviewing model performance.

## Files
- `app.py` — Streamlit interface, dashboard, and actionable next-step ideas.
- `ml_core.py` — existing ML pipeline and models.
- `requirements.txt` — Python dependencies.

## Run locally on Windows
1. Open this folder in VS Code and open a terminal in the project folder.
2. Create and activate a virtual environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install packages:
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Start the app:
   ```cmd
   python -m streamlit run app.py
   ```
5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud
Push `app.py`, `ml_core.py`, `requirements.txt`, and this README to the root of a GitHub repository. Create an app from that repository and select `app.py` as the main file.

## Important limitation
The current pipeline generates **synthetic demo data** in `ml_core.py`. Charts, model scores, and suggested actions are for learning and demonstration, not verified real-café findings. Replace synthetic data with appropriate real booking data and validate model performance before operational use. The app does not automatically cancel bookings or message customers.


### Weather input
The Reservation Prediction page includes Sunny, Cloudy, Rainy, and Windy options. Weather is one-hot encoded for the regression models; it is a categorical input, not a live weather forecast. The generated dataset is synthetic, so its prediction quality is illustrative only.
