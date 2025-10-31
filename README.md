# Intelligent Traffic Signal Control — MVP

This Streamlit app predicts and recommends green light duration using a combination of a Machine Learning model and a Fuzzy Logic system.

Files
- `traffic.csv` — dataset (provided)
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

Quick start (PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

Notes
- The app trains a model from `traffic.csv` when you click "Train model" in the sidebar.
- The fuzzy system is used to compute a congestion score (Low/Medium/High) from `density`, `queue`, and `wait_time`. The final recommendation adjusts the ML prediction based on that score.

Next steps / improvements
- Cache the trained model to avoid retraining each run.
- Add more visualizations and historical analysis.
- Save model to disk and provide an option to load a pre-trained model.
