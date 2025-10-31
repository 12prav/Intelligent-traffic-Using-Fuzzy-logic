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

## Deploying to Heroku (via GitHub Actions)

This repository includes a `Procfile` and a GitHub Actions workflow that will deploy to Heroku automatically when you push to `main`.

What I added:
- `Procfile` — runs the Streamlit app on the Heroku-assigned `$PORT`.
- `.github/workflows/deploy-heroku.yml` — Action that installs dependencies and deploys using the Heroku deploy action.
- `runtime.txt` — pins Python version for Heroku.

Required repository secrets (add these in GitHub: Settings → Secrets & variables → Actions):

- `HEROKU_API_KEY` — your Heroku account API key (Settings → Account on Heroku).
- `HEROKU_APP_NAME` — the name of the Heroku app you create (e.g. `my-intelligent-traffic`).
- `HEROKU_EMAIL` — email associated with your Heroku account.

Create a Heroku app (option A: Heroku CLI; option B: dashboard):

Option A — Heroku CLI (PowerShell):

```powershell
heroku login
heroku create your-app-name
``` 

Option B — Heroku Dashboard:
- Go to https://dashboard.heroku.com → New → Create new app → pick a name.

After creating the app and adding the three repository secrets above, push to `main` (or re-run the workflow) and GitHub Actions will deploy the app.

Troubleshooting
- If the Action fails, open the Actions tab in GitHub and inspect the logs; common issues:
	- Missing secrets — ensure `HEROKU_API_KEY`, `HEROKU_APP_NAME`, `HEROKU_EMAIL` are set.
	- Build errors — make sure all required packages are listed in `requirements.txt`.
	- Streamlit memory/timeouts — Heroku free/low-tier dynos may sleep or be limited; consider upgrading or moving to Render for persistent performance.

Helpful commands

```powershell
# Re-run the deployment (after adding secrets)
git commit --allow-empty -m "trigger: redeploy"; git push
```

If you'd like, I can also add a short section that shows how to set config vars on Heroku (for example to upload a model or store user-uploaded files), or convert this to a small Flask app served by Gunicorn if you prefer a more traditional WSGI setup.
