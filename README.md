# AI + Robots Growth Model UI (Streamlit)

This repo wraps a one-sector version of your model in a simple web UI (Streamlit):

- Calibrate parameters in a sidebar
- Click **Run simulation**
- App displays the plots (wages, rentals, K, Y, decompositions, μ paths, etc.)
- Download results as a CSV

## Quickstart (local)

```bash
git clone <YOUR_REPO_URL>
cd ai-growth-ui

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on GitHub with Streamlit Community Cloud (easiest)

1. Push this repo to GitHub.
2. In Streamlit Community Cloud:
   - **New app**
   - Pick your repo + branch
   - Main file: `app.py`
   - Deploy

## File layout

```
ai-growth-ui/
  app.py
  src/
    model.py
    plots.py
  requirements.txt
  runtime.txt
  .streamlit/
    config.toml
  .gitignore
```

## Notes / troubleshooting

- If the solver fails or plots look odd, start by changing the **solver warm start** guesses (`r guess`, `w_c guess`, `w_p guess`).
- Avoid setting CES parameters exactly equal to 1 (log-CES limit). The code handles the limit, but the Newton solver can still struggle.
- The app enforces `K0 = 1` (fixed).
