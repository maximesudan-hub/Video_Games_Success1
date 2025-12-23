# Video Games Success

Academic machine learning project (HEC Lausanne, Advanced Programming).

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Data
Raw dataset: `Video_Games_Sales_as_at_22_Dec_2016.csv`

Place the raw file in:
`data/raw/Video_Games_Sales_as_at_22_Dec_2016.csv`

Then build the processed dataset:

```bash
python -m src.data_loader
```

This creates:
`data/processed/games_modern.csv`

## Run
Train models and generate metrics/figures:

```bash
python main.py
```

Outputs are written to:
- `results/metrics/`
- `results/figures/`

## Reproducibility
All models use fixed `random_state` values. If you run on the same data and environment,
you should obtain identical results.
