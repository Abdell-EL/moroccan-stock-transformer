# Transformer Stock Price Prediction: Moroccan Market

A Transformer model (PyTorch) that predicts the daily closing price of **10 companies listed on the Casablanca Stock Exchange**, with a cross-sector comparison of how well it performs on each one.

This repository contains the code, data and paper for the research project *Optimizing Investment Portfolios Using Machine Learning: A Case Study on the Moroccan Market* (Al Akhawayn University, 2025).

## What it does

For each company, the notebook:

1. Loads daily prices (Open, High, Low, Close) from January 2018 to December 2023.
2. Normalizes the prices (z-score).
3. Splits the data 70 / 15 / 15 into training, validation and test sets.
4. Trains a Transformer to predict the **Close** price from the same day's **Open, High and Low**.
5. Evaluates it with MAE, MSE, RMSE, MAPE, R² and correlation, and plots the predictions.

It then compares the results across all companies and saves a summary to [`results/summary_metrics.csv`](results/summary_metrics.csv).

### Model

| Setting | Value |
|---|---|
| Architecture | `nn.Transformer` (encoder + decoder), linear input embedding and output head |
| Hidden size | 64 |
| Encoder / decoder layers | 2 / 2 |
| Attention heads | 4 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | MSE |
| Epochs | 100 |

## Companies covered

| Company | Sector | Data file |
|---|---|---|
| Attijariwafa Bank | Banking | `attijariwafa_bank.csv` |
| BCP (Banque Centrale Populaire) | Banking | `bcp.csv` |
| Bank of Africa (BMCE) | Banking | `bmce_bank.csv` |
| Ciments du Maroc | Construction materials | `ciments_du_maroc.csv` |
| LafargeHolcim Maroc | Construction materials | `lafargeholcim_maroc.csv` |
| Itissalat Al-Maghrib (Maroc Telecom) | Telecommunications | `itissalat_al_maghrib.csv` |
| Managem | Mining | `managem.csv` |
| Société d'Exploitation des Ports (Marsa Maroc) | Port logistics | `societe_exploitation_des_ports.csv` |
| Taqa Morocco | Energy | `taqa_morocco.csv` |
| Travaux Généraux de Construction (TGCC) | Construction | `travaux_generaux_de_construction.csv` |

TGCC's history starts in December 2021, when it was listed. All other companies cover 2018–2023.

## Results

Test-set results from the notebook run saved in this repository (seed 42). Errors are on normalized prices.

| Company | MAE | RMSE | R² | Correlation |
|---|---|---|---|---|
| Attijariwafa Bank | 0.0878 | 0.1062 | 0.968 | 0.988 |
| BCP | 0.1141 | 0.1471 | 0.929 | 0.972 |
| Bmce Bank | 0.1790 | 0.2822 | 0.853 | 0.962 |
| Ciments du Maroc | 0.0540 | 0.0775 | 0.988 | 0.994 |
| Itissalat Al-Maghrib | 0.0818 | 0.0972 | 0.859 | 0.988 |
| LafargeHolcim Maroc | 0.0603 | 0.0852 | 0.987 | 0.994 |
| Managem | 0.0252 | 0.0376 | 0.995 | 0.998 |
| Société d'Exploitation des Ports | 0.1286 | 0.1451 | 0.614 | 0.988 |
| Taqa Morocco | 0.0729 | 0.0989 | 0.977 | 0.992 |
| TGCC | 0.0996 | 0.1412 | 0.907 | 0.955 |
| **Average** | **0.0903** | **0.1218** | **0.908** | **0.983** |

The notebook also reports MAPE, which is left out here: it is computed on normalized values close to zero, so it is inflated for some companies. Model training is random, so the figures in the paper come from an earlier run and differ slightly from these.

## Repository structure

```
├── data/          Daily OHLC prices for the 10 companies (CSV)
├── notebooks/     transformer_stock_prediction.ipynb, the full pipeline with saved outputs
├── paper/         Research paper (manuscript.pdf) and abstract (abstract.pdf)
├── results/       summary_metrics.csv produced by the notebook
└── requirements.txt
```

## How to run

```bash
git clone <this-repo-url>
cd moroccan-stock-transformer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/transformer_stock_prediction.ipynb
```

Then run all cells. A full run takes about 3–5 minutes on a CPU. A GPU isn't needed.

**Google Colab:** upload the notebook and the `data/` folder (keeping the folder name), then run all cells.

## Limitations and next steps

- **Same-day inputs.** The model predicts a day's close from that same day's open, high and low, one day at a time. A natural next step is to feed it a window of past days (for example the last 30) so it forecasts the next close and the attention layers work across time.
- **Split order.** The CSV files are sorted newest first and the split follows file order, so the model trains on 2019–2023 and is tested on 2018. Sorting by date before splitting would give a standard forward-in-time evaluation.
- **Normalization.** Mean and standard deviation are computed on the full dataset before splitting. Computing them on the training set only would avoid leaking information from the test period.

## Authors

- Abdellah Elouazzani
- Najlae Acheache
- Yousra Chtouki

School of Science and Engineering, Al Akhawayn University, Ifrane, Morocco.
