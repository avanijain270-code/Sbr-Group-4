# Noon SBR – Customer Retention Analytics Dashboard

Streamlit dashboard for the Noon SBR project at SP Jain. Main focus: **customer retention and churn**,
with sales forecasting and campaign ROI as supporting problems.

## What it covers
- **Descriptive:** retention, delivery and seller-reliability metrics, CLV
- **Diagnostic:** RFM segmentation, K-Means clustering, churn drivers, purchase timing after late deliveries
- **Predictive:** churn classification, customer-interest classification, willingness-to-pay regression, ARIMA / SARIMA forecasting, association rules (support, confidence, lift)
- **Prescriptive:** A/B test, uplift modelling, market response, per-customer action list, campaign ROI

## Files
| File | Purpose |
|---|---|
| `app.py` | The dashboard |
| `requirements.txt` | Python packages |
| `noon_sbr_data.xlsx` | Dataset (see note below) |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data note
The dataset is **simulated** for this academic project. Patterns such as late deliveries leading to churn
were built in, so results demonstrate the methods, not real Noon figures.
To use real data, upload an Excel file in the sidebar with the same sheets and columns:
`customers`, `orders`, `monthly_category_sales`, `campaigns`, `baskets`.
