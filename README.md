# Online Retail II – Sales Trends & RFM Customer Segmentation

YBS405 Big Data Management project (Piri Reis University, MIS Department).

This project analyses about 1.07 million transactions of a UK online gift-ware retailer (2009–2011). It covers:

- data cleaning
- revenue trends and seasonality
- product, country and order-timing analysis
- cancellations
- **RFM customer segmentation**
- **cohort retention**

## Key findings

| Finding | Value |
|---|---|
| Clean revenue analysed | £19.6M over 39,516 orders |
| Share of revenue in Sep–Nov | 37% (peak: November 2011) |
| Revenue from the United Kingdom | 86% |
| Revenue from the top 20% of customers | 77% |
| Invoices that are cancellations | 17.3% |
| New customers who return the following month | ≈19% |

Full results are in [the notebook](notebooks/analysis.ipynb) and [the report](deliverables/Report.docx).

## Dataset

Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/502/online+retail+ii · doi:10.24432/C5CG6D

The raw file (45 MB) is not stored in this repository. Download and unzip it into `data/raw/`:

```bash
curl -L -o data/raw/online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip data/raw/online_retail_ii.zip -d data/raw/
```

## How to run

```bash
pip install -r requirements.txt
python src/clean.py          # cleaning   -> data/processed/*.parquet + cleaning_log.csv
python src/analysis.py       # analysis   -> figures/*.png + data/processed/metrics.json
```

To explore the full analysis, open `notebooks/analysis.ipynb` in Jupyter and choose *Run All*.

## Repository structure

```
data/processed/   cleaning log, metrics.json, rfm_customers.csv (small outputs)
figures/          10 charts used in the report and slides
notebooks/        analysis.ipynb – the full analysis with outputs
src/              clean.py (cleaning), analysis.py (charts & metrics)
deliverables/     Proposal.docx, Report.docx, Presentation.pptx
```

## Tools

Python 3 · pandas · NumPy · matplotlib · seaborn · Jupyter · Apache Parquet (pyarrow) · Git/GitHub
