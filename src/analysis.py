"""Analysis + figure functions for the Online Retail II project.

Each `fig_*` function saves a PNG to figures/ and returns the numbers it found,
which are collected into data/processed/metrics.json and used by the report/slides.
Run everything with:  python src/analysis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PRIMARY = "#1f5f8b"
ACCENT = "#e07b39"
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "axes.titlesize": 13})

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# The data ends on 2011-12-09, so December 2011 is incomplete and excluded from trends
LAST_FULL_MONTH = "2011-11"


def load():
    sales = pd.read_parquet(PROC / "sales.parquet")
    cancels = pd.read_parquet(PROC / "cancellations.parquet")
    return sales, cancels


def gbp(x, _=None):
    if abs(x) >= 1e6:
        return f"£{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"£{x/1e3:.0f}K"
    return f"£{x:.0f}"


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


# ---------------------------------------------------------------- overview
def overview(sales, cancels):
    known = sales[~sales["IsGuest"]]
    orders = sales.groupby("Invoice")["Revenue"].sum()
    return {
        "rows_clean": int(len(sales)),
        "date_min": str(sales["InvoiceDate"].min().date()),
        "date_max": str(sales["InvoiceDate"].max().date()),
        "total_revenue": float(sales["Revenue"].sum()),
        "n_invoices": int(sales["Invoice"].nunique()),
        "n_customers": int(known["Customer ID"].nunique()),
        "n_products": int(sales["StockCode"].nunique()),
        "n_countries": int(sales["Country"].nunique()),
        "avg_order_value": float(orders.mean()),
        "median_order_value": float(orders.median()),
        "guest_revenue_share": float(sales.loc[sales["IsGuest"], "Revenue"].sum() / sales["Revenue"].sum()),
    }


# ---------------------------------------------------------------- 1. monthly trend
def fig_monthly_trend(sales):
    m = sales[sales["Month"] <= LAST_FULL_MONTH].groupby("Month")["Revenue"].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.PeriodIndex(m.index, freq="M").to_timestamp()
    ax.plot(x, m.values, marker="o", color=PRIMARY, lw=2)
    peak = m.idxmax()
    ax.annotate(f"Peak: {peak}\n{gbp(m.max())}", xy=(pd.Period(peak).to_timestamp(), m.max()),
                xytext=(-110, -10), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(gbp))
    ax.set_title("Monthly Revenue (Dec 2009 – Nov 2011)")
    ax.set_xlabel("")
    ax.set_ylabel("Revenue")
    save(fig, "01_monthly_revenue.png")
    q4 = m[[k for k in m.index if k[-2:] in ("09", "10", "11")]].sum() / m.sum()
    return {"peak_month": peak, "peak_month_revenue": float(m.max()),
            "low_month": m.idxmin(), "low_month_revenue": float(m.min()),
            "sep_nov_share": float(q4)}


# ---------------------------------------------------------------- 2. YoY
def fig_yoy(sales):
    s = sales[(sales["InvoiceDate"] >= "2010-01-01") & (sales["Month"] <= LAST_FULL_MONTH)]
    p = s.groupby([s["InvoiceDate"].dt.year, s["InvoiceDate"].dt.month])["Revenue"].sum().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.4
    months = np.arange(1, 13)
    ax.bar(months - w / 2, p.get(2010, pd.Series(dtype=float)).reindex(months).values, w, label="2010", color="#9bb8cf")
    ax.bar(months + w / 2, p.get(2011, pd.Series(dtype=float)).reindex(months).values, w, label="2011", color=PRIMARY)
    ax.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(gbp))
    ax.set_title("Year-over-Year Monthly Revenue (Jan–Nov)")
    ax.legend()
    save(fig, "02_yoy_revenue.png")
    r10 = p[2010].loc[1:11].sum()
    r11 = p[2011].loc[1:11].sum()
    return {"rev_2010_jan_nov": float(r10), "rev_2011_jan_nov": float(r11),
            "yoy_growth": float(r11 / r10 - 1)}


# ---------------------------------------------------------------- 3. top products
def fig_top_products(sales):
    g = sales.groupby("StockCode").agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
    names = sales.dropna(subset=["Description"]).groupby("StockCode")["Description"].agg(
        lambda s: s.value_counts().index[0])
    g["Name"] = names.reindex(g.index).fillna(g.index.to_series()).str.slice(0, 34)
    top_r = g.nlargest(10, "Revenue")[::-1]
    top_q = g.nlargest(10, "Quantity")[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(top_r["Name"], top_r["Revenue"], color=PRIMARY)
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(gbp))
    axes[0].set_title("Top 10 Products by Revenue")
    axes[1].barh(top_q["Name"], top_q["Quantity"], color=ACCENT)
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    axes[1].set_title("Top 10 Products by Units Sold")
    fig.tight_layout()
    save(fig, "03_top_products.png")
    top100_share = g.nlargest(100, "Revenue")["Revenue"].sum() / g["Revenue"].sum()
    return {"top_product_revenue_name": top_r["Name"].iloc[-1].title(),
            "top_product_revenue": float(top_r["Revenue"].iloc[-1]),
            "top_product_qty_name": top_q["Name"].iloc[-1].title(),
            "top100_products_revenue_share": float(top100_share),
            "n_products": int(len(g))}


# ---------------------------------------------------------------- 4. countries
def fig_countries(sales):
    c = sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    uk_share = c.get("United Kingdom", 0) / c.sum()
    intl = c.drop("United Kingdom").head(10)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].pie([uk_share, 1 - uk_share], labels=["United Kingdom", "Rest of world"],
                autopct="%1.1f%%", colors=[PRIMARY, ACCENT], startangle=90,
                wedgeprops=dict(width=0.45, edgecolor="white"))
    axes[0].set_title("Revenue Share: UK vs International")
    axes[1].barh(intl.index, intl.values, color=ACCENT)
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(gbp))
    axes[1].set_title("Top 10 International Markets by Revenue")
    fig.tight_layout()
    save(fig, "04_countries.png")
    return {"uk_revenue_share": float(uk_share),
            "top_intl_countries": list(c.drop("United Kingdom").head(3).index)}


# ---------------------------------------------------------------- 5. weekday x hour
def fig_heatmap(sales):
    inv = sales.drop_duplicates("Invoice")
    h = inv.pivot_table(index="Weekday", columns="Hour", values="Invoice", aggfunc="count")
    h = h.reindex([d for d in WEEKDAYS if d in h.index]).fillna(0)
    h = h.loc[:, (h.sum() > 50)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    sns.heatmap(h, cmap="Blues", ax=ax, linewidths=0.4, cbar_kws={"label": "Orders"})
    ax.set_title("Number of Orders by Weekday and Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    save(fig, "05_weekday_hour_heatmap.png")
    by_day = h.sum(axis=1)
    by_hour = h.sum(axis=0)
    return {"busiest_day": by_day.idxmax(), "busiest_hour": int(by_hour.idxmax()),
            "saturday_orders": int(by_day.get("Saturday", 0))}


# ---------------------------------------------------------------- 6. cancellations
def fig_cancellations(sales, cancels):
    cm = cancels[cancels["InvoiceDate"].dt.to_period("M").astype(str) <= LAST_FULL_MONTH]
    cm = cm.groupby(cm["InvoiceDate"].dt.to_period("M"))["Invoice"].nunique()
    sm = sales[sales["Month"] <= LAST_FULL_MONTH].groupby("Month")["Invoice"].nunique()
    sm.index = pd.PeriodIndex(sm.index, freq="M")
    rate = (cm / (cm + sm)).dropna()
    reasons = cancels.groupby("Country")["Revenue"].sum().sort_values().head(8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(rate.index.to_timestamp(), rate.values * 100, marker="o", color="#c0392b")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_title("Monthly Cancellation Rate (% of invoices)")
    axes[1].barh(reasons.index, -reasons.values, color="#c0392b")
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(gbp))
    axes[1].set_title("Cancelled Value by Country (Top 8)")
    fig.tight_layout()
    save(fig, "06_cancellations.png")
    n_c = cancels["Invoice"].nunique()
    n_s = sales["Invoice"].nunique()
    return {"cancel_invoices": int(n_c), "cancel_rate": float(n_c / (n_c + n_s)),
            "cancel_value": float(-cancels["Revenue"].sum())}


# ---------------------------------------------------------------- 7. RFM
SEGMENTS = [  # (name, rule on R score and FM score) checked in order
    ("Champions", lambda r, fm: (r >= 4) & (fm >= 4)),
    ("Loyal Customers", lambda r, fm: (r >= 3) & (fm >= 3)),
    ("New / Promising", lambda r, fm: (r >= 4) & (fm < 3)),
    ("At Risk", lambda r, fm: (r <= 2) & (fm >= 3)),
    ("Need Attention", lambda r, fm: (r == 3) & (fm < 3)),
    ("Hibernating / Lost", lambda r, fm: (r <= 2) & (fm < 3)),
]
SEG_COLORS = ["#1f5f8b", "#4a90b8", "#7fb3d5", "#e07b39", "#f2b880", "#b0b0b0"]


def rfm_table(sales):
    known = sales[~sales["IsGuest"]]
    snapshot = known["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = known.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot - d.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"))
    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    fm = (rfm["F"] + rfm["M"]) / 2
    rfm["Segment"] = "Other"
    for name, rule in reversed(SEGMENTS):
        rfm.loc[rule(rfm["R"], fm), "Segment"] = name
    rfm.loc[rfm["Segment"] == "Other", "Segment"] = "Need Attention"
    return rfm


def fig_rfm(sales):
    rfm = rfm_table(sales)
    order = [s for s, _ in SEGMENTS]
    seg = rfm.groupby("Segment").agg(Customers=("R", "size"), Revenue=("Monetary", "sum")).reindex(order)
    seg["CustShare"] = seg["Customers"] / seg["Customers"].sum()
    seg["RevShare"] = seg["Revenue"] / seg["Revenue"].sum()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(order))
    ax.barh(y + 0.2, seg["CustShare"] * 100, 0.4, color="#b0c4d4", label="% of customers")
    ax.barh(y - 0.2, seg["RevShare"] * 100, 0.4, color=SEG_COLORS, label="% of revenue")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    for i, (c, r) in enumerate(zip(seg["CustShare"], seg["RevShare"])):
        ax.text(r * 100 + 0.5, i - 0.2, f"{r:.0%}", va="center", fontsize=9)
        ax.text(c * 100 + 0.5, i + 0.2, f"{c:.0%}", va="center", fontsize=9, color="#555")
    ax.set_title("RFM Customer Segments: Share of Customers vs Share of Revenue")
    ax.legend(loc="lower right")
    save(fig, "07_rfm_segments.png")
    rfm.reset_index().to_csv(PROC / "rfm_customers.csv", index=False)

    # Pareto: share of revenue from top 20% of customers
    m = rfm["Monetary"].sort_values(ascending=False)
    top20 = m.head(int(len(m) * 0.2)).sum() / m.sum()
    return {"rfm_segments": {k: {"customers": int(v["Customers"]), "cust_share": float(v["CustShare"]),
                                 "rev_share": float(v["RevShare"])} for k, v in seg.iterrows()},
            "top20_customer_revenue_share": float(top20),
            "median_recency_days": float(rfm["Recency"].median()),
            "one_time_buyer_share": float((rfm["Frequency"] == 1).mean())}


# ---------------------------------------------------------------- 8. cohort retention
def fig_cohort(sales):
    known = sales[~sales["IsGuest"] & (sales["Month"] <= LAST_FULL_MONTH)].copy()
    known["OrderMonth"] = known["InvoiceDate"].dt.to_period("M")
    known["Cohort"] = known.groupby("Customer ID")["OrderMonth"].transform("min")
    known["Age"] = (known["OrderMonth"] - known["Cohort"]).apply(lambda d: d.n)
    counts = known.groupby(["Cohort", "Age"])["Customer ID"].nunique().unstack()
    retention = counts.div(counts[0], axis=0)
    retention = retention.loc[: pd.Period("2010-12", "M"), :12]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(retention * 100, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax,
                vmin=0, vmax=50, cbar_kws={"label": "% of cohort active"}, annot_kws={"size": 7})
    ax.set_yticklabels([str(p) for p in retention.index], rotation=0)
    ax.set_title("Monthly Cohort Retention (%) – first 12 months")
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort (first purchase month)")
    save(fig, "08_cohort_retention.png")
    later = retention.iloc[1:][1].mean()
    return {"first_cohort_m1_retention": float(retention.iloc[0][1]),
            "first_cohort_m12_retention": float(retention.iloc[0][12]),
            "avg_m1_retention_other_cohorts": float(later)}


# ---------------------------------------------------------------- 9. basket / order value
def fig_order_value(sales):
    orders = sales.groupby("Invoice").agg(Value=("Revenue", "sum"), Items=("StockCode", "nunique"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    sns.histplot(np.log10(orders["Value"]), bins=60, ax=axes[0], color=PRIMARY)
    axes[0].set_xticks([0, 1, 2, 3, 4], ["£1", "£10", "£100", "£1K", "£10K"])
    axes[0].axvline(np.log10(orders["Value"].median()), color=ACCENT, ls="--",
                    label=f"Median {gbp(orders['Value'].median())}")
    axes[0].legend()
    axes[0].set_title("Order Value Distribution (log scale)")
    axes[0].set_xlabel("Order value")
    sns.histplot(orders["Items"].clip(upper=100), bins=50, ax=axes[1], color=ACCENT)
    axes[1].set_title("Distinct Products per Order (capped at 100)")
    axes[1].set_xlabel("Distinct products")
    fig.tight_layout()
    save(fig, "09_order_value.png")
    return {"median_items_per_order": float(orders["Items"].median()),
            "orders_over_1000_share": float((orders["Value"] > 1000).mean())}


# ---------------------------------------------------------------- 10. price vs quantity
def fig_price_quantity(sales):
    p = sales.groupby("StockCode").agg(AvgPrice=("Price", "median"), Units=("Quantity", "sum"))
    p = p[(p["AvgPrice"] > 0) & (p["Units"] > 0)]
    corr = np.corrcoef(np.log10(p["AvgPrice"]), np.log10(p["Units"]))[0, 1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(p["AvgPrice"], p["Units"], s=8, alpha=0.35, color=PRIMARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    coef = np.polyfit(np.log10(p["AvgPrice"]), np.log10(p["Units"]), 1)
    xs = np.logspace(np.log10(p["AvgPrice"].min()), np.log10(p["AvgPrice"].max()), 50)
    ax.plot(xs, 10 ** np.polyval(coef, np.log10(xs)), color=ACCENT, lw=2,
            label=f"Trend (log-log r = {corr:.2f})")
    ax.set_title("Product Price vs Total Units Sold")
    ax.set_xlabel("Median unit price (£, log)")
    ax.set_ylabel("Total units sold (log)")
    ax.legend()
    save(fig, "10_price_vs_quantity.png")
    return {"price_qty_log_corr": float(corr), "price_elasticity_slope": float(coef[0])}


def run_all():
    sales, cancels = load()
    metrics = {"overview": overview(sales, cancels)}
    metrics["trend"] = fig_monthly_trend(sales)
    metrics["yoy"] = fig_yoy(sales)
    metrics["products"] = fig_top_products(sales)
    metrics["countries"] = fig_countries(sales)
    metrics["timing"] = fig_heatmap(sales)
    metrics["cancellations"] = fig_cancellations(sales, cancels)
    metrics["rfm"] = fig_rfm(sales)
    metrics["cohort"] = fig_cohort(sales)
    metrics["orders"] = fig_order_value(sales)
    metrics["price"] = fig_price_quantity(sales)
    metrics["cleaning_log"] = pd.read_csv(PROC / "cleaning_log.csv").to_dict("records")
    (PROC / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, default=str))
