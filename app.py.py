"""Noon SBR - Customer Retention Analytics Dashboard (Streamlit)."""
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Noon SBR | Retention Analytics", page_icon="🟡", layout="wide")

YELLOW, DARK, RED, GREEN, GREY = "#FEEE00", "#1F2430", "#E5484D", "#2FA36B", "#8A93A5"
PALETTE = ["#1F2430", "#F5A623", "#2FA36B", "#E5484D", "#4C6EF5", "#8A93A5", "#B8860B", "#7B61FF"]
_here = Path(__file__).parent
DATA_PATH = _here / "noon_sbr_data.xlsx" if (_here / "noon_sbr_data.xlsx").exists() else _here / "data" / "noon_sbr_data.xlsx"
SHEETS = ["customers", "orders", "monthly_category_sales", "campaigns", "baskets"]
CHURN_FEATURES = ["tenure_months", "orders_12m", "avg_order_value_aed", "late_deliveries", "cancellations",
                  "complaints", "refunds", "avg_rating", "discount_usage_pct", "app_user", "noon_vip"]

st.markdown(f"""<style>
.block-container {{padding-top: 1.4rem;}}
div[data-testid="stMetric"] {{background:#F4F5F7;border-left:6px solid {YELLOW};padding:10px 14px;border-radius:8px;}}
h1,h2,h3 {{color:{DARK};}}
.insight {{background:#FFFBD6;border-left:5px solid {YELLOW};padding:10px 14px;border-radius:6px;margin:6px 0 14px 0;}}
</style>""", unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), colorway=PALETTE,
                      plot_bgcolor="white", font=dict(family="Arial"))
    return fig


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load(file_bytes=None):
    src = io.BytesIO(file_bytes) if file_bytes is not None else DATA_PATH
    d = pd.read_excel(src, sheet_name=SHEETS)
    d["orders"]["order_date"] = pd.to_datetime(d["orders"]["order_date"])
    d["monthly_category_sales"]["month"] = pd.to_datetime(d["monthly_category_sales"]["month"])
    d["campaigns"]["month"] = pd.to_datetime(d["campaigns"]["month"])
    return d


st.sidebar.markdown("## 🟡 Noon SBR")
up = st.sidebar.file_uploader("Upload your own Excel (optional)", type="xlsx")
try:
    D = load(up.getvalue() if up else None)
except Exception as e:  # noqa
    st.error(f"Could not read the Excel file: {e}")
    st.stop()
cust_all, orders_all = D["customers"], D["orders"]

page = st.sidebar.radio("Navigate", [
    "1. Executive summary", "2. Descriptive: metrics & CLV", "3. Segmentation (RFM + K-Means)",
    "4. Diagnostic: why customers churn", "5. Predictive: churn, interest, WTP",
    "6. Forecasting (ARIMA / SARIMA)", "7. Association rules", "8. Prescriptive: uplift, A/B, actions",
    "9. Campaign ROI", "10. Blogs & methodology"])

st.sidebar.markdown("---\n**Filters** (descriptive / diagnostic pages)")
f_city = st.sidebar.multiselect("City", sorted(cust_all.city.unique()))
f_ch = st.sidebar.multiselect("Acquisition channel", sorted(cust_all.acquisition_channel.unique()))
f_vip = st.sidebar.selectbox("Noon VIP", ["All", "VIP only", "Non-VIP"])
margin = st.sidebar.slider("Gross margin % (for CLV)", 5, 30, 12) / 100

cust = cust_all.copy()
if f_city: cust = cust[cust.city.isin(f_city)]
if f_ch: cust = cust[cust.acquisition_channel.isin(f_ch)]
if f_vip != "All": cust = cust[cust.noon_vip == (1 if f_vip == "VIP only" else 0)]
orders = orders_all[orders_all.customer_id.isin(cust.customer_id)]
if cust.empty:
    st.warning("No customers match the filters.")
    st.stop()


# ------------------------------------------------------------------ models (cached)
@st.cache_data(show_spinner="Training churn model...")
def churn_model(df):
    X, y = df[CHURN_FEATURES], df.churned
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=1)
    models = {"Logistic Regression": LogisticRegression(max_iter=2000), "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=1),
              "Gradient Boosting": GradientBoostingClassifier(random_state=1)}
    res, probs = {}, {}
    for n, m in models.items():
        Xa, Xb = (StandardScaler().fit(Xtr).transform(Xtr), StandardScaler().fit(Xtr).transform(Xte)) if n == "Logistic Regression" else (Xtr, Xte)
        m.fit(Xa, ytr); p = m.predict_proba(Xb)[:, 1]; probs[n] = p
        res[n] = dict(AUC=roc_auc_score(yte, p), Accuracy=accuracy_score(yte, p > .5), Precision=precision_score(yte, p > .5), Recall=recall_score(yte, p > .5), F1=f1_score(yte, p > .5))
    best = max(res, key=lambda k: res[k]["AUC"])
    final = RandomForestClassifier(300, min_samples_leaf=5, random_state=1).fit(X, y)  # scoring model on all customers
    imp = pd.Series(final.feature_importances_, CHURN_FEATURES).sort_values()
    return pd.DataFrame(res).T, yte.values, probs, best, imp, final.predict_proba(X)[:, 1]


@st.cache_data(show_spinner="Training offer-response models...")
def uplift_models(df):
    F = CHURN_FEATURES
    tr, te = train_test_split(df, test_size=.35, random_state=3)
    mt = GradientBoostingClassifier(max_depth=2, random_state=1).fit(tr[tr.ab_group == "Treatment"][F], tr[tr.ab_group == "Treatment"].repurchased_30d)
    mc = GradientBoostingClassifier(max_depth=2, random_state=1).fit(tr[tr.ab_group == "Control"][F], tr[tr.ab_group == "Control"].repurchased_30d)
    te = te.copy()
    te["uplift"] = mt.predict_proba(te[F])[:, 1] - mc.predict_proba(te[F])[:, 1]
    allc = df.copy()
    allc["uplift"] = mt.predict_proba(df[F])[:, 1] - mc.predict_proba(df[F])[:, 1]
    # market response: one model per offer type, recommend the offer with the higher predicted response
    for ot, col in [("Personalised Discount", "p_discount"), ("Proactive Service Recovery", "p_recovery")]:
        sub = df[df.offer_type == ot]
        m = GradientBoostingClassifier(max_depth=2, random_state=1).fit(sub[F], sub.repurchased_30d)
        allc[col] = m.predict_proba(df[F])[:, 1]
    mctrl = mc.predict_proba(df[F])[:, 1]
    allc["p_control"] = mctrl
    return te, allc


# ------------------------------------------------------------------ helpers
def rfm(df, ords):
    snap = ords.order_date.max()
    g = ords[ords.cancelled == 0].groupby("customer_id").agg(last=("order_date", "max"), F=("order_id", "count"), M=("order_value_aed", "sum"))
    g["R"] = (snap - g["last"]).dt.days
    g["R_s"] = pd.qcut(g.R.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    g["F_s"] = pd.qcut(g.F.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["M_s"] = pd.qcut(g.M.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def seg(r):
        if r.R_s >= 4 and r.F_s >= 4: return "Champions"
        if r.R_s >= 3 and r.F_s >= 3: return "Loyal"
        if r.R_s >= 4: return "New / Promising"
        if r.R_s <= 2 and r.F_s >= 4: return "At risk (high value)"
        if r.R_s <= 2: return "Hibernating / Lost"
        return "Needs attention"
    g["segment"] = g.apply(seg, axis=1)
    return g.reset_index()


def clv_table(df, probs):
    d = df.copy()
    d["churn_prob"] = probs
    d["annual_margin_aed"] = d.avg_order_value_aed * d.orders_12m * margin
    d["lifetime_yrs"] = (1 / d.churn_prob.clip(.08, .95)).clip(upper=5)
    d["clv_aed"] = d.annual_margin_aed * d.lifetime_yrs
    return d


# ------------------------------------------------------------------ pages
if page.startswith("1."):
    st.title("Noon – Customer Retention Dashboard")
    st.caption("Noon SBR project | Descriptive → Diagnostic → Predictive → Prescriptive | Data: simulated (see Excel README)")
    on_time = orders.on_time.mean()
    c = st.columns(6)
    c[0].metric("Customers", f"{len(cust):,}")
    c[1].metric("Churn rate", f"{cust.churned.mean():.1%}")
    c[2].metric("Repeat purchase rate", f"{(cust.orders_12m >= 2).mean():.1%}")
    c[3].metric("On-time delivery", f"{on_time:.1%}")
    c[4].metric("Complaints / 100 cust.", f"{cust.complaints.sum() / len(cust) * 100:.0f}")
    c[5].metric("Spend of churned (AED)", f"{cust[cust.churned == 1].total_spend_aed.sum() / 1e6:.2f}M")
    a, b = st.columns(2)
    cust["late_bucket"] = pd.cut(cust.late_deliveries, [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
    t = cust.groupby("late_bucket", observed=True).churned.mean().reset_index()
    a.plotly_chart(style(px.bar(t, x="late_bucket", y="churned", text_auto=".0%", title="Churn rate by number of late deliveries",
                                labels={"late_bucket": "Late deliveries (12m)", "churned": "Churn rate"}, color_discrete_sequence=[RED])), width="stretch")
    m = orders.assign(month=orders.order_date.dt.to_period("M").dt.to_timestamp()).groupby("month").agg(orders=("order_id", "count"), ontime=("on_time", "mean")).reset_index()
    fig = go.Figure([go.Bar(x=m.month, y=m.orders, name="Orders", marker_color=YELLOW), go.Scatter(x=m.month, y=m.ontime, name="On-time %", yaxis="y2", line=dict(color=DARK))])
    fig.update_layout(title="Orders and on-time delivery by month", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1]))
    b.plotly_chart(style(fig), width="stretch")
    base = cust[cust.late_deliveries == 0].churned.mean(); worst = cust[cust.late_deliveries >= 3].churned.mean()
    insight(f"Customers with 3+ late deliveries churn at <b>{worst:.0%}</b> vs <b>{base:.0%}</b> for customers with none. "
            f"Delivery reliability is the biggest controllable retention lever → target service recovery, not blanket discounts.")
    st.subheader("Problem → Approach map")
    st.table(pd.DataFrame({"Business problem": ["Customer retention & churn (MAIN)", "Sales forecasting & inventory", "Campaign effectiveness & ROI"],
                           "Pages": ["2, 3, 4, 5, 8", "6, 7", "8, 9"],
                           "Techniques": ["Metrics, RFM, K-Means, CLV, classification, uplift, A/B", "ARIMA, SARIMA, association rules", "Market response, uplift, A/B, ROI"]}))

elif page.startswith("2."):
    st.title("Descriptive analytics: know your metrics + CLV")
    insight("Blog 1 & 3: measure first. Metrics are grouped by Retention, Delivery and Seller reliability.")
    c = st.columns(4)
    c[0].metric("Churn rate", f"{cust.churned.mean():.1%}"); c[1].metric("Repeat purchase rate", f"{(cust.orders_12m >= 2).mean():.1%}")
    c[2].metric("Avg order value (AED)", f"{cust.avg_order_value_aed.mean():,.0f}"); c[3].metric("Avg rating", f"{cust.avg_rating.mean():.2f} / 5")
    c = st.columns(4)
    c[0].metric("On-time delivery %", f"{orders.on_time.mean():.1%}"); c[1].metric("Cancellation rate", f"{orders.cancelled.mean():.1%}")
    c[2].metric("Return rate", f"{orders.returned.mean():.1%}"); c[3].metric("Avg delivery days", f"{orders.delivery_days.mean():.1f}")
    a, b = st.columns(2)
    tier = orders.groupby("seller_tier").agg(on_time=("on_time", "mean"), returns=("returned", "mean"), cancel=("cancelled", "mean")).reset_index().melt("seller_tier")
    a.plotly_chart(style(px.bar(tier, x="seller_tier", y="value", color="variable", barmode="group", text_auto=".1%", title="Seller reliability by seller tier",
                                category_orders={"seller_tier": ["Top", "Mid", "Low"]})), width="stretch")
    cat = orders.groupby("category").order_value_aed.sum().sort_values().reset_index()
    b.plotly_chart(style(px.bar(cat, x="order_value_aed", y="category", orientation="h", title="Revenue by category (AED)", color_discrete_sequence=[DARK])), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(cust, x="avg_order_value_aed", nbins=40, title="Distribution of average order value", color_discrete_sequence=[YELLOW])), width="stretch")
    city = cust.groupby("city").agg(customers=("customer_id", "count"), churn=("churned", "mean")).reset_index()
    b.plotly_chart(style(px.bar(city, x="city", y="churn", text_auto=".0%", title="Churn by city", color_discrete_sequence=[RED])), width="stretch")
    st.subheader("Customer Lifetime Value (CLV)")
    _, _, _, _, _, prob_all = churn_model(cust_all)
    clv = clv_table(cust_all, prob_all)
    clv = clv[clv.customer_id.isin(cust.customer_id)]
    st.caption("CLV = annual margin × expected lifetime; expected lifetime = 1 / predicted churn probability (capped at 5 years). Change margin in the sidebar.")
    c = st.columns(3)
    c[0].metric("Average CLV (AED)", f"{clv.clv_aed.mean():,.0f}"); c[1].metric("Total CLV (AED)", f"{clv.clv_aed.sum() / 1e6:.2f}M")
    c[2].metric("CLV at risk (churn prob > 50%)", f"{clv[clv.churn_prob > .5].clv_aed.sum() / 1e6:.2f}M")
    clv["clv_band"] = pd.qcut(clv.clv_aed, 4, labels=["Low", "Medium", "High", "Top"])
    fig = px.box(clv, x="clv_band", y="churn_prob", title="Churn probability by CLV quartile", color_discrete_sequence=[DARK])
    st.plotly_chart(style(fig), width="stretch")
    insight("Where high-CLV customers also have high churn probability, retention spend pays back fastest (see page 8).")

elif page.startswith("3."):
    st.title("Segmentation: RFM + K-Means")
    insight("Blog 2: not every customer behaves the same. Segmentation shows where the real problem is.")
    r = rfm(cust, orders).merge(cust[["customer_id", "churned"]], on="customer_id")
    a, b = st.columns(2)
    s = r.groupby("segment").agg(customers=("customer_id", "count"), avg_spend=("M", "mean"), churn=("churned", "mean")).reset_index().sort_values("customers")
    a.plotly_chart(style(px.bar(s, x="customers", y="segment", orientation="h", title="RFM segments (customers)", color="churn", color_continuous_scale=["#2FA36B", "#FEEE00", "#E5484D"])), width="stretch")
    b.dataframe(s.rename(columns={"avg_spend": "Avg spend (AED)", "churn": "Churn rate"}).style.format({"Avg spend (AED)": "{:,.0f}", "Churn rate": "{:.1%}"}), width="stretch", hide_index=True)
    st.subheader("K-Means clustering: who to target and what to sell")
    feats = ["recency_days", "orders_12m", "total_spend_aed", "avg_rating", "late_deliveries", "complaints", "discount_usage_pct"]
    Z = StandardScaler().fit_transform(cust[feats])
    ks = list(range(2, 8)); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=5, random_state=1).fit(Z); inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_, sample_size=1500, random_state=1))
    a, b = st.columns(2)
    a.plotly_chart(style(px.line(x=ks, y=inertia, markers=True, title="Elbow curve (inertia)", labels={"x": "k", "y": "Inertia"}), 300), width="stretch")
    b.plotly_chart(style(px.line(x=ks, y=sil, markers=True, title="Silhouette score", labels={"x": "k", "y": "Silhouette"}), 300), width="stretch")
    k = st.slider("Number of clusters (k)", 2, 7, 4)
    cust["cluster"] = "Cluster " + (KMeans(k, n_init=10, random_state=1).fit_predict(Z) + 1).astype(str)
    pca = PCA(2, random_state=1).fit_transform(Z)
    st.plotly_chart(style(px.scatter(x=pca[:, 0], y=pca[:, 1], color=cust.cluster, opacity=.6, title="Clusters (PCA view)", labels={"x": "PC1", "y": "PC2"}), 420), width="stretch")
    prof = cust.groupby("cluster").agg(customers=("customer_id", "count"), recency=("recency_days", "mean"), orders=("orders_12m", "mean"), spend=("total_spend_aed", "mean"),
                                       rating=("avg_rating", "mean"), late=("late_deliveries", "mean"), complaints=("complaints", "mean"), discount_use=("discount_usage_pct", "mean"),
                                       churn=("churned", "mean"), top_category=("preferred_category", lambda x: x.mode()[0])).reset_index()
    st.dataframe(prof.style.format({"recency": "{:.0f}", "orders": "{:.1f}", "spend": "{:,.0f}", "rating": "{:.2f}", "late": "{:.2f}", "complaints": "{:.2f}", "discount_use": "{:.0%}", "churn": "{:.1%}"}), width="stretch", hide_index=True)
    hi = prof.sort_values("churn").iloc[-1]
    insight(f"<b>{hi.cluster}</b> has the highest churn ({hi.churn:.0%}) with {hi.late:.1f} late deliveries per customer on average, top category <b>{hi.top_category}</b>. Prioritise it for service recovery and category-specific offers.")

elif page.startswith("4."):
    st.title("Diagnostic analytics: why do customers churn?")
    insight("Blogs 4 & 5: link service failures to churn and to changes in purchase timing.")
    a, b = st.columns(2)
    for col, lab, w in [("complaints", "Complaints", a), ("cancellations", "Cancellations", b)]:
        t = cust.assign(x=cust[col].clip(upper=3).astype(int).astype(str).replace("3", "3+")).groupby("x").churned.agg(["mean", "count"]).reset_index()
        w.plotly_chart(style(px.bar(t, x="x", y="mean", text_auto=".0%", title=f"Churn rate by {lab.lower()} (12m)", labels={"x": lab, "mean": "Churn rate"}, color_discrete_sequence=[RED])), width="stretch")
    a, b = st.columns(2)
    t = cust.groupby("acquisition_channel").churned.mean().sort_values().reset_index()
    a.plotly_chart(style(px.bar(t, x="churned", y="acquisition_channel", orientation="h", text_auto=".0%", title="Churn by acquisition channel", color_discrete_sequence=[DARK])), width="stretch")
    t = cust.assign(seg=np.where(cust.noon_vip == 1, "VIP", "Non-VIP") + " / " + np.where(cust.app_user == 1, "App", "Web")).groupby("seg").churned.mean().reset_index()
    b.plotly_chart(style(px.bar(t, x="seg", y="churned", text_auto=".0%", title="Churn by VIP and app usage", color_discrete_sequence=[DARK])), width="stretch")
    st.subheader("Purchase timing after a bad delivery")
    o = orders.sort_values(["customer_id", "order_date"]).copy()
    o["next_gap_days"] = (o.groupby("customer_id").order_date.shift(-1) - o.order_date).dt.days
    g = o.dropna(subset=["next_gap_days"]).assign(delivery=lambda d: np.where(d.on_time == 1, "On-time", "Late")).groupby("delivery").next_gap_days.mean().reset_index()
    a, b = st.columns([1, 2])
    a.plotly_chart(style(px.bar(g, x="delivery", y="next_gap_days", text_auto=".0f", title="Avg days until next order", color="delivery", color_discrete_map={"On-time": GREEN, "Late": RED}), 320), width="stretch")
    if len(g) == 2:
        lt, ot = g.set_index("delivery").next_gap_days["Late"], g.set_index("delivery").next_gap_days["On-time"]
        b.markdown(f"<br><br>After a <b>late</b> delivery the next order comes <b>{lt - ot:.0f} days later</b> ({lt / ot - 1:+.0%}) than after an on-time delivery. "
                   f"<br>→ Build an alert: if a regular customer's gap exceeds ~{ot * 1.5:.0f} days after a late order, trigger a follow-up.", unsafe_allow_html=True)
    st.subheader("Correlation with churn")
    num = ["churned", "late_deliveries", "cancellations", "refunds", "complaints", "avg_rating", "orders_12m", "discount_usage_pct", "tenure_months", "noon_vip", "app_user"]
    corr = cust[num].corr()
    st.plotly_chart(style(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"), 520), width="stretch")

elif page.startswith("5."):
    st.title("Predictive analytics")
    tab1, tab2, tab3 = st.tabs(["Churn prediction (classification)", "Customer interest (classification)", "Willingness to pay (regression)"])
    with tab1:
        insight("Blog 4: predict churn before the customer is lost. <b>Recency is excluded from the features</b> because churn is defined by recency (it would leak the answer).")
        met, yte, probs, best, imp, prob_all = churn_model(cust_all)
        st.dataframe(met.style.format("{:.3f}").highlight_max(axis=0, color="#FFFBD6"), width="stretch")
        a, b = st.columns(2)
        fig = go.Figure()
        for n, p in probs.items():
            fpr, tpr, _ = roc_curve(yte, p); fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{n} (AUC {roc_auc_score(yte, p):.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=GREY), showlegend=False)); fig.update_layout(title="ROC curves", xaxis_title="FPR", yaxis_title="TPR")
        a.plotly_chart(style(fig), width="stretch")
        b.plotly_chart(style(px.bar(imp, orientation="h", title="Feature importance (Random Forest)", color_discrete_sequence=[DARK]).update_layout(showlegend=False)), width="stretch")
        th = st.slider("Decision threshold", .1, .9, .5, .05)
        p = probs[best]; cm = confusion_matrix(yte, p > th)
        a, b = st.columns(2)
        a.plotly_chart(style(px.imshow(cm, text_auto=True, x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"], color_continuous_scale="ylorbr", title=f"Confusion matrix ({best})"), 340), width="stretch")
        b.markdown(f"<br>At threshold <b>{th:.2f}</b>: precision <b>{precision_score(yte, p > th, zero_division=0):.2f}</b>, recall <b>{recall_score(yte, p > th):.2f}</b>. "
                   f"Lower thresholds catch more churners but contact more customers who would have stayed.", unsafe_allow_html=True)
        st.subheader("At-risk customer list")
        sc = clv_table(cust_all, prob_all)
        sc = sc[sc.customer_id.isin(cust.customer_id)]
        risk = sc[(sc.churned == 0) & (sc.churn_prob >= th)].sort_values("clv_aed", ascending=False)
        st.caption("Active customers (bought in last 90 days) with a high predicted churn probability, sorted by CLV.")
        show = risk[["customer_id", "city", "churn_prob", "clv_aed", "late_deliveries", "complaints", "avg_rating"]].head(200)
        st.dataframe(show.style.format({"churn_prob": "{:.0%}", "clv_aed": "{:,.0f}", "avg_rating": "{:.1f}"}), width="stretch", hide_index=True)
        st.download_button("Download at-risk list (CSV)", risk.to_csv(index=False), "at_risk_customers.csv")
    with tab2:
        insight("Classification: will the customer be interested in a Noon VIP / retention offer? Tells us who to target.")
        F2 = ["age", "tenure_months", "orders_12m", "avg_order_value_aed", "discount_usage_pct", "app_user", "noon_vip", "avg_rating", "complaints"]
        X, y = cust_all[F2], cust_all.interested_vip_offer
        Xtr, Xte, ytr, yte2 = train_test_split(X, y, test_size=.25, stratify=y, random_state=2)
        sc2 = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000).fit(sc2.transform(Xtr), ytr); rf = RandomForestClassifier(300, min_samples_leaf=5, random_state=2).fit(Xtr, ytr)
        pl, pr = lr.predict_proba(sc2.transform(Xte))[:, 1], rf.predict_proba(Xte)[:, 1]
        st.dataframe(pd.DataFrame({m: dict(AUC=roc_auc_score(yte2, p), Accuracy=accuracy_score(yte2, p > .5), F1=f1_score(yte2, p > .5)) for m, p in [("Logistic Regression", pl), ("Random Forest", pr)]}).T.style.format("{:.3f}"), width="stretch")
        coef = pd.Series(lr.coef_[0], F2).sort_values()
        st.plotly_chart(style(px.bar(coef, orientation="h", title="What drives interest? (standardised logistic coefficients)", color=coef > 0, color_discrete_map={True: GREEN, False: RED}).update_layout(showlegend=False)), width="stretch")
        insight("Customers who already use discounts heavily and are on the app are the most interested; older customers less so. Use this to select who receives the offer.")
    with tab3:
        insight("Regression: how much is a customer willing to pay? Useful for pricing, bundles and offer depth.")
        F3 = ["age", "avg_order_value_aed", "discount_usage_pct", "noon_vip", "app_user", "orders_12m", "tenure_months"]
        d3 = pd.get_dummies(cust_all[F3 + ["city"]], columns=["city"], drop_first=True).astype(float)
        y3 = cust_all.willingness_to_pay_aed
        Xtr, Xte, ytr, yte3 = train_test_split(d3, y3, test_size=.25, random_state=2)
        lin = LinearRegression().fit(Xtr, ytr); gb = GradientBoostingRegressor(random_state=2).fit(Xtr, ytr)
        st.dataframe(pd.DataFrame({n: dict(R2=r2_score(yte3, m.predict(Xte)), MAE_AED=mean_absolute_error(yte3, m.predict(Xte))) for n, m in [("Linear Regression", lin), ("Gradient Boosting", gb)]}).T.style.format("{:.2f}"), width="stretch")
        a, b = st.columns(2)
        a.plotly_chart(style(px.scatter(x=yte3, y=lin.predict(Xte), opacity=.5, title="Actual vs predicted WTP (linear)", labels={"x": "Actual (AED)", "y": "Predicted (AED)"}, color_discrete_sequence=[DARK])), width="stretch")
        w = cust_all.groupby("preferred_category").willingness_to_pay_aed.mean().sort_values().reset_index()
        b.plotly_chart(style(px.bar(w, x="willingness_to_pay_aed", y="preferred_category", orientation="h", title="Average willingness to pay by preferred category (AED)", color_discrete_sequence=[YELLOW])), width="stretch")

elif page.startswith("6."):
    st.title("Sales forecasting: ARIMA vs SARIMA")
    insight("Blog 6: better demand forecasts → better stock planning → fewer stock-outs and fewer late deliveries.")
    ms = D["monthly_category_sales"]
    a, b, c = st.columns(3)
    cat = a.selectbox("Category", ["All categories"] + sorted(ms.category.unique()))
    metric = b.selectbox("Metric", ["revenue_aed", "units_sold"])
    h = c.slider("Forecast horizon (months)", 3, 12, 6)
    s = (ms if cat == "All categories" else ms[ms.category == cat]).groupby("month")[metric].sum().asfreq("MS")
    train, test = s.iloc[:-6], s.iloc[-6:]

    def fit(y, seasonal):
        return SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12) if seasonal else (0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    try:
        res = {}
        for n, sea in [("ARIMA(1,1,1)", False), ("SARIMA(1,1,1)(1,1,0,12)", True)]:
            pr = fit(train, sea).forecast(6)
            res[n] = dict(MAPE=float(np.mean(np.abs((test.values - pr.values) / test.values))), MAE=float(np.mean(np.abs(test.values - pr.values))), RMSE=float(np.sqrt(np.mean((test.values - pr.values) ** 2))))
        comp = pd.DataFrame(res).T
        st.subheader("Hold-out test (last 6 months)")
        st.dataframe(comp.style.format({"MAPE": "{:.1%}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}"}), width="stretch")
        full = fit(s, True); fc = full.get_forecast(h); pm, ci = fc.predicted_mean, fc.conf_int()
        fig = go.Figure([go.Scatter(x=s.index, y=s.values, name="Actual", line=dict(color=DARK)),
                         go.Scatter(x=pm.index, y=pm.values, name="SARIMA forecast", line=dict(color=RED, dash="dash")),
                         go.Scatter(x=list(ci.index) + list(ci.index[::-1]), y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]), fill="toself", fillcolor="rgba(229,72,77,.15)", line=dict(width=0), name="95% interval")])
        fig.update_layout(title=f"{cat}: {metric} forecast (SARIMA)")
        st.plotly_chart(style(fig, 440), width="stretch")
        best = comp.MAPE.idxmin()
        insight(f"On the hold-out period <b>{best}</b> is more accurate (MAPE {comp.MAPE.min():.1%}). The seasonal model captures the White Friday / Ramadan peaks that plain ARIMA misses.")
        if metric == "units_sold":
            stock = ms if cat == "All categories" else ms[ms.category == cat]
            last_stock = stock[stock.month == stock.month.max()].closing_stock_units.sum()
            plan = ci.iloc[:, 1].clip(lower=0)
            st.subheader("Inventory planning")
            tbl = pd.DataFrame({"Month": pm.index.strftime("%b %Y"), "Forecast units": pm.values.round(0), "Stock to hold (95% upper)": plan.values.round(0)})
            st.dataframe(tbl.style.format({"Forecast units": "{:,.0f}", "Stock to hold (95% upper)": "{:,.0f}"}), width="stretch", hide_index=True)
            st.caption(f"Latest closing stock: {last_stock:,.0f} units. Holding the 95% upper bound before peak months reduces stock-outs.")
        so = ms.assign(m=ms.month.dt.month).groupby("m").stockout_days.mean().reset_index()
        st.plotly_chart(style(px.bar(so, x="m", y="stockout_days", title="Average stock-out days by calendar month (peak months suffer most)", color_discrete_sequence=[RED]), 300), width="stretch")
    except Exception as e:  # noqa
        st.error(f"Model could not be fitted for this selection: {e}")

elif page.startswith("7."):
    st.title("Association rule mining (market basket)")
    insight("Which products are bought together? Support = how common, confidence = P(B | A), lift > 1 = positive association.")
    bk = D["baskets"]
    M = pd.crosstab(bk.basket_id, bk["product"]).clip(upper=1).astype(int)
    n = len(M)
    a, b = st.columns(2)
    ms_ = a.slider("Minimum support", .005, .10, .01, .005, format="%.3f"); mc_ = b.slider("Minimum confidence", .05, .9, .20, .05)
    co = M.T.values @ M.values; sup_i = np.diag(co) / n
    rows = []
    for i, ai in enumerate(M.columns):
        for j, bj in enumerate(M.columns):
            if i != j and co[i, j] / n >= ms_:
                conf = co[i, j] / co[i, i]
                if conf >= mc_: rows.append((ai, bj, co[i, j] / n, conf, conf / sup_i[j]))
    rules = pd.DataFrame(rows, columns=["If customer buys", "Then also buys", "support", "confidence", "lift"]).sort_values("lift", ascending=False)
    if rules.empty:
        st.warning("No rules at these thresholds - lower support/confidence.")
    else:
        st.dataframe(rules.style.format({"support": "{:.3f}", "confidence": "{:.0%}", "lift": "{:.2f}"}), width="stretch", hide_index=True)
        rules["rule"] = rules["If customer buys"] + " → " + rules["Then also buys"]
        st.plotly_chart(style(px.scatter(rules, x="support", y="confidence", size="lift", color="lift", hover_name="rule", color_continuous_scale="YlOrRd", title="Rules: support vs confidence (bubble = lift)"), 440), width="stretch")
        t = rules.iloc[0]
        insight(f"Strongest rule: <b>{t['If customer buys']} → {t['Then also buys']}</b> (lift {t.lift:.1f}, confidence {t.confidence:.0%}). Use it for bundles, 'frequently bought together' and post-purchase cross-sell offers to keep customers active.")

elif page.startswith("8."):
    st.title("Prescriptive analytics: who to treat, with what, and does it work?")
    te, allc = uplift_models(cust_all)
    tab1, tab2, tab3, tab4 = st.tabs(["A/B test (Blog 9)", "Uplift modelling (Blog 8)", "Market response (Blog 7)", "Action list"])
    with tab1:
        insight("Group A = existing process (Control). Group B = new retention intervention (Treatment). Outcome: repurchase within 30 days.")
        cA, cB = cust_all[cust_all.ab_group == "Control"].repurchased_30d, cust_all[cust_all.ab_group == "Treatment"].repurchased_30d
        pa, pb, na, nb = cA.mean(), cB.mean(), len(cA), len(cB)
        se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb); z = (pb - pa) / se; pv = 2 * (1 - stats.norm.cdf(abs(z)))
        c = st.columns(4); c[0].metric("Control repurchase", f"{pa:.1%}"); c[1].metric("Treatment repurchase", f"{pb:.1%}", f"{(pb - pa) * 100:+.1f} pp")
        c[2].metric("p-value", f"{pv:.3f}"); c[3].metric("95% CI of lift", f"[{(pb - pa - 1.96 * se) * 100:.1f}, {(pb - pa + 1.96 * se) * 100:.1f}] pp")
        insight("Overall the offer looks " + ("significant" if pv < .05 else "<b>not significant</b> on average") + ", yet the effect is hidden because offers help some customers and not others → uplift modelling (next tab).")
        ot = cust_all.groupby("offer_type").repurchased_30d.agg(["mean", "count"]).reset_index()
        st.plotly_chart(style(px.bar(ot, x="offer_type", y="mean", text_auto=".1%", title="Repurchase rate by arm / offer type", color_discrete_sequence=[DARK]), 340), width="stretch")
    with tab2:
        insight("Uplift = P(repurchase | offer) − P(repurchase | no offer). Target only 'persuadables'. T-learner (two gradient boosting models), evaluated on held-out customers.")
        te = te.copy(); te["decile"] = pd.qcut(te.uplift.rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
        d = te.groupby(["decile", "ab_group"]).repurchased_30d.mean().unstack(); d["observed_uplift"] = d["Treatment"] - d["Control"]
        st.plotly_chart(style(px.bar(d.reset_index(), x="decile", y="observed_uplift", text_auto=".1%", title="Observed uplift by predicted-uplift decile (1 = most persuadable)", color_discrete_sequence=[GREEN]).update_yaxes(tickformat=".0%")), width="stretch")
        st.subheader("ROI of targeting the top X% only")
        c = st.columns(3); cost = c[0].number_input("Offer cost per customer (AED)", 1.0, 100.0, 15.0); val = c[1].number_input("Margin per repurchase (AED)", 1.0, 200.0, float(round(cust_all.avg_order_value_aed.mean() * margin * 2, 1)))
        scale = c[2].number_input("Customer base to treat", 1000, 5_000_000, 100_000, 1000)
        rows = []
        for k in range(10, 101, 10):
            sel = te[te.decile <= k // 10]; t, ct = sel[sel.ab_group == "Treatment"].repurchased_30d, sel[sel.ab_group == "Control"].repurchased_30d
            up = t.mean() - ct.mean(); n_t = scale * k / 100
            rows.append(dict(target_pct=k, uplift=up, incremental_repurchases=up * n_t, cost=n_t * cost, profit=up * n_t * val - n_t * cost))
        R = pd.DataFrame(rows)
        st.plotly_chart(style(px.line(R, x="target_pct", y="profit", markers=True, title="Expected net profit (AED) by % of customers targeted", color_discrete_sequence=[DARK]).update_xaxes(title="% of customers targeted (ranked by uplift)")), width="stretch")
        bestk = R.loc[R.profit.idxmax()]
        insight(f"Profit peaks when targeting the top <b>{bestk.target_pct:.0f}%</b> (≈ AED {bestk.profit:,.0f}). Treating everyone wastes budget on customers who would buy anyway or never respond.")
        st.dataframe(R.style.format({"uplift": "{:.1%}", "incremental_repurchases": "{:,.0f}", "cost": "{:,.0f}", "profit": "{:,.0f}"}), width="stretch", hide_index=True)
    with tab3:
        insight("Different customers respond to different offers. Compare predicted response to a personalised discount vs proactive service recovery.")
        allc["best_offer"] = np.where(allc.p_discount >= allc.p_recovery, "Personalised Discount", "Proactive Service Recovery")
        allc["best_gain"] = np.where(allc.p_discount >= allc.p_recovery, allc.p_discount, allc.p_recovery) - allc.p_control
        a, b = st.columns(2)
        a.plotly_chart(style(px.pie(allc.best_offer.value_counts().reset_index(), names="best_offer", values="count", title="Recommended offer mix", hole=.5, color_discrete_sequence=[YELLOW, DARK]), 340), width="stretch")
        allc["why"] = np.where((allc.late_deliveries + allc.complaints) >= 2, "Service issues", "Few service issues")
        t = allc.groupby(["why", "best_offer"]).size().reset_index(name="n")
        b.plotly_chart(style(px.bar(t, x="why", y="n", color="best_offer", barmode="group", title="Best offer by service experience", color_discrete_sequence=[YELLOW, DARK]), 340), width="stretch")
        insight("Customers with late deliveries / complaints respond better to service recovery (apology, proactive tracking, free delivery); price-sensitive customers respond to discounts.")
    with tab4:
        st.subheader("Recommended action per active customer")
        _, _, _, _, _, prob_all = churn_model(cust_all)
        A = clv_table(allc, prob_all)
        A["best_offer"] = np.where(A.p_discount >= A.p_recovery, "Personalised Discount", "Proactive Service Recovery")
        thr = st.slider("Treat if predicted uplift ≥", 0.0, .30, .05, .01)
        A["action"] = np.where((A.churned == 0) & (A.uplift >= thr) & (A.churn_prob >= .3), A.best_offer, "No offer (would buy anyway / low risk)")
        A.loc[(A.churned == 1) & (A.uplift >= thr), "action"] = "Win-back: " + A.best_offer
        c = st.columns(3); c[0].metric("Customers to treat", f"{(~A.action.str.startswith('No')).sum():,}"); c[1].metric("CLV of treated (AED)", f"{A[~A.action.str.startswith('No')].clv_aed.sum() / 1e6:.2f}M")
        c[2].metric("Avg predicted uplift (treated)", f"{A[~A.action.str.startswith('No')].uplift.mean():.1%}")
        st.plotly_chart(style(px.bar(A.action.value_counts().reset_index(), x="action", y="count", title="Action mix", color_discrete_sequence=[DARK]), 320), width="stretch")
        out = A[~A.action.str.startswith("No")].sort_values("clv_aed", ascending=False)[["customer_id", "city", "churn_prob", "clv_aed", "uplift", "action"]]
        st.dataframe(out.head(200).style.format({"churn_prob": "{:.0%}", "clv_aed": "{:,.0f}", "uplift": "{:.1%}"}), width="stretch", hide_index=True)
        st.download_button("Download action list (CSV)", out.to_csv(index=False), "retention_actions.csv")

elif page.startswith("9."):
    st.title("Campaign effectiveness & ROI")
    cp = D["campaigns"].copy()
    cp["roi"] = (cp.revenue_aed - cp.cost_aed) / cp.cost_aed
    cp["conv_rate"] = cp.conversions / cp.customers_targeted
    c = st.columns(4)
    c[0].metric("Total cost (AED)", f"{cp.cost_aed.sum():,.0f}"); c[1].metric("Total revenue (AED)", f"{cp.revenue_aed.sum():,.0f}")
    c[2].metric("Overall ROI", f"{(cp.revenue_aed.sum() - cp.cost_aed.sum()) / cp.cost_aed.sum():.1f}x"); c[3].metric("Avg conversion rate", f"{cp.conv_rate.mean():.1%}")
    a, b = st.columns(2)
    g = cp.groupby("channel").agg(cost=("cost_aed", "sum"), rev=("revenue_aed", "sum"), conv=("conv_rate", "mean")).reset_index(); g["ROI"] = (g.rev - g.cost) / g.cost
    a.plotly_chart(style(px.bar(g.sort_values("ROI"), x="ROI", y="channel", orientation="h", text_auto=".1f", title="ROI by channel (x)", color_discrete_sequence=[DARK])), width="stretch")
    g2 = cp.groupby("offer_type").agg(cost=("cost_aed", "sum"), rev=("revenue_aed", "sum"), conv=("conv_rate", "mean")).reset_index()
    b.plotly_chart(style(px.bar(g2, x="offer_type", y="conv", text_auto=".1%", title="Conversion rate by offer type", color_discrete_sequence=[YELLOW])), width="stretch")
    insight("Generic discounts convert worst while costing the most margin; personalised and service-recovery offers convert about twice as well. Shift budget towards them and verify with A/B tests.")
    st.dataframe(cp.style.format({"roi": "{:.1f}x", "conv_rate": "{:.1%}", "cost_aed": "{:,.0f}", "revenue_aed": "{:,.0f}"}), width="stretch", hide_index=True)

else:
    st.title("The 9 blogs → what we did in this dashboard")
    st.table(pd.DataFrame({
        "#": range(1, 10),
        "Blog": ["Know your metrics", "Segmentation", "Customer Lifetime Value", "Churn prediction", "Purchase timing / behaviour", "Sales forecasting", "Market response modelling", "Uplift modelling", "A/B testing"],
        "Business problem": ["All", "Retention", "Retention", "Retention (main)", "Retention", "Forecasting & inventory", "Campaign ROI", "Campaign ROI", "Campaign ROI"],
        "Where in dashboard": ["Page 2", "Page 3", "Page 2 (CLV)", "Page 5", "Page 4", "Page 6", "Page 8 (Market response)", "Page 8 (Uplift)", "Page 8 (A/B test)"]}))
    st.markdown("""
**Analytics types:** Descriptive (pages 1–2) · Diagnostic (3–4) · Predictive (5–7) · Prescriptive (8–9)

**Algorithms:** Classification (churn, offer interest) · K-Means clustering · Association rules (support, confidence, lift) · Linear/GB regression (willingness to pay) · ARIMA / SARIMA

**Data note:** the dataset is simulated for the academic project (random seed 42). Patterns (e.g. late deliveries → churn) were built in, so findings show the *method*, not real Noon numbers. Upload real data in the sidebar using the same sheets/columns.
""")
