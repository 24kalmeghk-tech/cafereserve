"""CafeReserve AI - Streamlit frontend. Run: streamlit run app.py"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ml_core as core

st.set_page_config(page_title="CaféReserve AI", page_icon="☕", layout="wide")
ESP, DARK, CREAM, LATTE, CARAMEL, SAGE = "#4B2E22", "#2D1E18", "#FAF6EF", "#E8D5BD", "#B87946", "#87977A"
COLORS = [CARAMEL, ESP, SAGE, LATTE]

st.markdown(f"""<style>
.stApp {{ background:{CREAM}; }}
[data-testid="stSidebar"] {{ background:{DARK}; }}
[data-testid="stSidebar"] * {{ color:{LATTE} !important; }}
h1,h2,h3 {{ color:{ESP}; font-family:Georgia,serif; }}
.card {{ background:#fff; border:1px solid {LATTE}; border-radius:14px; padding:16px 18px;
        box-shadow:0 2px 6px rgba(75,46,34,.08); }}
.card .l {{ color:#7a6657; font-size:13px; }} .card .v {{ color:{ESP}; font-size:28px; font-weight:700; }}
.card .s {{ color:{SAGE}; font-size:12px; }}
.stButton>button {{ background:{ESP}; color:#fff; border:0; border-radius:8px; padding:.5rem 1.2rem; }}
.stButton>button:hover {{ background:{CARAMEL}; color:#fff; }}
.note {{ background:{LATTE}55; border-left:4px solid {CARAMEL}; padding:8px 12px; border-radius:6px; color:{ESP}; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the existing models once...")
def load_all():
    raw = core.build_dataset()
    os.makedirs("data", exist_ok=True)
    raw.to_csv("data/cafe_reservations.csv", index=False)
    return raw, core.run_pipeline(raw)


try:
    RAW, R = load_all()
except Exception as e:                                   # graceful failure
    st.error(f"Could not load the project: {e}")
    st.stop()
DF = R["df"]


def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, h=330):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", font_color=ESP,
                      margin=dict(l=10, r=10, t=40, b=10), colorway=COLORS)
    return fig


def header(title, sub):
    st.markdown(f"# {title}")
    st.caption(sub)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def dashboard():
    header("Good day, Café Manager!", "Your café's reservations and customer insights at a glance.")
    c = st.columns(4)
    with c[0]: card("Total bookings", f"{len(DF):,}", "rows in dataset")
    with c[1]: card("Average party size", f"{DF.party_size.mean():.2f}", "people per booking")
    with c[2]: card("Total spend", f"₹{DF.spend.sum():,.0f}", "synthetic amounts")
    with c[3]: card("No-show rate", f"{DF.no_show.mean()*100:.1f}%", f"{int(DF.no_show.sum())} no-shows")
    st.write("")
    hours = DF.groupby("hour").size().reset_index(name="bookings")
    peak = hours.loc[hours.bookings.idxmax()]
    note(f"Peak hour: {int(peak.hour)}:00 with {int(peak.bookings)} bookings. "
         "Demo dataset: synthetic data for educational purposes.")
    a, b = st.columns(2)
    a.plotly_chart(style(px.bar(hours, x="hour", y="bookings", title="Bookings by hour")), use_container_width=True)
    b.plotly_chart(style(px.histogram(DF, x="party_size", title="Party size distribution")), use_container_width=True)
    a, b = st.columns(2)
    ns = DF.no_show.map({0: "Attended", 1: "No-show"}).value_counts().reset_index()
    ns.columns = ["status", "count"]
    a.plotly_chart(style(px.pie(ns, names="status", values="count", hole=.55, title="Attended vs no-show")), use_container_width=True)
    b.plotly_chart(style(px.histogram(DF, x="spend", nbins=30, title="Spending distribution")), use_container_width=True)
    st.subheader("Recent bookings")
    st.dataframe(DF.tail(8), use_container_width=True, hide_index=True)


def reservation():
    header("Reservation Prediction", "Estimate the party size using your Linear Regression models.")
    which = st.radio("Model", ["Multiple LR", "Simple LR"], horizontal=True)
    with st.form("reg"):
        c = st.columns(5)
        hour = c[0].number_input("Hour", 9, 21, 19)
        temp = c[1].number_input("Temperature (°C)", 10.0, 50.0, 28.0)
        adv = c[2].number_input("Advance days", 0, 14, 3)
        past = c[3].number_input("Past visits", 0, 19, 5)
        spend = c[4].number_input("Expected spend (₹)", 100.0, 1500.0, 600.0)
        go_ = st.form_submit_button("Predict Party Size")
    if go_:
        try:
            if which == "Simple LR":
                pred = R["simple"].predict(pd.DataFrame([[temp]], columns=["temp"]))[0]
            else:
                row = pd.DataFrame([[temp, hour, adv, past, spend]], columns=core.REG_FEATURES)   # same order as training
                pred = R["multi"].predict(row)[0]
            size = int(np.clip(round(pred), 1, 8))
            c1, c2 = st.columns(2)
            with c1: card("Estimated party size", f"{size} people", f"raw model output: {pred:.2f}")
            with c2: card("Model used", which, "trained on 80% of the data")
            note("This is an estimate only. The training data is random, so do not expect high accuracy.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.subheader("Test-set metrics (computed from your model)")
    st.dataframe(pd.DataFrame(R["reg_scores"]).T.round(3), use_container_width=True)
    n = min(60, len(R["reg_y_test"]))
    fig = go.Figure()
    fig.add_scatter(y=R["reg_y_test"][:n], name="Actual", line_color=ESP)
    fig.add_scatter(y=R["reg_pred_simple"][:n], name="Simple LR", line=dict(color=SAGE, dash="dash"))
    fig.add_scatter(y=R["reg_pred_multi"][:n], name="Multiple LR", line=dict(color=CARAMEL, dash="dash"))
    fig.update_layout(title="Actual vs predicted (first 60 test rows)")
    st.plotly_chart(style(fig), use_container_width=True)


def noshow():
    header("No-Show Prediction", "Use your classification models to flag bookings that may not turn up.")
    name = st.selectbox("Model", list(R["clf_models"]))
    thr = st.slider("Probability threshold", 0.05, 0.95, 0.50, 0.05)
    with st.form("clf"):
        c = st.columns(6)
        hour = c[0].number_input("Hour", 9, 21, 19)
        temp = c[1].number_input("Temp (°C)", 10.0, 50.0, 28.0)
        size = c[2].number_input("Party size", 1, 8, 4)
        adv = c[3].number_input("Advance days", 0, 14, 10)
        past = c[4].number_input("Past visits", 0, 19, 2)
        spend = c[5].number_input("Spend (₹)", 100.0, 1500.0, 600.0)
        go_ = st.form_submit_button("Predict No-Show Risk")
    if go_:
        try:
            row = pd.DataFrame([[hour, temp, size, adv, past, spend]], columns=core.FEATURES)   # same order as training
            x = R["scaler"].transform(row)                                                      # same fitted scaler
            p = R["clf_models"][name].predict_proba(x)[0, 1]                                    # real model probability
            label = "Potential No-Show" if p >= thr else "Likely to Attend"
            c1, c2 = st.columns(2)
            with c1: card("Prediction", label, f"threshold {thr:.2f}")
            with c2: card("No-show probability", f"{p*100:.1f}%", name)
            st.progress(float(p))
            note("An estimate, not a guarantee. Suggested action: send a reminder to flagged bookings. "
                 "Never cancel a booking automatically based on this output.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.subheader("Model comparison (test set, default threshold)")
    st.dataframe(pd.DataFrame(R["clf_scores"]).T.round(3), use_container_width=True)
    st.caption("Recall shows how many real no-shows are caught; precision shows how many flagged bookings were truly no-shows.")


def segmentation():
    header("Customer Segmentation", "K-Means and GMM groups of customers (3 clusters, as in your code).")
    cust = R["cust"]
    algo = st.radio("Algorithm", ["K-Means", "GMM"], horizontal=True)
    col = "cluster" if algo == "K-Means" else "gmm_cluster"
    a, b = st.columns(2)
    a.plotly_chart(style(px.line(x=list(range(1, 9)), y=R["wcss"], markers=True, title="Elbow method",
                                 labels={"x": "K", "y": "WCSS"})), use_container_width=True)
    plot = cust.assign(group=cust[col].astype(str))
    b.plotly_chart(style(px.scatter(plot, x="PC1", y="PC2", color="group", title=f"{algo} clusters (PCA, for visualization)",
                                    labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})), use_container_width=True)
    st.subheader("Cluster summary (average per cluster)")
    summ = cust.groupby(col)[["spend", "past_visits", "advance_days"]].mean().round(2)
    summ.insert(0, "customers", cust.groupby(col).size())
    st.dataframe(summ, use_container_width=True)
    if algo == "GMM":
        st.subheader("Soft membership probabilities (first 10 customers)")
        st.dataframe(cust[["prob_0", "prob_1", "prob_2"]].head(10).round(2), use_container_width=True)
        note("GMM gives each customer a probability for every cluster; K-Means assigns exactly one.")
    st.subheader("First 5 customers in each cluster")
    st.dataframe(cust.groupby(col).head(5)[["spend", "past_visits", "advance_days", col]].round(2), use_container_width=True)


def performance():
    header("Model Performance", "Every number here is computed by your code on the current dataset.")
    t1, t2, t3, t4 = st.tabs(["Regression", "Classification", "Ensembles", "Validation & tuning"])
    with t1:
        st.dataframe(pd.DataFrame(R["reg_scores"]).T.round(3), use_container_width=True)
    with t2:
        sc = pd.DataFrame(R["clf_scores"]).T
        st.dataframe(sc.round(3), use_container_width=True)
        st.plotly_chart(style(px.bar(sc.reset_index().melt("index"), x="index", y="value", color="variable",
                                     barmode="group", title="Classifier comparison", labels={"index": "Model"})), use_container_width=True)
        m = st.selectbox("Confusion matrix for", list(R["confusion"]))
        st.plotly_chart(style(px.imshow(R["confusion"][m], text_auto=True, x=["Attended", "No-show"], y=["Attended", "No-show"],
                                        color_continuous_scale=["#FAF6EF", CARAMEL], labels=dict(x="Predicted", y="Actual")), 320),
                        use_container_width=True)
    with t3:
        bag = R["bagging"]
        st.plotly_chart(style(px.line(x=list(bag), y=list(bag.values()), markers=True, title="Bagging: trees vs accuracy",
                                      labels={"x": "Number of estimators", "y": "Accuracy"})), use_container_width=True)
        st.dataframe(pd.DataFrame({"Number of Estimators": list(bag), "Accuracy": [round(v, 3) for v in bag.values()]}),
                     hide_index=True)
        st.dataframe(pd.Series(R["tree_vs_forest"], name="Accuracy").round(3).to_frame(), use_container_width=True)
    with t4:
        cv = R["cv_scores"]
        card("5-fold stratified CV (Random Forest)", f"{cv.mean():.3f}", f"std {cv.std():.3f}")
        st.write("")
        g, r = R["grid_best"], R["rand_best"]
        st.dataframe(pd.DataFrame({"Best params": [str(g[0]), str(r[0])], "CV score": [round(g[1], 3), round(r[1], 3)]},
                                  index=["GridSearchCV", "RandomizedSearchCV"]), use_container_width=True)
        imp = R["importance"]
        st.plotly_chart(style(px.bar(imp, orientation="h", title="Feature importance (tuned Random Forest)",
                                     labels={"value": "Importance", "index": "Feature"})), use_container_width=True)
        note(f"Top 3 features: {', '.join(reversed(list(imp.index[-3:])))}")


def explorer():
    header("Dataset Explorer", "Synthetic café bookings generated by your code (raw, before missing values are filled).")
    c = st.columns(3)
    with c[0]: card("Rows", f"{RAW.shape[0]:,}")
    with c[1]: card("Columns", f"{RAW.shape[1]}")
    with c[2]: card("Missing values", f"{int(RAW.isna().sum().sum())}", "temp column")
    h = st.slider("Hour range", 9, 21, (9, 21))
    ns = st.multiselect("No-show status", [0, 1], default=[0, 1], format_func=lambda v: "No-show" if v else "Attended")
    view = RAW[RAW.hour.between(*h) & RAW.no_show.isin(ns)]
    st.caption(f"Showing {len(view)} rows")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False).encode(), "cafe_reservations.csv", "text/csv")
    a, b = st.columns(2)
    a.subheader("Columns")
    a.dataframe(pd.DataFrame({"dtype": RAW.dtypes.astype(str), "missing": RAW.isna().sum()}))
    b.subheader("Descriptive statistics")
    b.dataframe(RAW.describe().round(2))


def about():
    header("About the Project", "CaféReserve AI: Table Reservation Demand Prediction and Customer Analytics")
    st.write("**Aim:** predict party size and no-shows and group customers for a café, using all seven lab practicals.")
    st.markdown("""
**Practicals:** 1 Missing values and EDA · 2 Simple and multiple linear regression · 3 Logistic, KNN and Naive Bayes ·
4 K-Means and elbow · 5 GMM and PCA · 6 Bagging and Random Forest · 7 Cross-validation, GridSearch and RandomizedSearch.

**Workflow:** Dataset → Preprocessing → Model training → Evaluation → Predictions → Café insights

**Technologies:** Python, pandas, NumPy, scikit-learn, Plotly, Streamlit

**Limitations:** the dataset is synthetic and random, so scores demonstrate the methods only. The scaler is fitted on all rows
before the train/test split, as in the original script; this is a simplification to mention in your viva.
""")


PAGES = {"📊  Dashboard": dashboard, "🪑  Reservation Prediction": reservation, "🚫  No-Show Prediction": noshow,
         "👥  Customer Segmentation": segmentation, "📈  Model Performance": performance,
         "🗂️  Dataset Explorer": explorer, "ℹ️  About Project": about}
st.sidebar.markdown("## ☕ CaféReserve AI")
st.sidebar.caption("Smart Café Analytics")
PAGES[st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")]()
