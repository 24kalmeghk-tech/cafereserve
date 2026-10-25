"""Cafe Insight — Smart Reservations. Happier Customers. Better Decisions."""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ml_core as core

st.set_page_config(page_title="Cafe Insight", page_icon="☕", layout="wide", initial_sidebar_state="expanded")
ESP, DARK, CREAM, LATTE, CARAMEL, SAGE = "#4B2E22", "#2D1E18", "#FAF6EF", "#E8D5BD", "#B87946", "#87977A"
COLORS = [CARAMEL, ESP, SAGE, LATTE]

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp {{ background:linear-gradient(180deg,#fbf8f2 0%,{CREAM} 48%,#f5eee5 100%); color:{DARK}; }}
.block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1440px; }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,{DARK} 0%,#40291f 100%); border-right:1px solid #604638; }}
[data-testid="stSidebar"] * {{ color:{LATTE} !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{ color:#fff7ed !important; font-family:'Playfair Display',Georgia,serif; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:9px 12px; border-radius:10px; margin:2px 0; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:#5b3b2d; }}
h1,h2,h3 {{ color:{ESP}; font-family:'Playfair Display',Georgia,serif; letter-spacing:-.02em; }}
p,label,.stCaption {{ font-family:'DM Sans',sans-serif; }}
/* High-contrast labels and clearly visible radio/select options */
.stApp label, .stApp [data-testid="stWidgetLabel"] p {{ color:{DARK} !important; font-weight:600 !important; }}
.stApp [role="radiogroup"] label {{ color:{DARK} !important; background:#fffdf9; border:1px solid #ead9c8; padding:8px 12px; border-radius:10px; margin-right:6px; }}
.stApp [role="radiogroup"] label p {{ color:{DARK} !important; font-weight:600 !important; }}
.stApp [role="radiogroup"] label:has(input:checked) {{ background:#ead9c8; border-color:{CARAMEL}; }}
.stApp [data-baseweb="select"] * {{ color:{DARK} !important; }}
.stApp input, .stApp textarea {{ color:{DARK} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ background:transparent; border-color:transparent; color:{LATTE} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ color:{LATTE} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background:#5b3b2d; border-color:#8f6b55; }}
h1 {{ font-size:2.4rem !important; margin-bottom:.15rem; }}
[data-testid="stMetric"] {{ background:#fffdf9; border:1px solid #ead9c8; border-radius:16px; padding:18px 20px; box-shadow:0 5px 18px rgba(75,46,34,.055); }}
[data-testid="stMetricLabel"] p {{ color:#806b5b !important; font-size:.9rem; }}
[data-testid="stMetricValue"] {{ color:{ESP} !important; font-weight:700; }}
.card {{ background:linear-gradient(145deg,#fffefa,#fff); border:1px solid #ead9c8; border-radius:18px; padding:18px 20px; box-shadow:0 6px 22px rgba(75,46,34,.065); min-height:108px; }}
.card .l {{ color:#806b5b; font-size:13px; font-weight:600; }} .card .v {{ color:{ESP}; font-size:28px; font-weight:700; margin-top:4px; }} .card .s {{ color:{SAGE}; font-size:12px; margin-top:4px; }}
.stButton>button,[data-testid="stFormSubmitButton"] button {{ background:{ESP}; color:#fff; border:0; border-radius:10px; padding:.58rem 1.15rem; font-weight:600; transition:all .2s ease; }}
.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover {{ background:{CARAMEL}; color:#fff; transform:translateY(-1px); }}
.note {{ background:#f2e4d3; border-left:4px solid {CARAMEL}; padding:12px 15px; border-radius:10px; color:{ESP}; }}
.action-card {{ background:#fffdf9; border:1px solid #ead9c8; border-radius:16px; padding:17px 18px; margin:8px 0 12px; box-shadow:0 4px 16px rgba(75,46,34,.045); }}
.action-title {{ font-weight:700; color:{ESP}; font-size:1.02rem; margin-bottom:5px; }} .action-body {{ color:#695649; font-size:.93rem; line-height:1.5; }}
.pill {{ display:inline-block; padding:4px 9px; border-radius:999px; background:#e7eddf; color:#40563a; font-size:11px; font-weight:700; margin-bottom:8px; }}
div[data-testid="stDataFrame"] {{ border:1px solid #ead9c8; border-radius:12px; overflow:hidden; }}
/* High-contrast values in number inputs, dropdowns, and other form controls. */
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stDateInput"] input,
.stApp [data-testid="stTimeInput"] input,
.stApp textarea {{ background:#fffdf9 !important; color:#2D1E18 !important; -webkit-text-fill-color:#2D1E18 !important; caret-color:#2D1E18 !important; border:1px solid #bda58f !important; }}
.stApp [data-testid="stNumberInput"] button {{ background:#fffdf9 !important; color:#2D1E18 !important; border-left:1px solid #ead9c8 !important; }}
.stApp [data-testid="stNumberInput"] button svg {{ fill:#2D1E18 !important; color:#2D1E18 !important; }}
.stApp [data-baseweb="select"] > div {{ background:#fffdf9 !important; color:#2D1E18 !important; border-color:#bda58f !important; }}
.stApp [data-baseweb="select"] input {{ background:transparent !important; color:#2D1E18 !important; -webkit-text-fill-color:#2D1E18 !important; }}
.stApp [data-baseweb="select"] svg {{ fill:#2D1E18 !important; }}
.stApp [data-testid="stForm"] {{ color:#2D1E18 !important; }}
/* Make sidebar navigation labels consistently light and readable on the dark background. */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span {{ color:#FFF7ED !important; -webkit-text-fill-color:#FFF7ED !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background:#684633 !important; border-color:#C58B5B !important; }}
/* Explicit weather-select styling: readable closed value and open option menu. */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] span,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] input {{
    background-color:#fffdf9 !important;
    color:#2D1E18 !important;
    -webkit-text-fill-color:#2D1E18 !important;
    opacity:1 !important;
}}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    border:1px solid #b87946 !important;
    border-radius:9px !important;
    min-height:44px !important;
}}
.stApp [data-baseweb="popover"] ul,
.stApp [data-baseweb="popover"] li,
.stApp [data-baseweb="popover"] li div,
.stApp [role="listbox"], .stApp [role="option"] {{
    background:#fffdf9 !important;
    color:#2D1E18 !important;
    -webkit-text-fill-color:#2D1E18 !important;
    opacity:1 !important;
}}
.stApp [role="option"]:hover,
.stApp [role="option"][aria-selected="true"] {{
    background:#ead9c8 !important;
    color:#2D1E18 !important;
    -webkit-text-fill-color:#2D1E18 !important;
}}
/* Make tab labels, selected tab, radio values and dropdown values readable in light mode. */
.stApp [data-testid="stTabs"] button {{ color:{DARK} !important; font-weight:700 !important; background:#fffaf3 !important; border-radius:9px 9px 0 0; }}
.stApp [data-testid="stTabs"] button p {{ color:{DARK} !important; font-weight:700 !important; }}
.stApp [data-testid="stTabs"] button[aria-selected="true"] {{ color:#ffffff !important; background:{ESP} !important; border-bottom:3px solid {CARAMEL} !important; }}
.stApp [data-testid="stTabs"] button[aria-selected="true"] p {{ color:#ffffff !important; }}
.stApp [data-baseweb="select"] input, .stApp [data-baseweb="select"] div, .stApp [data-baseweb="popover"] *, .stApp [role="option"] {{ color:{DARK} !important; }}
.stApp [data-testid="stRadio"] label, .stApp [data-testid="stCheckbox"] label, .stApp [data-testid="stMultiSelect"] label {{ color:{DARK} !important; }}
.stApp [data-testid="stRadio"] label p, .stApp [data-testid="stCheckbox"] label p {{ color:{DARK} !important; }}
hr {{ border-color:#ead9c8; }}
@media(max-width:700px) {{ h1 {{ font-size:1.8rem !important; }} .block-container {{ padding-left:1rem; padding-right:1rem; }} }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the weather-aware models...")
def load_all_weather_v2(model_version="weather-category-v2"):
    # Versioned cache key forces a fresh model after changing temperature to categorical weather.
    raw = core.build_dataset()
    os.makedirs("data", exist_ok=True)
    raw.to_csv("data/cafe_reservations.csv", index=False)
    return raw, core.run_pipeline(raw)


try:
    RAW, R = load_all_weather_v2("weather-category-v2")
except Exception as e:                                   # graceful failure
    st.error(f"Could not load the project: {e}")
    st.stop()
DF = R["df"]


def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, h=360):
    # Extra space + automargins keep axis titles and tick labels visible in Streamlit.
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", font_color=ESP,
                      margin=dict(l=78, r=28, t=68, b=78), colorway=COLORS,
                      title=dict(x=0.02, xanchor="left", font=dict(size=17, color=ESP)),
                      legend=dict(font=dict(size=12)))
    fig.update_xaxes(automargin=True, title_font=dict(size=13, color=ESP),
                     tickfont=dict(size=11, color=ESP), showline=True, linecolor="#d9c8b8")
    fig.update_yaxes(automargin=True, title_font=dict(size=13, color=ESP),
                     tickfont=dict(size=11, color=ESP), showline=True, linecolor="#d9c8b8", gridcolor="#eee5dc")
    return fig


def header(title, sub):
    st.markdown(f"# {title}")
    st.caption(sub)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ insights and pages
def action_card(tag, title, body):
    st.markdown(f'<div class="action-card"><span class="pill">{tag}</span><div class="action-title">{title}</div><div class="action-body">{body}</div></div>', unsafe_allow_html=True)


def next_steps():
    header("What to Do Next ✨", "A practical, optimistic checklist to turn café data into better decisions.")
    st.info("This project currently uses synthetic demo data. These are ideas to test—not verified findings about a real café.")
    hours = DF.groupby("hour").size().sort_values(ascending=False)
    peak_hour = int(hours.index[0]) if len(hours) else 19
    no_show_rate = float(DF["no_show"].mean()) if len(DF) else 0.0
    repeat_visits = float(DF["past_visits"].mean()) if len(DF) else 0.0
    avg_party = float(DF["party_size"].mean()) if len(DF) else 2.0
    avg_spend = float(DF["spend"].mean()) if len(DF) else 0.0
    a, b = st.columns(2)
    with a:
        action_card("PLAN AHEAD", "Prepare for busy periods", f"The demo data has the most bookings around {peak_hour}:00. Once real booking history is connected, use busy periods to plan staff coverage and table turnover.")
        action_card("FRIENDLY FOLLOW-UP", "Try a gentle reminder routine", f"The sample no-show rate is {no_show_rate:.1%}. Test a polite confirmation or reminder, then compare attendance before and after. Never cancel automatically from a model score.")
        action_card("BETTER TABLE FIT", "Match tables to group size", f"The average party size in this sample is {avg_party:.1f}. Track real party sizes to improve seating plans and avoid awkward gaps between tables.")
    with b:
        action_card("BUILD LOYALTY", "Thank returning guests", f"The sample averages {repeat_visits:.1f} past visits per record. With real customer history, consider a thank-you offer for repeat visitors without assuming every guest wants a promotion.")
        action_card("TEST & LEARN", "Experiment with one small improvement", f"Average spend in the sample is ₹{avg_spend:,.0f}. Test one offer or service change at a time and measure booking conversion, spend, and guest feedback before scaling it.")
        action_card("KEEP IMPROVING", "Review model quality regularly", "Compare precision and recall for no-show predictions and track party-size prediction errors. Validate with representative real data before operational use.")
    st.subheader("Your next-step checklist")
    items = ["Replace synthetic records with consented real reservation data", "Check data quality and define what counts as a no-show", "Measure a baseline before changing reminders or staffing", "Review model metrics and errors regularly", "Collect guest feedback and protect personal information"]
    for i, item in enumerate(items):
        st.checkbox(item, key=f"next_step_{i}")
    st.caption("This checklist is a planning aid. Checking an item does not modify data or contact customers.")


def dashboard():
    header("A brighter view of your café ☕", "Small insights today can lead to happier guests tomorrow.")
    hero_left, hero_right = st.columns([1.5, 1])
    with hero_left:
        st.markdown("### Welcome to Cafe Insight")
        st.write("Explore booking patterns, estimate party size, review possible no-show risk, and understand customer groups—all in one place.")
        st.caption("Illustrative café image · demo analytics only")
    with hero_right:
        st.image("https://images.unsplash.com/photo-1554118811-1e0d58224f24?auto=format&fit=crop&w=1000&q=80", use_container_width=True)
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
    a.plotly_chart(style(px.bar(hours, x="hour", y="bookings", title="Bookings by hour",
                                labels={"hour": "Booking hour (24-hour clock)", "bookings": "Number of bookings"})), use_container_width=True)
    b.plotly_chart(style(px.histogram(DF, x="party_size", title="Party size distribution",
                                     labels={"party_size": "People in the party", "count": "Number of bookings"})), use_container_width=True)
    a, b = st.columns(2)
    ns = DF.no_show.map({0: "Attended", 1: "No-show"}).value_counts().reset_index()
    ns.columns = ["status", "count"]
    a.plotly_chart(style(px.pie(ns, names="status", values="count", hole=.55, title="Attended vs no-show",
                                labels={"status": "Booking outcome", "count": "Number of bookings"})), use_container_width=True)
    b.plotly_chart(style(px.histogram(DF, x="spend", nbins=30, title="Spending distribution",
                                     labels={"spend": "Spend per booking (₹)", "count": "Number of bookings"})), use_container_width=True)
    st.subheader("Recent bookings")
    st.dataframe(DF.tail(8), use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("✨ What to Do Next")
    st.caption("A couple of positive, practical ideas to start with.")
    hours_sorted = DF.groupby("hour").size().sort_values(ascending=False)
    peak_hour = int(hours_sorted.index[0]) if len(hours_sorted) else 19
    action_card("QUICK WIN", "Plan for peak periods", f"Explore {peak_hour}:00 as a possible busy period. Confirm the pattern with real bookings before changing staff schedules.")
    action_card("GUEST EXPERIENCE", "Make confirmations easy", "Test a friendly reminder and a simple way for guests to confirm or update a booking. Compare actual attendance before and after.")
    if st.button("Explore all next steps →"):
        st.session_state["selected_page"] = "✨  What to Do Next"
        st.rerun()


def reservation():
    header("Reservation Prediction", "Estimate party size using weather conditions and booking details.")
    which = st.radio("Prediction model", ["Weather-only LR", "Weather + booking details LR"], horizontal=True)
    with st.form("reg"):
        c = st.columns(5)
        weather = c[0].selectbox("Weather", core.WEATHER_OPTIONS, index=0,
                                 help="Choose the expected weather for the reservation day.")
        hour = c[1].number_input("Booking hour", 9, 21, 19)
        adv = c[2].number_input("Advance booking (days)", 0, 14, 3)
        past = c[3].number_input("Customer's past visits", 0, 19, 5)
        spend = c[4].number_input("Expected spend (₹)", 100.0, 1500.0, 600.0)
        go_ = st.form_submit_button("Predict Party Size")
    if go_:
        try:
            weather_row = pd.DataFrame([{f"weather_{w}": float(w == weather) for w in core.WEATHER_OPTIONS}])
            if which == "Weather-only LR":
                pred = R["simple"].predict(weather_row[core.WEATHER_FEATURES])[0]
            else:
                row = weather_row.copy()
                row["hour"] = hour
                row["advance_days"] = adv
                row["past_visits"] = past
                row["spend"] = spend
                row = row[core.REG_FEATURES]
                pred = R["multi"].predict(row)[0]
            size = int(np.clip(round(pred), 1, 8))
            c1, c2 = st.columns(2)
            with c1: card("Estimated party size", f"{size} people", f"Selected weather: {weather}")
            with c2: card("Model used", which, "trained on 80% of the synthetic dataset")
            note("This is an educational estimate, not a guarantee. The dataset is synthetic, so validate predictions with real booking data before using them for business decisions.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.subheader("Test-set metrics (computed from your model)")
    st.caption("Weather-aware model version: weather-category-v2. If you recently deployed changes, reboot Streamlit once to clear any old in-memory models.")
    st.dataframe(pd.DataFrame(R["reg_scores"]).T.round(3), use_container_width=True)
    n = min(60, len(R["reg_y_test"]))
    fig = go.Figure()
    fig.add_scatter(y=R["reg_y_test"][:n], name="Actual party size", line_color=ESP)
    fig.add_scatter(y=R["reg_pred_simple"][:n], name="Weather-only LR", line=dict(color=SAGE, dash="dash"))
    fig.add_scatter(y=R["reg_pred_multi"][:n], name="Weather + booking details LR", line=dict(color=CARAMEL, dash="dash"))
    fig.update_layout(title="Actual vs predicted party size (first 60 test rows)",
                      xaxis_title="Test booking number", yaxis_title="Party size (people)")
    st.plotly_chart(style(fig), use_container_width=True)


def noshow():
    header("No-Show Prediction", "Use your classification models to flag bookings that may not turn up.")
    name = st.selectbox("Model", list(R["clf_models"]))
    thr = st.slider("Probability threshold", 0.05, 0.95, 0.50, 0.05)
    with st.form("clf"):
        c = st.columns(6)
        hour = c[0].number_input("Hour", 9, 21, 19)
        temp = c[1].number_input("Temperature (°C)", 10.0, 50.0, 28.0)
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
    st.markdown("**Choose a clustering algorithm**")
    st.caption("K-Means assigns each record to one cluster. GMM estimates the probability that a record belongs to each cluster.")
    cust = R["cust"]
    algo = st.radio("Clustering algorithm", ["K-Means", "GMM"], horizontal=True, label_visibility="collapsed")
    col = "cluster" if algo == "K-Means" else "gmm_cluster"
    a, b = st.columns(2)
    a.plotly_chart(style(px.line(x=list(range(1, 9)), y=R["wcss"], markers=True, title="Elbow method",
                                 labels={"x": "Number of clusters (K)", "y": "Within-cluster sum of squares (WCSS)"})), use_container_width=True)
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

    st.divider()
    st.subheader("What to Do Next")
    st.caption("Use these actions as ideas to test once real, representative café data is available.")
    action_card("1 · INTERPRET", "Give clusters meaningful names", "Review average spend, past visits, and advance booking days. Use neutral labels such as frequent visitors or occasional visitors only when the numbers support them.")
    action_card("2 · VALIDATE", "Check the groups against real bookings", "Compare the patterns with actual customer history and staff feedback. The current records are synthetic, so cluster profiles are examples—not confirmed customer behaviour.")
    action_card("3 · PERSONALIZE CAREFULLY", "Choose a useful service improvement", "Consider practical changes such as easier repeat booking or a suitable table layout. Avoid sensitive assumptions and do not treat cluster membership as a fixed customer identity.")
    todo = ["Compare cluster averages before naming groups", "Test whether groups are stable on new data", "Review privacy and consent before using customer history", "Measure whether any service change improves guest experience"]
    for i, item in enumerate(todo):
        st.checkbox(item, key=f"segmentation_next_{i}")


def performance():
    header("Model Performance", "Every number here is computed by your code on the current dataset.")
    t1, t2, t3, t4 = st.tabs(["Regression", "Classification", "Ensembles", "Validation & tuning"])
    with t1:
        st.dataframe(pd.DataFrame(R["reg_scores"]).T.round(3), use_container_width=True)
    with t2:
        sc = pd.DataFrame(R["clf_scores"]).T
        st.dataframe(sc.round(3), use_container_width=True)
        st.plotly_chart(style(px.bar(sc.reset_index().melt("index"), x="index", y="value", color="variable",
                                     barmode="group", title="Classifier comparison",
                                     labels={"index": "Classification model", "value": "Score", "variable": "Metric"})), use_container_width=True)
        m = st.selectbox("Confusion matrix for", list(R["confusion"]))
        st.plotly_chart(style(px.imshow(R["confusion"][m], text_auto=True, x=["Attended", "No-show"], y=["Attended", "No-show"],
                                        color_continuous_scale=["#FAF6EF", CARAMEL], labels=dict(x="Predicted", y="Actual")), 320),
                        use_container_width=True)
    with t3:
        bag = R["bagging"]
        st.plotly_chart(style(px.line(x=list(bag), y=list(bag.values()), markers=True, title="Bagging: trees vs accuracy",
                                      labels={"x": "Number of decision trees (estimators)", "y": "Accuracy score"})), use_container_width=True)
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
                                     labels={"value": "Feature importance score", "index": "Input feature"})), use_container_width=True)
        note(f"Top 3 features: {', '.join(reversed(list(imp.index[-3:])))}")


def explorer():
    header("Dataset Explorer", "Synthetic café bookings generated by your code (raw, before missing values are filled).")
    c = st.columns(3)
    with c[0]: card("Rows", f"{RAW.shape[0]:,}")
    with c[1]: card("Columns", f"{RAW.shape[1]}")
    with c[2]: card("Missing values", f"{int(RAW.isna().sum().sum())}", "temperature column")
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
    header("About Cafe Insight", "Smart Reservations. Happier Customers. Better Decisions.")
    st.write("**Aim:** predict party size and no-shows and group customers for a café, using all seven lab practicals.")
    st.markdown("""
**Practicals:** 1 Missing values and EDA · 2 Simple and multiple linear regression · 3 Logistic, KNN and Naive Bayes ·
4 K-Means and elbow · 5 GMM and PCA · 6 Bagging and Random Forest · 7 Cross-validation, GridSearch and RandomizedSearch.

**Workflow:** Dataset → Preprocessing → Model training → Evaluation → Predictions → Café insights

**Technologies:** Python, pandas, NumPy, scikit-learn, Plotly, Streamlit

**Limitations:** the dataset is synthetic and random, so scores demonstrate the methods only. The scaler is fitted on all rows
before the train/test split, as in the original script; this is a simplification to mention in your viva.
""")


# Defensive fallback: prevents a NameError if this page is reused in a trimmed deployment.
if "next_steps" not in globals():
    def next_steps():
        header("What to Do Next ✨", "Practical steps for improving café decisions.")
        st.info("The current dataset is synthetic. Validate these ideas with real, consented booking data.")
        for title, body in [
            ("Validate the data", "Replace demo records with real reservation data and check missing values."),
            ("Review busy periods", "Use real booking history to confirm peak hours before changing staffing."),
            ("Test reminders", "Compare no-show rates before and after a friendly reminder."),
            ("Check model quality", "Review precision, recall, and prediction errors regularly."),
            ("Protect customer privacy", "Use only necessary data and follow consent and privacy requirements."),
        ]:
            action_card("NEXT STEP", title, body)

PAGES = {"📊  Dashboard": dashboard, "✨  What to Do Next": next_steps,
         "🪑  Reservation Prediction": reservation, "🚫  No-Show Prediction": noshow,
         "👥  Customer Segmentation": segmentation, "📈  Model Performance": performance,
         "🗂️  Dataset Explorer": explorer, "ℹ️  About Project": about}
st.sidebar.markdown("## ☕ Cafe Insight")
st.sidebar.caption("Smart Reservations · Happier Customers")
st.sidebar.markdown("---")
page_names = list(PAGES)
default_page = st.session_state.pop("selected_page", page_names[0])
selected = st.sidebar.radio("Navigate", page_names, index=page_names.index(default_page) if default_page in page_names else 0,
                            label_visibility="collapsed")
PAGES[selected]()
