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

