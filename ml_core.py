"""ml_core.py - the SAME logic as cafereserve_simple.py (same seed, data, models, settings),
wrapped in functions so Streamlit can call it. Nothing here is invented or hard-coded."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

WEATHER_OPTIONS = ["Sunny", "Rainy", "Cloudy", "Windy"]
WEATHER_FEATURES = [f"weather_{w}" for w in WEATHER_OPTIONS]
FEATURES = ["hour", "temp", "party_size", "advance_days", "past_visits", "spend"]   # no-show classifier features
REG_FEATURES = WEATHER_FEATURES + ["hour", "advance_days", "past_visits", "spend"]  # reservation regression features


def build_dataset(n=1000):
    """Step 0 of the original script: synthetic cafe data (8 columns, 31 missing temp values)."""
    np.random.seed(42)
    df = pd.DataFrame({
        "customer_id": np.random.randint(1, 101, n),
        "hour": np.random.randint(9, 22, n),
        "temp": np.random.normal(28, 5, n).round(1),
        "weather": np.random.choice(WEATHER_OPTIONS, n, p=[0.35, 0.20, 0.30, 0.15]),
        "party_size": np.random.randint(1, 9, n),
        "advance_days": np.random.randint(0, 15, n),
        "past_visits": np.random.randint(0, 20, n),
        "spend": np.random.normal(600, 150, n).round(0),
    })
    df["no_show"] = ((df.advance_days * 0.05 - df.past_visits * 0.02 + np.random.rand(n)) > 0.7).astype(int)
    df.loc[:30, "temp"] = np.nan
    return df


def _reg_scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"MAE": mean_absolute_error(y_true, pred), "MSE": mse, "RMSE": mse ** 0.5, "R2": r2_score(y_true, pred)}


def run_pipeline(raw):
    """Practicals 1-7, exactly as in the original script. Returns every fitted model and result."""
    out = {}
    df = raw.copy()
    df["temp"] = df["temp"].fillna(df["temp"].median())                      # Practical 1
    out["df"] = df

    # Practical 2: linear regression using one-hot encoded weather categories.
    # Weather-only model uses weather_* columns; multi-feature model adds booking details.
    y = df["party_size"]
    weather_encoded = pd.get_dummies(df["weather"], prefix="weather")
    weather_encoded = weather_encoded.reindex(columns=WEATHER_FEATURES, fill_value=0).astype(float)
    X_weather = weather_encoded
    X_multi = weather_encoded.copy()
    for col in ["hour", "advance_days", "past_visits", "spend"]:
        X_multi[col] = df[col].values
    Xtr, Xte, ytr, yte = train_test_split(X_weather, y, test_size=0.2, random_state=42)
    simple = LinearRegression().fit(Xtr, ytr)
    X2tr, X2te, y2tr, y2te = train_test_split(X_multi[REG_FEATURES], y, test_size=0.2, random_state=42)
    multi = LinearRegression().fit(X2tr, y2tr)
    p1, p2 = simple.predict(Xte), multi.predict(X2te)
    out.update(simple=simple, multi=multi, reg_y_test=yte.values, reg_pred_simple=p1, reg_pred_multi=p2,
               reg_scores={"Weather-only LR": _reg_scores(yte, p1), "Weather + booking details LR": _reg_scores(y2te, p2)})

    # Practical 3: classification
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    t = df["no_show"]
    Xa, Xb, ya, yb = train_test_split(X, t, test_size=0.25, random_state=42)
    models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Naive Bayes": GaussianNB()}
    scores, preds = {}, {}
    for name, m in models.items():
        pred = m.fit(Xa, ya).predict(Xb)
        preds[name] = pred
        scores[name] = {"Accuracy": accuracy_score(yb, pred), "Precision": precision_score(yb, pred, zero_division=0),
                        "Recall": recall_score(yb, pred), "F1": f1_score(yb, pred)}
    out.update(scaler=scaler, clf_models=models, clf_scores=scores, clf_preds=preds, clf_y_test=yb.values,
               confusion={k: confusion_matrix(yb, v) for k, v in preds.items()})

    # Practical 4: K-Means
    cust = df.groupby("customer_id")[["spend", "past_visits", "advance_days"]].mean()
    Xc = StandardScaler().fit_transform(cust)
    out["wcss"] = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc).inertia_ for k in range(1, 9)]
    cust["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Xc)

    # Practical 5: GMM + PCA
    gmm = GaussianMixture(n_components=3, random_state=42).fit(Xc)
    cust["gmm_cluster"] = gmm.predict(Xc)
    proba = gmm.predict_proba(Xc)
    pts = PCA(n_components=2).fit_transform(Xc)
    cust["PC1"], cust["PC2"] = pts[:, 0], pts[:, 1]
    for i in range(3):
        cust[f"prob_{i}"] = proba[:, i]
    out["cust"] = cust

    # Practical 6: bagging and random forest
    out["bagging"] = {tr: BaggingClassifier(DecisionTreeClassifier(), n_estimators=tr, random_state=42)
                      .fit(Xa, ya).score(Xb, yb) for tr in [5, 10, 50, 100]}
    dt = DecisionTreeClassifier(random_state=42).fit(Xa, ya)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42).fit(Xa, ya)
    out["tree_vs_forest"] = {"Decision Tree": dt.score(Xb, yb), "Random Forest (balanced)": rf.score(Xb, yb)}

    # Practical 7: cross-validation + tuning
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(RandomForestClassifier(random_state=42), Xa, ya, cv=skf)
    params = {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5).fit(Xa, ya)
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), params, n_iter=4, cv=5, random_state=42).fit(Xa, ya)
    imp = pd.Series(grid.best_estimator_.feature_importances_, index=FEATURES).sort_values()
    out.update(cv_scores=cv, grid_best=(grid.best_params_, grid.best_score_),
               rand_best=(rand.best_params_, rand.best_score_), importance=imp)
    return out
