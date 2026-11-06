import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

RANDOM_STATE = 42
N_FOLDS = 5
DATA_DIR = "Clean data"

# ---------------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------------

def load_raw_tables():
    brandhealth = pd.read_csv(f"{DATA_DIR}/clean_brand_health.csv")
    companion = pd.read_csv(f"{DATA_DIR}/clean_companion.csv")
    dayofweek = pd.read_csv(f"{DATA_DIR}/clean_day_of_week.csv")
    daypart = pd.read_csv(f"{DATA_DIR}/clean_day_part.csv")
    needstate = pd.read_csv(f"{DATA_DIR}/clean_need_state.csv")
    savar = pd.read_csv(f"{DATA_DIR}/clean_sa_var.csv")
    coffeevisit = pd.read_csv(f"{DATA_DIR}/clean_coffee_visit.csv")
    return brandhealth, companion, dayofweek, daypart, needstate, savar, coffeevisit


# ---------------------------------------------------------------------------
# 2. Aggregate event-level logs to ONE ROW PER CUSTOMER
# ---------------------------------------------------------------------------

def aggregate_companion(companion: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot: which companion types has this customer ever logged?"""
    dummies = pd.get_dummies(
        companion[["ID", "Companion#group"]], columns=["Companion#group"],
        prefix="companion",
    )
    return dummies.groupby("ID").max().reset_index()


def aggregate_dayofweek(dayofweek: pd.DataFrame) -> pd.DataFrame:
    """Sum visit counts per weekday -> wide columns, plus weekday/weekend totals."""
    dow_pivot = dayofweek.pivot_table(
        index="ID", columns="Dayofweek", values="Visit#Dayofweek",
        aggfunc="sum", fill_value=0,
    )
    dow_pivot.columns = [f"dow_visits_{c}" for c in dow_pivot.columns]
    dow_pivot = dow_pivot.reset_index()

    weekend_pivot = dayofweek.pivot_table(
        index="ID", columns="Weekday#end", values="Visit#Dayofweek",
        aggfunc="sum", fill_value=0,
    )
    weekend_pivot.columns = [f"visits_{c}" for c in weekend_pivot.columns]
    weekend_pivot = weekend_pivot.reset_index()

    return dow_pivot.merge(weekend_pivot, on="ID", how="outer").fillna(0)


def aggregate_daypart(daypart: pd.DataFrame) -> pd.DataFrame:
    """Sum visit counts per daypart -> wide columns."""
    pivot = daypart.pivot_table(
        index="ID", columns="Daypart", values="Visit#Daypart",
        aggfunc="sum", fill_value=0,
    )
    pivot.columns = [f"daypart_visits_{c}" for c in pivot.columns]
    return pivot.reset_index()


def aggregate_needstate(needstate: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot on the coarser NeedstateGroup (6 categories, less sparse
    than the 15-category Needstates), plus an occasion-diversity count."""
    dummies = pd.get_dummies(
        needstate[["ID", "NeedstateGroup"]], columns=["NeedstateGroup"],
        prefix="need",
    )
    agg = dummies.groupby("ID").max().reset_index()

    diversity = (
        needstate.groupby("ID")["Needstates"].nunique()
        .reset_index().rename(columns={"Needstates": "needstate_diversity"})
    )
    return agg.merge(diversity, on="ID", how="left")


def load_and_merge_data():
    (brandhealth, companion, dayofweek, daypart,
     needstate, savar, coffeevisit) = load_raw_tables()

    companion_agg = aggregate_companion(companion)
    dow_agg = aggregate_dayofweek(dayofweek)
    daypart_agg = aggregate_daypart(daypart)
    needstate_agg = aggregate_needstate(needstate)

    print(f"brand_health (base, customer x brand): {brandhealth.shape}")
    print(f"companion_agg (per customer):          {companion_agg.shape}")
    print(f"dow_agg (per customer):                 {dow_agg.shape}")
    print(f"daypart_agg (per customer):              {daypart_agg.shape}")
    print(f"needstate_agg (per customer):            {needstate_agg.shape}")

    # base = brand_health (one row per customer-brand pair). Every merge
    # below is many-to-one (many brand rows -> one customer summary row).
    # validate="m:1" makes pandas RAISE instead of silently exploding rows
    # if that assumption is ever violated again.
    df = brandhealth.merge(savar, on="ID", how="left", validate="m:1",
                            suffixes=("", "_savar"))
    df = df.merge(coffeevisit, on=["ID", "City"], how="left", validate="m:1",
                  suffixes=("", "_cv"))
    df = df.merge(companion_agg, on="ID", how="left", validate="m:1")
    df = df.merge(dow_agg, on="ID", how="left", validate="m:1")
    df = df.merge(daypart_agg, on="ID", how="left", validate="m:1")
    df = df.merge(needstate_agg, on="ID", how="left", validate="m:1")

    print(f"\nFinal merged shape: {df.shape} "
          f"(expected ~{brandhealth.shape[0]} rows, NOT millions)")
    assert df.shape[0] <= brandhealth.shape[0] * 1.01, (
        "Row count grew past the brand_health base — a merge is still "
        "fanning out. Check merge keys before continuing."
    )
    return df


def create_churn_target(df):
    df = df.copy()
    df["Churn"] = ((df["P3M"] == 1) & (df["P1M"] == 0)).astype(int)
    return df


def filter_to_existing_customers(df):
    """Restrict modeling population to customers who WERE active in P3M.
    Without this, 'never tried the brand' rows (P3M==0, trivially
    Churn=0) dominate the negative class and let the model shortcut to
    'does this look like a customer at all' instead of learning real
    stay-vs-leave drivers. For a marketing/retention use case this
    filter isn't optional — it's the difference between finding real
    churn drivers and finding 'never heard of us' drivers.
    """
    before = len(df)
    df = df[df["P3M"] == 1].copy()
    print(f"Filtered to P3M==1 (existing customers): {before} -> {len(df)} rows "
          f"({len(df) / before:.1%} kept)")
    return df


# ---------------------------------------------------------------------------
# 3. Feature setup
# ---------------------------------------------------------------------------

# Known brand_health numeric fields + engineered ratios.
# NOTE: 'Spending', 'Spending_use', 'Segmentation', and 'PPA' were REMOVED
# after the leakage diagnostic showed they are only populated for
# customers active in the most recent period (skip-logic in the survey) -
# meaning "missing" on these fields is a near-direct restatement of churn
# itself, not a genuine predictor.
#
# 'Fre#visit', 'Weekly', and 'Daily' were ALSO REMOVED after a second
# diagnostic (single-feature AUC computed WITHIN the P3M==1 population)
# showed Fre#visit alone reaches 0.99 AUC and Weekly reaches 0.77 - both
# almost certainly measure CURRENT visit frequency, i.e. they encode the
# same "still active right now" concept the P1M half of the label is
# built from, not a genuine upstream driver of churn.
BASE_NUMERIC_COLS = [
    "Spontaneous", "Awareness", "Trial", "Comprehension", "Brand_Likability",
    "NPS#P3M",
]
BASE_CATEGORICAL_COLS = [
    "City", "Brand", "NPS#P3M#Group",
    # demographic / savar columns — kept only if present in your savar file
    "Gender", "Occupation", "Occupation#group", "Age#group",
]
OPTIONAL_NUMERIC_COLS = ["Age", "MPI#Mean", "Group_size"]

AGG_PREFIXES = ("companion_", "dow_visits_", "visits_", "daypart_visits_",
                "need_")


def parse_locale_number(value):
    """Parse numbers that may use a comma decimal separator (e.g. '12,5'
    -> 12.5), possibly with '.' as a thousands separator (e.g. '1.234,56'
    -> 1234.56). Falls back to NaN for anything unparseable rather than
    crashing the whole pipeline."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s in ("", "nan", "None", "NaN"):
        return np.nan
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        s = s.replace(".", "").replace(",", ".")
    elif has_comma:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_numeric_locale(df, cols):
    """Coerce object-typed numeric columns (comma-decimal locale) to
    proper floats. Only touches columns that are actually non-numeric,
    so it's a no-op for columns pandas already parsed correctly."""
    df = df.copy()
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            n_before = df[c].notna().sum()
            df[c] = df[c].apply(parse_locale_number)
            n_after = df[c].notna().sum()
            if n_after < n_before:
                print(f"  [locale-clean] '{c}': {n_before - n_after} values "
                      f"could not be parsed as numbers -> set to NaN")
    return df


def engineer_raw_features(df):
    df = df.copy()
    numeric_candidates = BASE_NUMERIC_COLS + OPTIONAL_NUMERIC_COLS
    df = clean_numeric_locale(df, numeric_candidates)
    # NOTE: Usage_Frequency (from 'Fre#visit') and Spend_Per_Visit (from
    # 'Spending') were removed — both derived from leaky current-activity
    # columns (see notes above BASE_NUMERIC_COLS).
    return df


def build_feature_frame(df):
    df = engineer_raw_features(df)

    agg_cols = [c for c in df.columns if c.startswith(AGG_PREFIXES)
                or c == "needstate_diversity"]
    num_cols = [c for c in BASE_NUMERIC_COLS + OPTIONAL_NUMERIC_COLS + agg_cols
                if c in df.columns]
    cat_cols = [c for c in BASE_CATEGORICAL_COLS if c in df.columns]

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    y = df["Churn"]
    groups = df["ID"]
    return X, y, groups, num_cols, cat_cols


# ---------------------------------------------------------------------------
# 3b. Driver analysis — signed, interpretable, for the marketing plan
# ---------------------------------------------------------------------------

def get_encoded_feature_names(fitted_pipeline):
    """Pull the expanded feature names out of a fitted ColumnTransformer
    (numeric columns keep their name; one-hot categorical columns become
    'col_value')."""
    preprocessor = fitted_pipeline.named_steps["prep"]
    return list(preprocessor.get_feature_names_out())


def print_churn_drivers(lr_pipeline, rf_pipeline, top_n=15):
    feature_names = get_encoded_feature_names(lr_pipeline)

    lr_coefs = lr_pipeline.named_steps["clf"].coef_[0]
    lr_df = pd.DataFrame({"feature": feature_names, "lr_coefficient": lr_coefs})
    lr_df["direction"] = np.where(
        lr_df["lr_coefficient"] > 0, "-> INCREASES churn risk",
        "-> DECREASES churn risk",
    )
    lr_df["abs_coef"] = lr_df["lr_coefficient"].abs()
    lr_top = lr_df.sort_values("abs_coef", ascending=False).head(top_n)

    print("\n=== Top churn drivers — Logistic Regression (signed, for action) ===")
    print(lr_top[["feature", "lr_coefficient", "direction"]].to_string(index=False))

    rf_importances = rf_pipeline.named_steps["clf"].feature_importances_
    rf_df = pd.DataFrame({"feature": feature_names, "rf_importance": rf_importances})
    rf_top = rf_df.sort_values("rf_importance", ascending=False).head(top_n)

    print("\n=== Top churn drivers — Random Forest importance (magnitude cross-check) ===")
    print(rf_top.to_string(index=False))

    print(
        "\nRead this as: LR tells you WHICH WAY a factor pushes churn risk "
        "(e.g. low Brand_Likability -> higher churn -> worth a satisfaction "
        "campaign); RF confirms magnitude and can surface non-linear effects "
        "LR might miss. Features that show up strongly in BOTH are the "
        "safest bets to anchor a retention/marketing plan on."
    )


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

def main():
    df = load_and_merge_data()
    df = create_churn_target(df)
    df = filter_to_existing_customers(df)
    X, y, groups, num_cols, cat_cols = build_feature_frame(df)

    # ---- 60/20/20 split, GROUPED by customer ID ----
    gss_outer = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss_outer.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]

    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss_inner.split(X_temp, y_temp, groups=groups_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    # sanity check: no customer ID should appear in more than one split
    train_ids = set(groups_train)
    val_ids = set(groups_temp.iloc[val_idx])
    test_ids = set(groups_temp.iloc[test_idx])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), \
        "Customer ID leakage across splits!"

    print(f"\nTrain: {X_train.shape} ({len(train_ids)} customers) | "
          f"Val: {X_val.shape} ({len(val_ids)} customers) | "
          f"Test: {X_test.shape} ({len(test_ids)} customers)")
    print(f"Churn rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, "
          f"test: {y_test.mean():.3f}")

    # ---- preprocessing, fit only on the training fold each time ----
    numeric_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    models = {
        "Logistic_Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random_Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=RANDOM_STATE,
                n_jobs=4)),
        ]),
        "Small_Feedforward_NN": Pipeline([
            ("prep", preprocessor),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=500,
                early_stopping=True, n_iter_no_change=15,
                random_state=RANDOM_STATE)),
        ]),
    }

    # ---- 5-fold GROUPED, stratified CV on the training set ----
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, pipe in models.items():
        fold_aucs = []
        for fold_i, (fit_idx, hold_idx) in enumerate(
                cv.split(X_train, y_train, groups=groups_train), start=1):
            pipe.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            proba = pipe.predict_proba(X_train.iloc[hold_idx])[:, 1]
            fold_auc = roc_auc_score(y_train.iloc[hold_idx], proba)
            fold_aucs.append(fold_auc)
            print(f"  {name} fold {fold_i}/{N_FOLDS}: AUC = {fold_auc:.4f}")
        cv_results[name] = {
            "cv_auc_mean": float(np.mean(fold_aucs)),
            "cv_auc_std": float(np.std(fold_aucs)),
        }
        print(f"{name}: CV AUC = {cv_results[name]['cv_auc_mean']:.4f} "
              f"+/- {cv_results[name]['cv_auc_std']:.4f}")

    # ---- refit each model on full train, evaluate ONCE on held-out test ----
    test_results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test_results[name] = {
            "test_auc": roc_auc_score(y_test, proba),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "proba": proba,
        }
        print(f"{name}: TEST AUC = {test_results[name]['test_auc']:.4f}")

    # ---- embedding-based NN (Keras), optional dependency ----
    try:
        emb_result = train_embedding_model(
            X_train, y_train, X_val, y_val, X_test, y_test, num_cols, cat_cols
        )
        test_results["Embedding_NN"] = emb_result
        cv_results["Embedding_NN"] = {"cv_auc_mean": np.nan, "cv_auc_std": np.nan}
        print(f"Embedding_NN: test AUC = {emb_result['test_auc']:.4f} "
              "(trained with early stopping on val set)")
    except ImportError:
        print("\n[!] TensorFlow not installed — skipping embedding-based NN. "
              "Install with `pip install tensorflow-macos tensorflow-metal` "
              "(Apple Silicon) to include it.")

    # ---- comparison table ----
    rows = []
    for name in test_results:
        rows.append({
            "Model": name,
            "CV_AUC_mean": cv_results.get(name, {}).get("cv_auc_mean", np.nan),
            "CV_AUC_std": cv_results.get(name, {}).get("cv_auc_std", np.nan),
            "Test_AUC": test_results[name]["test_auc"],
            "Test_Precision": test_results[name]["test_precision"],
            "Test_Recall": test_results[name]["test_recall"],
            "Test_F1": test_results[name]["test_f1"],
        })
    comparison_table = pd.DataFrame(rows).sort_values("Test_AUC", ascending=False)
    print("\n=== Model Comparison Table ===")
    print(comparison_table.to_string(index=False))

    best_model_name = comparison_table.iloc[0]["Model"]
    print(f"\nBest model on held-out test set: {best_model_name} "
          f"(AUC = {comparison_table.iloc[0]['Test_AUC']:.4f})")

    # ---- ROC curves on test set ----
    plt.figure(figsize=(10, 7))
    for name, res in test_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Held-out Test Set (grouped by customer)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_test_set.png", dpi=150)
    plt.show()

    # ---- driver analysis for the marketing plan ----
    # Signed logistic regression coefficients tell you DIRECTION (does
    # this factor push someone toward or away from churning), which is
    # what a marketing/retention plan actually needs to act on. Random
    # Forest importance is shown alongside as a magnitude cross-check
    # (it can catch non-linear drivers LR might underweight, but it
    # doesn't tell you direction on its own).
    print_churn_drivers(models["Logistic_Regression"], models["Random_Forest"])

    return comparison_table, test_results


# ---------------------------------------------------------------------------
# 5. Embedding-based NN for categorical features (Keras)
# ---------------------------------------------------------------------------

def train_embedding_model(X_train, y_train, X_val, y_val, X_test, y_test,
                           num_cols, cat_cols):
    import tensorflow as tf
    from keras import layers, Model

    cat_vocab = {c: {v: i + 1 for i, v in enumerate(X_train[c].unique())}
                 for c in cat_cols}  # 0 reserved for unseen/unknown

    def encode_cats(X):
        return {c: X[c].map(cat_vocab[c]).fillna(0).astype(int).values
                for c in cat_cols}

    imputer = SimpleImputer(strategy="median").fit(X_train[num_cols])
    scaler = StandardScaler().fit(imputer.transform(X_train[num_cols]))
    num_train = scaler.transform(imputer.transform(X_train[num_cols]))
    num_val = scaler.transform(imputer.transform(X_val[num_cols]))
    num_test = scaler.transform(imputer.transform(X_test[num_cols]))

    cat_train = encode_cats(X_train)
    cat_val = encode_cats(X_val)
    cat_test = encode_cats(X_test)

    inputs, embeds = [], []
    for c in cat_cols:
        vocab_size = len(cat_vocab[c]) + 1
        emb_dim = min(50, (vocab_size + 1) // 2) or 1
        inp = layers.Input(shape=(1,), name=f"cat_{c}")
        emb = layers.Embedding(vocab_size, emb_dim)(inp)
        emb = layers.Flatten()(emb)
        inputs.append(inp)
        embeds.append(emb)

    num_input = layers.Input(shape=(len(num_cols),), name="numeric")
    inputs.append(num_input)

    x = layers.Concatenate()(embeds + [num_input]) if embeds else num_input
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])

    def to_input_list(cat_dict, num_arr):
        return [cat_dict[c] for c in cat_cols] + [num_arr]

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=10, restore_best_weights=True
    )

    model.fit(
        to_input_list(cat_train, num_train), y_train.values,
        validation_data=(to_input_list(cat_val, num_val), y_val.values),
        epochs=100, batch_size=64, callbacks=[early_stop], verbose=1,
    )

    proba = model.predict(to_input_list(cat_test, num_test), verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)

    return {
        "test_auc": roc_auc_score(y_test, proba),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_f1": f1_score(y_test, pred, zero_division=0),
        "proba": proba,
    }


if __name__ == "__main__":
    comparison_table, test_results = main()