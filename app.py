import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paris Agreement Tracker",
    page_icon="🌍",
    layout="wide"
)

# ── Feature label mapping (human-readable) ─────────────────────────────────────
FEATURE_LABELS = {
    "ghg_2024":                  "GHG Emissions 2024 (Mt CO₂e)",
    "ghg_per_capita_2024":       "GHG per Capita 2024 (t CO₂e)",
    "pct_change_2015_2024":      "% Change, 2015–2024",
    "slope_post_paris":          "Emissions Slope Post-Paris (Mt/yr)",
    "slope_pc_post_paris":       "Per Capita Slope Post-Paris (t/yr)",
    "pct_change_pre_post_paris": "% Change, Pre vs. Post-Paris",
    "pct_from_peak":             "% Change from Peak Emissions",
    "reduction_ratio":           "Reduction Ratio (post-2015)",
}

def label(f):
    return FEATURE_LABELS.get(f, f)

def fmt(v):
    """Format a number with commas and 2 decimal places."""
    return f"{v:,.2f}"

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model.sav', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.sav', 'rb') as f:
        scaler = pickle.load(f)
    with open('features_list.sav', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv('country_features.csv')

model, scaler, FEATURES = load_artifacts()
df = load_data()

# ── Rating helpers ─────────────────────────────────────────────────────────────
RATING_COLORS = {
    'Critically Insufficient': '#d62828',
    'Highly Insufficient':     '#f77f00',
    'Insufficient':            '#fcbf49',
    'Almost Sufficient':       '#06d6a0',
}
RATING_EMOJI = {
    'Critically Insufficient': '🔴',
    'Highly Insufficient':     '🟠',
    'Insufficient':            '🟡',
    'Almost Sufficient':       '🟢',
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌍 Paris Agreement Tracker")
st.markdown(
    "**Predicting whether countries are on track to meet the Paris Agreement** "
    "using historical GHG emission trajectories (1950–2024)."
)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Explore a Country")
countries = sorted(df['country'].unique())
selected = st.sidebar.selectbox(
    "Select a Country", countries,
    index=countries.index("Norway") if "Norway" in countries else 0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Random Forest (tuned)")
st.sidebar.markdown("**Data:** GHG Emissions 1950–2024 + CAT Ratings")
st.sidebar.markdown("**Source:** [Climate Action Tracker](https://climateactiontracker.org)")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Global Overview", "🔬 Country Deep-Dive", "📊 Model Insights"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: Global Map
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Global Climate Action Ratings")

    X_all = df[FEATURES].fillna(df[FEATURES].median())
    X_all_sc = scaler.transform(X_all)
    df['predicted_on_track'] = model.predict(X_all_sc)
    df['predicted_label'] = df['predicted_on_track'].map(
        {1: 'Almost Sufficient (On Track)', 0: 'Not on Track'}
    )

    ghg_iso = pd.read_csv('data/ghg_emissions.csv')[['country', 'iso_code']].drop_duplicates()
    df_map = df.merge(ghg_iso, on='country', how='left')

    rating_order = ['Critically Insufficient', 'Highly Insufficient', 'Insufficient', 'Almost Sufficient']
    df_map['cat_rating'] = pd.Categorical(df_map['cat_rating'], categories=rating_order, ordered=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Actual CAT Ratings (from Climate Action Tracker)")
        fig_actual = px.choropleth(
            df_map.dropna(subset=['iso_code', 'cat_rating']),
            locations='iso_code',
            color='cat_rating',
            hover_name='country',
            color_discrete_map=RATING_COLORS,
            category_orders={'cat_rating': rating_order},
            title='Official CAT Ratings (39 countries)'
        )
        fig_actual.update_layout(
            height=380, margin=dict(l=0, r=0, t=40, b=0),
            legend_title='CAT Rating'
        )
        st.plotly_chart(fig_actual, use_container_width=True)

    with col2:
        st.markdown("##### ML-Predicted On-Track Status")
        fig_pred = px.choropleth(
            df_map.dropna(subset=['iso_code']),
            locations='iso_code',
            color='predicted_label',
            hover_name='country',
            color_discrete_map={
                'Almost Sufficient (On Track)': '#06d6a0',
                'Not on Track': '#e63946'
            },
            title='Predicted Paris Agreement Alignment'
        )
        fig_pred.update_layout(
            height=380, margin=dict(l=0, r=0, t=40, b=0),
            legend_title='Prediction'
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # Rating breakdown table — rename cat_rating column
    st.markdown("##### Rating Breakdown")
    summary = df.groupby('cat_rating').agg(
        Countries=('country', lambda x: ', '.join(sorted(x))),
        Count=('country', 'count')
    ).reset_index().rename(columns={'cat_rating': 'CAT Rating'})
    st.dataframe(summary, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: Country Deep-Dive
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    row = df[df['country'] == selected].iloc[0]
    actual_rating = row.get('cat_rating', 'N/A')
    on_track_pred = int(row['predicted_on_track'])
    pred_label = "✅ On Track (Almost Sufficient)" if on_track_pred else "❌ Not on Track"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Country", selected)
    col_b.metric(
        "Actual CAT Rating",
        f"{RATING_EMOJI.get(actual_rating, '❓')} {actual_rating}"
    )
    col_c.metric("ML Prediction", pred_label)

    st.markdown("---")

    # Feature profile bar chart with human-readable labels
    st.subheader("Emission Trend Profile")
    feat_vals = row[FEATURES].to_dict()
    feat_df = pd.DataFrame({
        'Feature': list(feat_vals.keys()),
        'Label':   [label(f) for f in feat_vals.keys()],
        'Value':   list(feat_vals.values())
    })

    on_track_avg  = df[df['on_track'] == 1][FEATURES].mean()
    not_track_avg = df[df['on_track'] == 0][FEATURES].mean()
    feat_df['On-Track Avg']  = [on_track_avg[f]  for f in feat_df['Feature']]
    feat_df['Not-Track Avg'] = [not_track_avg[f] for f in feat_df['Feature']]

    fig_feat, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(feat_df))
    w = 0.28
    ax.bar(x - w, feat_df['Value'],        w, label=selected,       color='#264653', zorder=3)
    ax.bar(x,     feat_df['On-Track Avg'], w, label='On-Track Avg', color='#06d6a0', alpha=0.8, zorder=3)
    ax.bar(x + w, feat_df['Not-Track Avg'],w, label='Not-Track Avg',color='#e63946', alpha=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_df['Label'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title(f"Feature Profile: {selected} vs. Group Averages")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_feat)

    # SHAP waterfall with human-readable feature names
    st.subheader("SHAP Explanation — Why this prediction?")
    idx = df.index[df['country'] == selected][0]
    X_all_arr = scaler.transform(df[FEATURES].fillna(df[FEATURES].median()))
    explainer = shap.TreeExplainer(model)
    shap_exp_all = explainer(X_all_arr)

    if shap_exp_all.values.ndim == 3:
        sv_vals = shap_exp_all.values[:, :, 1]
        sv_base = shap_exp_all.base_values[:, 1]
    else:
        sv_vals = shap_exp_all.values
        sv_base = shap_exp_all.base_values

    readable_names = [label(f) for f in FEATURES]
    sample_exp = shap.Explanation(
        values=sv_vals[idx],
        base_values=float(sv_base[idx]),
        data=shap_exp_all.data[idx],
        feature_names=readable_names
    )
    fig_wf, ax_wf = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(sample_exp, show=False)
    plt.title(f"SHAP Waterfall: {selected}")
    plt.tight_layout()
    st.pyplot(fig_wf)

    st.caption(
        "SHAP values show how each feature pushes the prediction toward 'On Track' (positive) "
        "or 'Not on Track' (negative). The base value is the model's average prediction."
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Insights
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Summary (Stratified 5-Fold CV)")
    perf_df = pd.DataFrame({
        'Model':     ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'F1 Score':  ['See notebook', 'See notebook', 'See notebook'],
        'ROC-AUC':   ['See notebook', '0.933',        'See notebook'],
        'Accuracy':  ['See notebook', '84.6%',         'See notebook'],
        'Notes':     [
            'Baseline — linear, interpretable',
            '★ Selected for deployment — best F1 & AUC',
            'Comparable performance, higher overfitting risk at n=39'
        ]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Global SHAP Feature Importance")
    X_all_arr2 = scaler.transform(df[FEATURES].fillna(df[FEATURES].median()))
    shap_exp2 = explainer(X_all_arr2)
    if shap_exp2.values.ndim == 3:
        sv2 = shap_exp2.values[:, :, 1]
    else:
        sv2 = shap_exp2.values

    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
    shap.summary_plot(
        sv2,
        df[FEATURES].fillna(df[FEATURES].median()).values,
        feature_names=readable_names,
        show=False,
        plot_type='bar'
    )
    ax_shap.set_title("Mean |SHAP Value| — Global Feature Importance", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig_shap)

    st.markdown("---")
    st.subheader("Custom Prediction")
    st.markdown("Adjust the sliders to simulate a hypothetical country's emission profile:")

    input_vals = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        col = cols[i % 2]
        fmin = float(df[feat].min())
        fmax = float(df[feat].max())
        fmed = float(df[feat].median())
        input_vals[feat] = col.slider(
            label(feat),
            min_value=fmin,
            max_value=fmax,
            value=fmed,
            format="%.2f",
            key=feat
        )

    X_custom = np.array([[input_vals[f] for f in FEATURES]])
    X_custom_sc = scaler.transform(X_custom)
    pred_custom = model.predict(X_custom_sc)[0]
    prob_custom = model.predict_proba(X_custom_sc)[0][1]

    if pred_custom == 1:
        st.success(f"✅ Prediction: **Almost Sufficient (On Track)** — Confidence: {prob_custom:.1%}")
    else:
        st.error(f"❌ Prediction: **Not on Track** — On-track probability: {prob_custom:.1%}")

st.markdown("---")
st.caption(
    "BSAN 6070 Final Project — Spring 2026 | "
    "Data: Our World in Data / Climate Action Tracker | "
    "Model: Random Forest Classifier with hyperparameter tuning"
)
