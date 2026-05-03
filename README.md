# 🌍 Paris Agreement Tracker — GHG Emissions Classifier

**BSAN 6070: Introduction to Machine Learning | Spring 2026 | Loyola Marymount University**

> Can we predict whether a country is on track to meet the Paris Agreement using only its historical greenhouse gas emissions?

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bsan6070-final-ghg.streamlit.app/)

---

## Problem Statement

The Paris Agreement (2015) requires countries to reduce emissions enough to limit global warming to 1.5–2°C. The [Climate Action Tracker (CAT)](https://climateactiontracker.org) independently rates each country's alignment — but only for 39 major emitters, using expert analysis.

This project builds an ML classifier that **replicates CAT ratings using only historical GHG emission trajectories (1950–2024)**, making alignment prediction scalable, transparent, and auditable for any country with emissions data.

**Predictive Questions:**
1. Can we classify a country as "on track" (Almost Sufficient) vs. "not on track" using emission trends alone?
2. What are the top 5–8 features most predictive of Paris Agreement alignment?

---

## Live Demo

**[bsan6070-final-ghg.streamlit.app](https://bsan6070-final-ghg.streamlit.app/)**

The app includes:
- 🗺️ **Global Overview** — world map of actual CAT ratings vs. ML predictions
- 🔬 **Country Deep-Dive** — feature profile + SHAP waterfall for any country
- 📊 **Model Insights** — feature importance, performance table, custom prediction sliders

---

## Data Sources

| Dataset | Source | Size | Role |
|---------|--------|------|------|
| GHG Emissions by Country (1950–2024) | [Our World in Data](https://ourworldindata.org/co2-and-greenhouse-gas-emissions) | 14,925 rows × 5 cols | Feature engineering |
| CAT Country Ratings | [Climate Action Tracker](https://climateactiontracker.org) (scraped 2025) | 39 countries | Classification target |

---

## Features Engineered

From the raw time series we derive 8 predictive features per country:

| Feature | Description |
|---------|-------------|
| GHG Emissions 2024 | Total GHG in most recent year (Mt CO₂e) |
| GHG per Capita 2024 | Per-person emissions in 2024 (t CO₂e) |
| % Change 2015–2024 | Emission change since Paris Agreement |
| Emissions Slope Post-Paris | Linear trend slope 2015–2024 (Mt/yr) |
| Per Capita Slope Post-Paris | Per-capita trend 2015–2024 (t/yr) |
| % Change Pre vs. Post-Paris | Average emissions shift before vs. after 2015 |
| % From Peak Emissions | How far emissions have fallen from historic high |
| Reduction Ratio | Proportion of post-2015 years with year-on-year decline |

---

## Models & Results

Three models trained (one per team member), evaluated with **Stratified 5-Fold Cross-Validation**:

| Model | ROC-AUC | Accuracy | Role |
|-------|---------|----------|------|
| Logistic Regression | 0.900 | 77.1% | Member 1 · Interpretable baseline |
| **Random Forest ★** | **0.933** | **84.6%** | Member 2 · **Selected for deployment** |
| Gradient Boosting | 0.885 | 79.3% | Member 3 · Strong but higher overfitting risk at n=39 |

**Final model:** Random Forest with GridSearchCV hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`, `class_weight`).

**Top predictive features (SHAP):**
1. Reduction Ratio (post-2015) — consistency of year-on-year declines
2. % Change 2015–2024
3. Emissions Slope Post-Paris
4. % From Peak Emissions
5. % Change Pre vs. Post-Paris

---

## Project Structure

```
├── app.py                        # Streamlit application
├── notebooks/
│   └── GHG_Paris_Prediction.ipynb  # Full ML pipeline (fully executed)
├── data/
│   ├── ghg_emissions.csv         # GHG time series (199 countries, 1950–2024)
│   └── cat_ratings.csv           # CAT ratings (39 countries)
├── country_features.csv          # Engineered features used by Streamlit
├── model.sav                     # Trained Random Forest (pickle)
├── scaler.sav                    # StandardScaler (pickle)
├── features_list.sav             # Selected feature names (pickle)
└── requirements.txt              # Python dependencies
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/Sadifs/BSAN6070-Final.git
cd BSAN6070-Final

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

The notebook can be opened with Jupyter:
```bash
jupyter notebook notebooks/GHG_Paris_Prediction.ipynb
```

---

## Team

| Member | Algorithm |
|--------|-----------|
| [Member 1] | Logistic Regression |
| [Member 2] | Random Forest |
| [Member 3] | Gradient Boosting |

---

## References

- Rogelj et al. (2016). Paris Agreement climate proposals need a boost to keep warming well below 2°C. *Nature Climate Change*.
- Climate Action Tracker (2024). [climateactiontracker.org](https://climateactiontracker.org)
- UNFCCC NDC Registry. [unfccc.int/NDCREG](https://unfccc.int/NDCREG)
- Our World in Data — CO₂ and Greenhouse Gas Emissions. [ourworldindata.org](https://ourworldindata.org/co2-and-greenhouse-gas-emissions)
