import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans


# =========================================================
# 1. PAGE CONFIGURATION AND PROFESSIONAL UI
# =========================================================

st.set_page_config(
    page_title="Healthcare Risk Profiling",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #167D8D;
        --primary-dark: #105E6B;
        --surface: #FFFFFF;
        --text: #17324D;
        --muted: #52677D;
    }

    .stApp {
        background-color: #F5F7FA;
        color: #243B53;
    }

    /* Sidebar: ensure every navigation label remains readable. */
    [data-testid="stSidebar"] {
        background-color: #EAF0F6 !important;
        border-right: 1px solid #D5DEE8;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #243B53 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #17324D !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [role="radiogroup"] p {
        color: #243B53 !important;
        font-size: 0.92rem !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 5px 7px;
        border-radius: 6px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #DCE8F1 !important;
    }

    /* Main page typography. */
    h1, h2, h3, h4, h5, h6 {
        color: #17324D !important;
        letter-spacing: -0.25px;
    }

    .stApp p,
    .stApp label,
    .stApp [data-testid="stCaptionContainer"] {
        color: #40566D;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5ED;
        border-radius: 10px;
        padding: 15px 17px;
        box-shadow: 0 1px 3px rgba(20, 40, 60, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #52677D !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #17324D !important;
        font-weight: 650;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #DCE5ED;
        border-radius: 10px;
        padding: 20px;
    }

    /* Buttons: strong contrast in normal, hover and focus states. */
    .stButton > button,
    .stFormSubmitButton > button,
    button[kind="primary"] {
        background-color: #167D8D !important;
        color: #FFFFFF !important;
        border: 1px solid #167D8D !important;
        border-radius: 7px;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button *,
    .stFormSubmitButton > button *,
    button[kind="primary"] * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    button[kind="primary"]:hover {
        background-color: #105E6B !important;
        color: #FFFFFF !important;
        border-color: #105E6B !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 6px;
    }

    .page-subtitle {
        color: #52677D !important;
        font-size: 0.94rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .section-note {
        color: #52677D !important;
        font-size: 0.88rem;
    }

    .result-panel {
        background: #FFFFFF;
        border: 1px solid #DCE5ED;
        border-left: 4px solid #167D8D;
        border-radius: 8px;
        padding: 15px 18px;
        margin: 8px 0 16px 0;
        color: #17324D !important;
    }

    .result-panel strong {
        color: #17324D !important;
    }

    hr {
        border-color: #DFE6ED;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. DATASET CONFIGURATION
# =========================================================

DATA_PATH = Path("dataset/processed.cleveland.data")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

FEATURES = COLUMNS[:-1]

NUMERIC_COLS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_COLS = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


@st.cache_data
def load_data(path_string):
    data = pd.read_csv(
        path_string,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True
    )

    # Convert all columns to numeric and handle invalid values
    for column in COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["target"]).reset_index(drop=True)
    data["target_binary"] = (data["target"] > 0).astype(int)

    return data


if not DATA_PATH.exists():
    st.error(
        "Dataset file was not found. Please keep "
        "'processed.cleveland.data' inside the dataset folder."
    )
    st.stop()

df = load_data(str(DATA_PATH))

if df.empty:
    st.error("The dataset is empty or could not be loaded.")
    st.stop()


# =========================================================
# 3. LOGISTIC REGRESSION MODEL
# =========================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURES]
    y = data["target_binary"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS)
    ])

    classifier = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    classifier.fit(X_train, y_train)

    test_predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)

    return classifier, accuracy


model, model_accuracy = train_model(df)


# =========================================================
# 4. K-MEANS CLUSTERING
# =========================================================

@st.cache_resource
def train_clustering(data):

    X = data[FEATURES]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS)
    ])

    X_processed = preprocessing.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_processed)

    score = silhouette_score(X_processed, labels)

    return preprocessing, kmeans, labels, score


cluster_preprocessor, kmeans_model, cluster_labels, silhouette = (
    train_clustering(df)
)

cluster_data = df.copy()
cluster_data["Cluster"] = cluster_labels

cluster_summary = cluster_data.groupby("Cluster").agg(
    Patients=("age", "count"),
    Average_Age=("age", "mean"),
    Average_Cholesterol=("chol", "mean"),
    Average_BP=("trestbps", "mean"),
    Average_Heart_Rate=("thalach", "mean"),
    Disease_Count=("target_binary", "sum"),
    Disease_Rate=("target_binary", "mean")
).reset_index()

cluster_summary["Disease_Rate"] *= 100


# =========================================================
# 5. COMMON HELPERS
# =========================================================

def page_heading(title, subtitle):
    st.title(title)
    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def cluster_profile_table():
    table = cluster_summary.copy()
    table["Cluster"] = table["Cluster"].apply(
        lambda value: f"Cluster {int(value)}"
    )
    table = table.rename(columns={
        "Patients": "Patient Count",
        "Average_Age": "Average Age",
        "Average_Cholesterol": "Average Cholesterol",
        "Average_BP": "Average Blood Pressure",
        "Average_Heart_Rate": "Average Heart Rate",
        "Disease_Count": "Disease-positive Patients",
        "Disease_Rate": "Disease Prevalence (%)"
    })

    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(2)

    return table


def style_chart(fig):
    """Apply consistent high-contrast styling to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#243B53"
        ),
        title_font=dict(size=16, color="#17324D"),
        legend=dict(
            font=dict(size=12, color="#243B53"),
            title_font=dict(size=12, color="#17324D"),
            bgcolor="rgba(255,255,255,0.92)"
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(color="#17324D", size=12)
        ),
        margin=dict(t=48, b=48, l=55, r=25)
    )

    fig.update_xaxes(
        title_font=dict(size=12, color="#243B53"),
        tickfont=dict(size=11, color="#40566D"),
        gridcolor="#E5EAF0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1"
    )

    fig.update_yaxes(
        title_font=dict(size=12, color="#243B53"),
        tickfont=dict(size=11, color="#40566D"),
        gridcolor="#E5EAF0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1"
    )

    return fig


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("Healthcare Analytics")
st.sidebar.caption("DMDW Academic Project")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient Prediction",
        "Clustering Analysis"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Dataset: UCI Cleveland Heart Disease")
st.sidebar.caption(f"Records available: {len(df)}")


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":

    page_heading(
        "Healthcare Patient Risk Profiling",
        "Heart Disease Data Mining and Analytics | DMDW Project"
    )

    st.info(
        "An academic decision-support prototype built using "
        "the UCI Cleveland Heart Disease dataset."
    )

    total = len(df)
    disease = int(df["target_binary"].sum())
    healthy = total - disease

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total Patients", total)
    metric2.metric("Healthy Records", healthy)
    metric3.metric("Disease-positive Records", disease)
    metric4.metric("Model Accuracy", f"{model_accuracy * 100:.1f}%")

    st.divider()
    st.subheader("Dataset Overview")

    left, right = st.columns(2)

    with left:
        st.markdown("**Patient Distribution**")

        counts = pd.DataFrame({
            "Patient Status": ["Healthy", "Disease"],
            "Patients": [healthy, disease]
        })

        fig = px.pie(
            counts,
            names="Patient Status",
            values="Patients",
            hole=0.48,
            color="Patient Status",
            color_discrete_map={
                "Healthy": "#299D9F",
                "Disease": "#D66B6B"
            }
        )

        fig.update_layout(
            margin=dict(t=20, b=15, l=10, r=10),
            legend_title_text=""
        )

        fig = style_chart(fig)
    st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("**Age Distribution**")

        age_data = df.copy()
        age_data["Patient Status"] = age_data["target_binary"].map({
            0: "Healthy",
            1: "Disease"
        })

        fig = px.histogram(
            age_data,
            x="age",
            color="Patient Status",
            nbins=20,
            barmode="overlay",
            opacity=0.8,
            color_discrete_map={
                "Healthy": "#299D9F",
                "Disease": "#D66B6B"
            },
            labels={
                "age": "Age (years)",
                "count": "Number of patients"
            }
        )

        fig.update_layout(
            margin=dict(t=20, b=15, l=10, r=10),
            legend_title_text=""
        )

        fig = style_chart(fig)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Dataset Preview")
    st.caption("First 10 records from the Cleveland dataset.")

    st.dataframe(
        df[COLUMNS].head(10),
        width="stretch",
        hide_index=True
    )


# =========================================================
# 8. PATIENT PREDICTION WITH CLUSTER PROFILE
# =========================================================

elif page == "Patient Prediction":

    page_heading(
        "Patient Risk Analysis",
        "Enter patient attributes to generate a model prediction "
        "and identify the corresponding K-Means cluster."
    )

    st.info(
        "Enter the 13 clinical attributes. Both Logistic Regression "
        "and K-Means will analyse the same patient input."
    )

    with st.form("patient_form"):

        st.subheader("Patient Clinical Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)", 20, 100, 50
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)", 80, 250, 130
            )

            restecg = st.selectbox(
                "Resting ECG", [0, 1, 2]
            )

            oldpeak = st.number_input(
                "Oldpeak", 0.0, 10.0, 1.0, step=0.1
            )

            ca = st.selectbox(
                "Major Vessels (CA)", [0, 1, 2, 3]
            )

        with col2:
            sex = st.selectbox(
                "Sex",
                [1, 0],
                format_func=lambda value:
                    "Male" if value == 1 else "Female"
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)", 100, 600, 240
            )

            thalach = st.number_input(
                "Maximum Heart Rate", 60, 250, 150
            )

            slope = st.selectbox(
                "Slope", [1, 2, 3]
            )

            thal = st.selectbox(
                "Thal", [3, 6, 7]
            )

        with col3:
            cp = st.selectbox(
                "Chest Pain Type", [1, 2, 3, 4]
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120", [0, 1]
            )

            exang = st.selectbox(
                "Exercise Induced Angina", [0, 1]
            )

        submitted = st.form_submit_button(
            "Analyse Patient",
            width="stretch"
        )

    if submitted:

        patient = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }], columns=FEATURES)

        # Logistic Regression
        prediction = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0][1])

        if probability < 0.30:
            risk = "Low"
        elif probability <= 0.70:
            risk = "Moderate"
        else:
            risk = "High"

        # K-Means cluster assignment
        patient_processed = cluster_preprocessor.transform(patient)
        patient_cluster = int(
            kmeans_model.predict(patient_processed)[0]
        )

        profile = cluster_summary[
            cluster_summary["Cluster"] == patient_cluster
        ].iloc[0]

        st.divider()
        st.subheader("Analysis Results")

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Estimated Disease Probability",
            f"{probability * 100:.2f}%"
        )

        result2.metric(
            "Model Prediction",
            "Disease" if prediction == 1 else "Healthy"
        )

        result3.metric(
            "Model Risk Band",
            f"{risk} Risk"
        )

        if risk == "Low":
            st.success(
                "The model estimates a relatively lower disease probability."
            )
        elif risk == "Moderate":
            st.warning(
                "The model estimates an intermediate disease probability."
            )
        else:
            st.error(
                "The model estimates a higher disease probability."
            )

        st.progress(probability)

        st.divider()
        st.subheader("K-Means Cluster Assignment")

        st.markdown(
            f'<div class="result-panel">'
            f'<strong>Assigned Group: Cluster {patient_cluster}</strong><br>'
            f'<span class="section-note">'
            f'The patient is grouped with records having similar '
            f'clinical feature patterns.'
            f'</span></div>',
            unsafe_allow_html=True
        )

        cluster_col1, cluster_col2, cluster_col3 = st.columns(3)

        cluster_col1.metric(
            "Patients in This Cluster",
            int(profile["Patients"])
        )

        cluster_col2.metric(
            "Cluster Disease Prevalence",
            f"{profile['Disease_Rate']:.2f}%"
        )

        cluster_col3.metric(
            "Cluster Average Age",
            f"{profile['Average_Age']:.1f} years"
        )

        st.markdown("**Cluster Profile**")

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Average Cholesterol",
            f"{profile['Average_Cholesterol']:.1f} mg/dl"
        )

        p2.metric(
            "Average Blood Pressure",
            f"{profile['Average_BP']:.1f} mm Hg"
        )

        p3.metric(
            "Average Maximum Heart Rate",
            f"{profile['Average_Heart_Rate']:.1f}"
        )

        st.markdown("**Comparison with Other Clusters**")

        st.dataframe(
            cluster_profile_table(),
            width="stretch",
            hide_index=True
        )

        st.divider()
        st.subheader("Patient Input Summary")

        st.dataframe(
            patient,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Disease probability is produced by Logistic Regression. "
            "Cluster disease prevalence describes existing records "
            "in the assigned group; it is not the individual patient's "
            "disease probability."
        )

        st.warning(
            "Academic demonstration only. This machine-learning output "
            "is not a medical diagnosis or a substitute for professional "
            "medical assessment."
        )


# =========================================================
# 9. CLUSTERING ANALYSIS
# =========================================================

elif page == "Clustering Analysis":

    page_heading(
        "K-Means Clustering Analysis",
        "Patient segmentation based on similarities in clinical features."
    )

    st.info(
        "K-Means is an unsupervised learning algorithm. "
        "It groups the 303 patient records into three clusters."
    )

    cluster_counts = (
        cluster_data["Cluster"].value_counts().sort_index()
    )

    st.subheader("Cluster Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Patients", len(df))
    c2.metric("Number of Clusters", 3)
    c3.metric("Silhouette Score", f"{silhouette:.3f}")
    c4.metric("Input Features", len(FEATURES))

    st.divider()
    st.subheader("Patient Distribution by Cluster")

    distribution = pd.DataFrame({
        "Cluster": [
            f"Cluster {int(i)}" for i in cluster_counts.index
        ],
        "Patients": cluster_counts.values
    })

    fig = px.bar(
        distribution,
        x="Cluster",
        y="Patients",
        color="Cluster",
        text="Patients",
        color_discrete_sequence=["#299D9F", "#3976A8", "#D6A15D"],
        title="Number of Patients in Each Cluster"
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(t=45, b=15, l=10, r=10),
        xaxis_title="Cluster",
        yaxis_title="Number of patients"
    )

    fig = style_chart(fig)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Cluster Visualization")

    fig = px.scatter(
        cluster_data,
        x="age",
        y="thalach",
        color=cluster_data["Cluster"].astype(str),
        hover_data=["chol", "trestbps", "target"],
        labels={
            "age": "Patient Age (years)",
            "thalach": "Maximum Heart Rate",
            "color": "Cluster"
        },
        color_discrete_sequence=["#299D9F", "#3976A8", "#D6A15D"],
        title="Patient Groups: Age vs Maximum Heart Rate"
    )

    fig.update_layout(
        margin=dict(t=45, b=15, l=10, r=10),
        legend_title_text="Cluster"
    )

    fig = style_chart(fig)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Cluster-wise Patient Profile")

    st.dataframe(
        cluster_profile_table(),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Cluster IDs are generated by K-Means and do not represent "
        "predefined medical risk categories. Disease prevalence is "
        "calculated from the original dataset labels."
    )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.caption(
    "Healthcare Patient Risk Profiling and Disease Prediction"
)

st.caption(
    "DMDW Academic Project | UCI Cleveland Heart Disease Dataset"
)

st.caption(
    "For educational and research purposes only. "
    "Not intended for clinical diagnosis."
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.viewerBadge_container__1QSob {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)