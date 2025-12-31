import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Tourism Dashboard", layout='wide')

# ==== PREMIUM UI THEME ==== #
st.markdown("""
    <style>
        .main {background-color: #0e1117; color: #FAFAFA;}
        .stMetric {background-color:#1d232f; padding:20px; border-radius:16px;}
        .header-text {font-size:32px; font-weight:700; color:#00C0FF;}
        .subheader-text {font-size:22px; color:#C4C4C4;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #FAFAFA; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .kpi-card {
        background:#1d232f;
        padding:18px;
        border-radius:16px;
        text-align:center;
        box-shadow:0 6px 18px rgba(0,0,0,0.45);
    }

    .kpi-title { font-size:13px; color:#9aa4b2; }
    .kpi-value { font-size:28px; font-weight:700; }
    .kpi-icon { font-size:26px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ==== DATA LOAD ==== #
file_path = r"C:\Users\nikhi\Machine Learning\Project\Traveling_Data.csv"
df = pd.read_csv(file_path)

# ---- Converting dtype ---- #
object_cols = [
    'CityTier','ProdTaken','NumberOfPersonVisiting','OwnCar',
    'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips',
    'NumberOfChildrenVisiting','Passport','PitchSatisfactionScore'
]

df[object_cols] = df[object_cols].astype('object')


df_selection = df.copy()

st.markdown("<div class='header-text'>🧳 Travel Package Purchase Insights Dashboard</div>", unsafe_allow_html=True)

# === KPI FUNCTION === # 
def kpi_card(title, value, icon):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ==== KPI SECTION ==== #
total_customers = len(df_selection)
purchased = df_selection['ProdTaken'].sum()
not_purchased = total_customers - purchased
conversion_rate = round((purchased / total_customers) * 100, 2)
avg_followups = int(np.ceil(df_selection[df_selection['ProdTaken'] == 1]['NumberOfFollowups'].mean()))
avg_age = int(round(df_selection['Age'].mean()))

st.subheader("📌 Key Metrics")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1: kpi_card("Total Customers", total_customers, "👥")
with c2: kpi_card("Purchased", purchased, "🛒")
with c3: kpi_card("Not Purchased", not_purchased, "❌")
with c4: kpi_card("Conversion Rate", f"{conversion_rate}%", "📈")
with c5: kpi_card("Avg Follow-ups", avg_followups, "📞")
with c6: kpi_card("Avg Age", avg_age, "🎯")

st.markdown("---")

st.subheader("📊 EDA Visualization")


num_cols = df_selection.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_selection.select_dtypes(include=['object']).columns.tolist()

# st.subheader("Column Segregation")
# st.write("Numerical Columns:", num_cols)
# st.write("Categorical Columns:", cat_cols)


# ==== EDA ==== #
analysis_type = st.sidebar.selectbox(
    "Choose EDA Type",
    ["Univariate Analysis", "Bivariate Analysis"],
    key="analysis_type"
)


# ==== UNIVARIATE Analysis ==== #

if analysis_type == "Univariate Analysis":

    variable_type = st.sidebar.selectbox(
        "Select Variable Type",
        ["Numerical", "Categorical"],
        key="variable_type"
    )

    # ===== NUMERICAL ===== #
    if variable_type == "Numerical":

        col = st.sidebar.selectbox(
            "Select Numerical Column",
            num_cols,
            key="num_col"
        )

        colA, colB = st.columns(2)

        # ==== Histogram + KDE ==== #
        with colA:
            st.markdown(f"#### Histogram + KDE of {col}")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            sns.histplot(df_selection[col], kde=True, ax=ax,
                         color="#57A8FF", edgecolor="white")

            ax.tick_params(colors="white")
            ax.set_xlabel(col, color="white")
            ax.set_ylabel("Count", color="white")
            for spine in ax.spines.values():
                spine.set_color("gray")

            st.pyplot(fig)

        # ==== Boxplot ==== #
        with colB:
            st.markdown(f"#### Boxplot of {col}")
            fig, ax = plt.subplots(figsize=(6, 3.6))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            sns.boxplot(x=df_selection[col], ax=ax, color="#70FF75")

            ax.tick_params(colors="white")
            ax.set_xlabel(col, color="white")
            for spine in ax.spines.values():
                spine.set_color("gray")

            st.pyplot(fig)

        # ==== Statistical Summary ==== #
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Statistical Summary")

        stats = df_selection[col].describe().round(2)

        html_summary = f"""
        <div style="
            background-color:#1d232f;
            padding:15px;
            border-radius:10px;
            line-height:1.8;
            font-size:15px;
        ">
        <b>Summary of {col}</b><br><br>

        <b>Count   :</b> {stats['count']:.0f}<br>
        <b>Mean    :</b> {stats['mean']}<br>
        <b>Std Dev :</b> {stats['std']}<br>
        <b>Min     :</b> {stats['min']}<br>
        <b>Max     :</b> {stats['max']}<br>
        <b>25%     :</b> {stats['25%']}<br>
        <b>Median  :</b> {stats['50%']}<br>
        <b>75%     :</b> {stats['75%']}<br>
        </div>
        """

        st.sidebar.markdown(html_summary, unsafe_allow_html=True)


    # ===== CATEGORICAL ===== #
    else:

        col = st.sidebar.selectbox(
            "Select Categorical Column",
            cat_cols,
            key="cat_col"
        )

        colA, colB = st.columns(2)

        # prepare grouped df
        cat_df = (
            df_selection[col]
            .value_counts()
            .rename_axis(col)
            .reset_index(name="count")
        )

        # === Bar Plot === #
        with colA:
            st.markdown(f"#### Bar Chart - {col}")
            fig = px.bar(
                cat_df,
                x=col,
                y="count",
                opacity=0.85
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        # === Pie Plot === #
        with colB:
            st.markdown(f"#### Pie Chart - {col}")
            fig = px.pie(cat_df, names=col, values="count")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        # categorical summary
        st.sidebar.subheader("📊 Count Summary")
        for c, count in cat_df.values:
            st.sidebar.markdown(f"{c} : {count}")

# ==== BIVARIATE ANALYSIS ==== #
elif analysis_type == "Bivariate Analysis":

    relation_type = st.sidebar.selectbox(
        "Select Variable Relationship",
        ["Num vs Num", "Cat vs Cat", "Num vs Cat"],
        key="bivariate_type"
    )

    colA, colB = st.columns(2)      # side-by-side layout   

    # ==== NUM vs NUM ==== #
    if relation_type == "Num vs Num":
        x = st.sidebar.selectbox("Select X variable", num_cols, key="num_num_x")
        y = st.sidebar.selectbox("Select Y variable", num_cols, key="num_num_y")

        # === Scatter plot === # 
        with colA:
            st.subheader(f"📍 Scatter Plot ({x} vs {y})")
            fig1 = px.scatter(df_selection, x=x, y=y, opacity=0.8)
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig1, use_container_width=True)

        # === Density Heatmap === #
        with colB:
            st.subheader(f"📍 Correlation Heat Tile")
            fig2 = px.density_heatmap(df_selection, x=x, y=y)
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)


    # ==== CAT vs CAT ==== #
    elif relation_type == "Cat vs Cat":
        x = st.sidebar.selectbox("Select X variable", cat_cols, key="cat_cat_x")
        y = st.sidebar.selectbox("Select Y variable", cat_cols, key="cat_cat_y")
        
        # === Histogram === # 
        with colA:
            st.subheader(f"📍 Count Grouped ({x} vs {y})")
            fig1 = px.histogram(df_selection, x=x, color=y, barmode="group")
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # === Crosstab === # 
        with colB:
            st.subheader(f"📍 Cross Tab (%)")
            ctab = pd.crosstab(df_selection[x], df_selection[y], normalize='index') * 100
            fig2 = px.imshow(ctab, text_auto=True)
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)


    # ==== NUM vs CAT ==== #
    else:
        x = st.sidebar.selectbox("Select Numerical Variable", num_cols, key="num_cat_x")
        y = st.sidebar.selectbox("Select Category Variable", cat_cols, key="num_cat_y")

        # === Box Plot === #
        with colA:
            st.subheader(f"📍 Box Plot ({x} across {y})")
            fig1 = px.box(df_selection, x=y, y=x)
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig1, use_container_width=True)

        # === Bar plot === # 
        with colB:
            st.subheader(f"📍 Mean {x} by {y}")
            mean_df = df_selection.groupby(y)[x].mean().reset_index()
            fig2 = px.bar(mean_df, x=y, y=x)
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)
