import os
import re
from glob import glob

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="MSR Port Coupling Coordination Analysis",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #315f72 0%, #2f7d6f 55%, #d79b47 100%);
        padding: 1.7rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.15rem;
        letter-spacing: 0;
    }
    .main-header p {
        margin: .4rem 0 0 0;
        opacity: .94;
    }
    .small-note {
        color: #667085;
        font-size: .9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


NATURAL_KEYWORDS = [
    "slope",
    "elev",
    "elevation",
    "bathy",
    "bathymetry",
    "evi",
    "ndvi",
    "rugged",
    "terrain",
    "coast",
    "water",
    "forest",
    "grass",
    "cropland",
]

NEGATIVE_INDICATORS = [
    "so2",
    "pollution",
    "exposure",
    "slope",
    "rugged",
    "tri",
]


class CouplingCoordinationModel:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def min_max_normalize(data, positive=True):
        values = np.array(data, dtype=float)
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        if np.isnan(min_val) or np.isnan(max_val) or max_val - min_val == 0:
            return np.ones_like(values) * 0.5
        norm = (values - min_val) / (max_val - min_val)
        return norm if positive else 1 - norm

    @staticmethod
    def entropy_weight_method(data_matrix):
        n_samples, _ = data_matrix.shape
        if n_samples <= 1:
            return np.ones(data_matrix.shape[1]) / data_matrix.shape[1]
        p_matrix = data_matrix / (np.sum(data_matrix, axis=0) + 1e-10)
        p_matrix = np.clip(p_matrix, 1e-10, 1 - 1e-10)
        e_j = -(1 / np.log(n_samples)) * np.sum(p_matrix * np.log(p_matrix), axis=0)
        d_j = 1 - e_j
        return d_j / (np.sum(d_j) + 1e-10)

    @staticmethod
    def calculate_coupling_degree(f_score, g_score):
        denominator = f_score + g_score
        if denominator == 0:
            return 0.0
        return 2 * np.sqrt((f_score * g_score) / (denominator**2 + 1e-10))

    def calculate_coordination_degree(self, f_score, g_score):
        combined_score = self.alpha * f_score + self.beta * g_score
        coupling = self.calculate_coupling_degree(f_score, g_score)
        return np.sqrt(coupling * combined_score)

    def fit(self, geo_data, socio_data, negative_columns=None):
        negative_columns = set(negative_columns or [])
        geo_df = geo_data.copy()
        socio_df = socio_data.copy()

        geo_norm = pd.DataFrame(index=geo_df.index)
        for col in geo_df.columns:
            geo_norm[col] = self.min_max_normalize(geo_df[col], positive=col not in negative_columns)

        socio_norm = pd.DataFrame(index=socio_df.index)
        for col in socio_df.columns:
            socio_norm[col] = self.min_max_normalize(socio_df[col], positive=col not in negative_columns)

        geo_weights = self.entropy_weight_method(geo_norm.values)
        socio_weights = self.entropy_weight_method(socio_norm.values)

        f_scores = np.sum(geo_norm.values * geo_weights, axis=1)
        g_scores = np.sum(socio_norm.values * socio_weights, axis=1)
        coupling_values = np.array(
            [self.calculate_coupling_degree(f, g) for f, g in zip(f_scores, g_scores)]
        )
        coordination_values = np.array(
            [self.calculate_coordination_degree(f, g) for f, g in zip(f_scores, g_scores)]
        )

        return {
            "f_scores": f_scores,
            "g_scores": g_scores,
            "coupling_C": coupling_values,
            "coordination_D": coordination_values,
            "geo_weights": geo_weights,
            "socio_weights": socio_weights,
            "geo_columns": list(geo_df.columns),
            "socio_columns": list(socio_df.columns),
            "geo_norm": geo_norm,
            "socio_norm": socio_norm,
        }

    @staticmethod
    def get_level(value):
        if value >= 0.9:
            return "Excellent Coordination"
        if value >= 0.8:
            return "Good Coordination"
        if value >= 0.7:
            return "Moderate Coordination"
        if value >= 0.6:
            return "Primary Coordination"
        if value >= 0.5:
            return "Barely Coordination"
        if value >= 0.4:
            return "Near Disorder"
        if value >= 0.3:
            return "Mild Disorder"
        if value >= 0.2:
            return "Moderate Disorder"
        return "Severe Disorder"


def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def find_port_column(df):
    candidates = ["port_name", "port", "name", "portname", "harbor", "terminal"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_indicator_name(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"^(msr_|port_|ports_|msr_port_)", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_").lower()


def is_negative_indicator(column_name):
    name = column_name.lower()
    return any(keyword in name for keyword in NEGATIVE_INDICATORS)


def read_csv_any(source):
    df = pd.read_csv(source)
    return clean_columns(df)


def extract_indicator_file(file_path):
    df = read_csv_any(file_path)
    port_col = find_port_column(df)
    value_col = "mean" if "mean" in df.columns else None

    if value_col is None:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["lat", "lon", "latitude", "longitude", "year"]]
        if numeric_cols:
            value_col = numeric_cols[0]

    if not port_col or not value_col:
        return None

    indicator = safe_indicator_name(file_path)
    out = df[[port_col, value_col]].copy()
    out.columns = ["port_name", indicator]
    out["port_name"] = out["port_name"].astype(str).str.strip()
    out[indicator] = pd.to_numeric(out[indicator], errors="coerce")
    return out.groupby("port_name", as_index=False).mean(numeric_only=True)


def summarize_pollution(df, year_mode, value_mode):
    df = clean_columns(df)
    port_col = find_port_column(df)
    if not port_col or "year" not in df.columns:
        return None, None

    pwe_col = "so2_population_weighted_exposure_mol_per_m2"
    area_col = "so2_area_mean_mol_per_m2"
    value_col = pwe_col if value_mode == "Population weighted exposure" and pwe_col in df.columns else area_col
    if value_col not in df.columns:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "year"]
        if not numeric_cols:
            return None, None
        value_col = numeric_cols[0]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["year", value_col]).copy()
    df["port_name"] = df[port_col].astype(str).str.strip()

    if year_mode == "Latest year":
        idx = df.groupby("port_name")["year"].idxmax()
        indicator = df.loc[idx, ["port_name", value_col]].copy()
    elif year_mode == "Mean across years":
        indicator = df.groupby("port_name", as_index=False)[value_col].mean()
    else:
        selected_year = int(year_mode)
        indicator = df[df["year"] == selected_year][["port_name", value_col]].copy()

    indicator = indicator.rename(columns={value_col: "so2_exposure"})
    return indicator.groupby("port_name", as_index=False).mean(numeric_only=True), df


@st.cache_data(show_spinner=False)
def load_port_data(folder_path, pollution_path, year_mode, value_mode):
    patterns = ["MSR_Port_*.csv", "Port_*.csv", "Ports_*.csv"]
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(folder_path, pattern)))

    indicator_frames = []
    skipped_files = []
    for file_path in sorted(set(files)):
        if pollution_path and os.path.abspath(file_path) == os.path.abspath(pollution_path):
            continue
        extracted = extract_indicator_file(file_path)
        if extracted is None:
            skipped_files.append(os.path.basename(file_path))
        else:
            indicator_frames.append((safe_indicator_name(file_path), extracted))

    pollution_long = None
    if pollution_path and os.path.exists(pollution_path):
        pollution_df = read_csv_any(pollution_path)
        pollution_indicator, pollution_long = summarize_pollution(pollution_df, year_mode, value_mode)
        if pollution_indicator is not None:
            indicator_frames.append(("so2_exposure", pollution_indicator))

    if not indicator_frames:
        return None, None, None, None, skipped_files

    all_ports = sorted(
        set().union(*[set(frame["port_name"].dropna().astype(str)) for _, frame in indicator_frames])
    )
    merged = pd.DataFrame({"port_name": all_ports})
    for _, frame in indicator_frames:
        merged = merged.merge(frame, on="port_name", how="left")

    natural_cols = [
        col for col in merged.columns if col != "port_name" and any(k in col.lower() for k in NATURAL_KEYWORDS)
    ]
    socio_cols = [col for col in merged.columns if col != "port_name" and col not in natural_cols]

    return merged, natural_cols, socio_cols, pollution_long, skipped_files


@st.cache_data(show_spinner=False)
def load_coordinates(folder_path):
    coord_files = [
        os.path.join(folder_path, "port_coordinates.csv"),
        os.path.join(folder_path, "ports_coordinates.csv"),
    ]
    for path in coord_files:
        if os.path.exists(path):
            df = read_csv_any(path)
            port_col = find_port_column(df)
            lat_col = "latitude" if "latitude" in df.columns else "lat" if "lat" in df.columns else None
            lon_col = "longitude" if "longitude" in df.columns else "lon" if "lon" in df.columns else None
            if port_col and lat_col and lon_col:
                out = df[[port_col, lat_col, lon_col]].copy()
                out.columns = ["port_name", "latitude", "longitude"]
                out["port_name"] = out["port_name"].astype(str).str.strip()
                out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
                out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
                return out.dropna(subset=["latitude", "longitude"])
    return None


def run_model(merged, natural_cols, socio_cols, alpha, beta):
    if not natural_cols or not socio_cols:
        return None, None

    model_input = merged.copy()
    numeric_cols = natural_cols + socio_cols
    for col in numeric_cols:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")

    model_input[numeric_cols] = model_input[numeric_cols].fillna(model_input[numeric_cols].median())
    model_input[numeric_cols] = model_input[numeric_cols].fillna(0)

    geo_df = model_input.set_index("port_name")[natural_cols]
    socio_df = model_input.set_index("port_name")[socio_cols]
    negative_cols = [c for c in numeric_cols if is_negative_indicator(c)]

    model = CouplingCoordinationModel(alpha=alpha, beta=beta)
    results = model.fit(geo_df, socio_df, negative_columns=negative_cols)
    result_df = pd.DataFrame(
        {
            "port_name": geo_df.index,
            "geo_score": results["f_scores"],
            "socio_score": results["g_scores"],
            "coupling_C": results["coupling_C"],
            "coordination_D": results["coordination_D"],
            "level": [model.get_level(v) for v in results["coordination_D"]],
        }
    ).sort_values("coordination_D", ascending=False)

    return results, result_df.reset_index(drop=True)


def render_summary(result_df):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ports", len(result_df))
    col2.metric("Mean D", f"{result_df['coordination_D'].mean():.3f}")
    col3.metric("Max D", f"{result_df['coordination_D'].max():.3f}")
    col4.metric("Min D", f"{result_df['coordination_D'].min():.3f}")


def render_ranking(result_df):
    col_left, col_right = st.columns([3, 2])
    with col_left:
        display = result_df[["port_name", "coordination_D", "level", "geo_score", "socio_score"]].copy()
        display.columns = ["Port", "Coordination D", "Level", "Geo Score", "Socio Score"]
        st.dataframe(
            display.round(4),
            use_container_width=True,
            column_config={
                "Coordination D": st.column_config.ProgressColumn(
                    "Coordination D", min_value=0, max_value=1, format="%.3f"
                )
            },
        )
    with col_right:
        counts = result_df["level"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.42, title="Coordination levels")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    top_n = st.slider("Display top ports", 5, min(50, len(result_df)), min(15, len(result_df)))
    top_df = result_df.head(top_n).sort_values("coordination_D")
    fig_bar = px.bar(
        top_df,
        x="coordination_D",
        y="port_name",
        orientation="h",
        color="level",
        labels={"coordination_D": "Coordination D", "port_name": "Port"},
        title="Top port ranking",
    )
    fig_bar.update_layout(height=max(420, top_n * 28), yaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)


def render_scatter(result_df):
    fig = px.scatter(
        result_df,
        x="geo_score",
        y="socio_score",
        color="coordination_D",
        size="coordination_D",
        hover_name="port_name",
        color_continuous_scale="Viridis",
        labels={
            "geo_score": "Geo-environmental score",
            "socio_score": "Socio-economic / pollution score",
            "coordination_D": "Coordination D",
        },
        title="Port system balance",
    )
    fig.add_vline(x=result_df["geo_score"].median(), line_dash="dash", line_color="#8a8f98")
    fig.add_hline(y=result_df["socio_score"].median(), line_dash="dash", line_color="#8a8f98")
    st.plotly_chart(fig, use_container_width=True)


def render_weights(results):
    weight_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "system": "Geo-environmental",
                    "indicator": results["geo_columns"],
                    "weight": results["geo_weights"],
                }
            ),
            pd.DataFrame(
                {
                    "system": "Socio-economic / pollution",
                    "indicator": results["socio_columns"],
                    "weight": results["socio_weights"],
                }
            ),
        ],
        ignore_index=True,
    )
    fig = px.bar(weight_df, x="weight", y="indicator", color="system", orientation="h", title="Entropy weights")
    fig.update_layout(yaxis_title=None, height=max(420, len(weight_df) * 32))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(weight_df.round(5), use_container_width=True)


def render_pollution(pollution_long):
    if pollution_long is None or pollution_long.empty:
        st.info("No port-year SO2 table was loaded.")
        return

    pwe_col = "so2_population_weighted_exposure_mol_per_m2"
    area_col = "so2_area_mean_mol_per_m2"
    available = [c for c in [pwe_col, area_col, "total_population"] if c in pollution_long.columns]
    ports = sorted(pollution_long["port_name"].dropna().unique())
    selected = st.multiselect("Ports", ports, default=ports[: min(5, len(ports))])
    metric = st.selectbox("Metric", available)
    plot_df = pollution_long[pollution_long["port_name"].isin(selected)]
    fig = px.line(plot_df, x="year", y=metric, color="port_name", markers=True, title="Port-year pollution trend")
    fig.update_layout(xaxis=dict(dtick=1), legend_title_text="Port")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(plot_df.sort_values(["port_name", "year"]), use_container_width=True)


def render_map(result_df, folder_path):
    coords = load_coordinates(folder_path)
    if coords is None:
        st.warning("No port coordinate file found.")
        st.markdown(
            """
Create `port_coordinates.csv` in the data folder with this format:

```csv
port_name,latitude,longitude
Gwadar,25.1216,62.3254
Singapore,1.2644,103.8223
```
"""
        )
        return

    map_df = result_df.merge(coords, on="port_name", how="inner")
    if map_df.empty:
        st.warning("Coordinates were found, but no port names matched the analysis table.")
        return

    center = [map_df["latitude"].mean(), map_df["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron", control_scale=True)
    for _, row in map_df.iterrows():
        color = "#218c74" if row["coordination_D"] >= 0.6 else "#d9902f" if row["coordination_D"] >= 0.4 else "#c0392b"
        radius = 6 + float(row["coordination_D"]) * 13
        popup_html = (
            f"<b>{row['port_name']}</b><br>"
            f"Coordination D: {row['coordination_D']:.3f}<br>"
            f"Geo score: {row['geo_score']:.3f}<br>"
            f"Socio score: {row['socio_score']:.3f}<br>"
            f"Level: {row['level']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fmap)

    st_folium(fmap, width=1100, height=560, returned_objects=[])


def render_raw(merged, result_df):
    st.subheader("Merged indicator table")
    st.dataframe(merged, use_container_width=True)
    st.subheader("Model output")
    st.dataframe(result_df, use_container_width=True)


def main():
    st.markdown(
        """
<div class="main-header">
    <h1>21st Century Maritime Silk Road Port CCDM</h1>
    <p>Port-scale coupling coordination analysis using GEE-derived indicators and SO2 exposure.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Control Panel")
        data_path = st.text_input("Data folder", value=".")
        pollution_path = st.text_input(
            "SO2 port-year CSV",
            value=os.path.join(data_path, "Ports_SO2_population_weighted_exposure_2018_2025.csv"),
            help="Use the CSV exported by GEE_pollution.txt. Leave it as a path in this folder after exporting from Google Drive.",
        )
        value_mode = st.radio(
            "SO2 indicator",
            ["Population weighted exposure", "Area mean SO2"],
            horizontal=False,
        )
        year_mode = st.selectbox(
            "SO2 year aggregation",
            ["Latest year", "Mean across years"] + [str(y) for y in range(2018, 2026)],
        )
        st.divider()
        alpha = st.slider("Geo-environmental weight alpha", 0.0, 1.0, 0.5, 0.05)
        beta = st.slider("Socio-economic / pollution weight beta", 0.0, 1.0, 0.5, 0.05)
        run = st.button("Run port analysis", type="primary", use_container_width=True)

        st.markdown(
            """
<p class="small-note">
Accepted indicator files: <code>MSR_Port_*.csv</code>, <code>Port_*.csv</code>, or <code>Ports_*.csv</code>.
Each file should contain a port name column and a numeric value column, preferably <code>mean</code>.
</p>
""",
            unsafe_allow_html=True,
        )

    if not run and "port_result_df" not in st.session_state:
        st.info("Set the data folder and SO2 CSV path, then click Run port analysis.")
        st.markdown(
            """
Expected GEE pollution export columns:

```csv
port_name,year,start_date,end_date,population_year_used,SO2_area_mean_mol_per_m2,total_population,SO2_population_weighted_exposure_mol_per_m2
```
"""
        )
        return

    with st.spinner("Loading port data and running CCDM..."):
        merged, natural_cols, socio_cols, pollution_long, skipped = load_port_data(
            data_path, pollution_path, year_mode, value_mode
        )

    if merged is None:
        st.error("No port-level indicator data was found.")
        st.info(
            "Put the GEE SO2 export CSV in this folder, or add files like MSR_Port_Elevation.csv with port_name and mean columns."
        )
        return

    if skipped:
        st.warning(f"Skipped files without usable port/value columns: {', '.join(skipped)}")

    if not natural_cols:
        st.error("No geo-environmental port indicators were found.")
        st.info("Natural indicators are detected from names such as elevation, slope, evi, water, forest, coast, terrain.")
        return

    if not socio_cols:
        st.error("No socio-economic or pollution port indicators were found.")
        st.info("Add at least one non-natural indicator, or load the GEE SO2 port-year export CSV.")
        return

    results, result_df = run_model(merged, natural_cols, socio_cols, alpha, beta)
    if results is None:
        st.error("The model could not run with the current indicator groups.")
        return

    st.session_state.port_result_df = result_df
    st.success(f"Analysis complete: {len(result_df)} ports, {len(natural_cols)} geo indicators, {len(socio_cols)} socio/pollution indicators.")
    render_summary(result_df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Ranking", "Scatter", "Weights", "SO2 Trend", "Map", "Raw Data"]
    )

    with tab1:
        render_ranking(result_df)
    with tab2:
        render_scatter(result_df)
    with tab3:
        render_weights(results)
    with tab4:
        render_pollution(pollution_long)
    with tab5:
        render_map(result_df, data_path)
    with tab6:
        render_raw(merged, result_df)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download port CCDM results",
            result_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="msr_port_ccdm_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        report = f"""MSR Port Coupling Coordination Analysis Report
Ports analyzed: {len(result_df)}
Geo indicators: {', '.join(natural_cols)}
Socio/pollution indicators: {', '.join(socio_cols)}
Mean Coordination D: {result_df['coordination_D'].mean():.4f}
Max Coordination D: {result_df.iloc[0]['port_name']} ({result_df.iloc[0]['coordination_D']:.4f})
Min Coordination D: {result_df.iloc[-1]['port_name']} ({result_df.iloc[-1]['coordination_D']:.4f})

Level distribution:
{result_df['level'].value_counts().to_string()}
"""
        st.download_button(
            "Download report",
            report,
            file_name="msr_port_ccdm_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

