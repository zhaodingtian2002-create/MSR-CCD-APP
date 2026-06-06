import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from glob import glob
import re
import folium
from streamlit_folium import st_folium

# Set page configuration (must be at the top)
st.set_page_config(
    page_title="MSR Coupling Coordination Analysis System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CCDM Model Class
# ============================================================
class CouplingCoordinationModel:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        
    def min_max_normalize(self, data):
        data = np.array(data, dtype=float)
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val - min_val == 0:
            return np.ones_like(data) * 0.5
        return (data - min_val) / (max_val - min_val)
    
    def entropy_weight_method(self, data_matrix):
        n_samples, n_indicators = data_matrix.shape
        p_matrix = data_matrix / (np.sum(data_matrix, axis=0) + 1e-10)
        p_matrix = np.clip(p_matrix, 1e-10, 1 - 1e-10)
        e_j = - (1 / np.log(n_samples)) * np.sum(p_matrix * np.log(p_matrix), axis=0)
        d_j = 1 - e_j
        weights = d_j / (np.sum(d_j) + 1e-10)
        return weights
    
    def calculate_coupling_degree(self, f_score, g_score):
        denominator = f_score + g_score
        if denominator == 0:
            return 0.0
        return 2 * np.sqrt((f_score * g_score) / (denominator ** 2 + 1e-10))
    
    def calculate_coordination_degree(self, f_score, g_score):
        T = self.alpha * f_score + self.beta * g_score
        C = self.calculate_coupling_degree(f_score, g_score)
        return np.sqrt(C * T)
    
    def fit(self, geo_data, socio_data):
        geo_df = geo_data if isinstance(geo_data, pd.DataFrame) else pd.DataFrame(geo_data)
        socio_df = socio_data if isinstance(socio_data, pd.DataFrame) else pd.DataFrame(socio_data)
        
        geo_norm = geo_df.apply(lambda x: self.min_max_normalize(x))
        socio_norm = socio_df.apply(lambda x: self.min_max_normalize(x))
        
        geo_weights = self.entropy_weight_method(geo_norm.values)
        socio_weights = self.entropy_weight_method(socio_norm.values)
        
        f_scores = np.sum(geo_norm.values * geo_weights, axis=1)
        g_scores = np.sum(socio_norm.values * socio_weights, axis=1)
        
        C_values = np.array([self.calculate_coupling_degree(f, g) for f, g in zip(f_scores, g_scores)])
        D_values = np.array([self.calculate_coordination_degree(f, g) for f, g in zip(f_scores, g_scores)])
        
        return {
            'f_scores': f_scores, 'g_scores': g_scores,
            'coupling_C': C_values, 'coordination_D': D_values,
            'geo_weights': geo_weights, 'socio_weights': socio_weights,
            'geo_columns': list(geo_df.columns), 'socio_columns': list(socio_df.columns),
            'geo_norm': geo_norm, 'socio_norm': socio_norm
        }
    
    @staticmethod
    def get_level(D):
        if D >= 0.9: return "Excellent Coordination"
        if D >= 0.8: return "Good Coordination"
        if D >= 0.7: return "Moderate Coordination"
        if D >= 0.6: return "Primary Coordination"
        if D >= 0.5: return "Barely Coordination"
        if D >= 0.4: return "Near Disorder"
        if D >= 0.3: return "Mild Disorder"
        if D >= 0.2: return "Moderate Disorder"
        return "Severe Disorder"


# ============================================================
# Data Loading Function (with caching)
# ============================================================
@st.cache_data
def load_data(folder_path):
    """Load all MSR CSV files"""
    files = glob(os.path.join(folder_path, "MSR_*.csv"))
    
    if not files:
        return None, None, None
    
    natural_list = []
    human_list = []
    all_countries = set()
    
    natural_keywords = ['slope', 'elev', 'bathy', 'evi', 'rugged', 'coast', 'terrain', 'ndvi']
    
    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.search(r'MSR_(\w+)\.csv', filename, re.IGNORECASE)
        if not match:
            continue
        
        indicator = match.group(1).lower()
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'country_na' not in df.columns or 'mean' not in df.columns:
            continue
        
        df['country_na'] = df['country_na'].astype(str).str.strip()
        df = df[['country_na', 'mean']].copy()
        df = df.rename(columns={'mean': indicator})
        df = df.groupby('country_na', as_index=False).mean()
        
        all_countries.update(df['country_na'].tolist())
        
        is_natural = any(kw in indicator.lower() for kw in natural_keywords)
        if is_natural:
            natural_list.append(df)
        else:
            human_list.append(df)
    
    all_countries = sorted(list(all_countries))
    
    # Merge data
    natural_df = pd.DataFrame({'country_na': all_countries})
    for df in natural_list:
        natural_df = natural_df.merge(df, on='country_na', how='left')
    
    human_df = pd.DataFrame({'country_na': all_countries})
    for df in human_list:
        human_df = human_df.merge(df, on='country_na', how='left')
    
    return natural_df, human_df, all_countries


@st.cache_data
def run_ccdm_model(geo_df, socio_df, alpha, beta):
    """Run CCDM model and cache results"""
    model = CouplingCoordinationModel(alpha=alpha, beta=beta)
    results = model.fit(geo_df, socio_df)
    
    # Build results table
    levels = [model.get_level(d) for d in results['coordination_D']]
    result_df = pd.DataFrame({
        'country': geo_df.index,
        'geo_score': results['f_scores'],
        'socio_score': results['g_scores'],
        'coupling_C': results['coupling_C'],
        'coordination_D': results['coordination_D'],
        'level': levels
    }).sort_values('coordination_D', ascending=False).reset_index(drop=True)
    
    return results, result_df, model


# ============================================================
# Visualization Function (with session_state persistence)
# ============================================================
def create_dashboard(results, result_df, natural_df, human_df):
    """Create interactive dashboard"""
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌍 Number of Countries", len(result_df))
    with col2:
        st.metric("📊 Mean Coordination D", f"{result_df['coordination_D'].mean():.3f}")
    with col3:
        st.metric("📈 Max Coordination D", f"{result_df['coordination_D'].max():.3f}")
    with col4:
        st.metric("📉 Min Coordination D", f"{result_df['coordination_D'].min():.3f}")
    #tab layout
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Ranking", 
        "🗺️ Scatter Analysis", 
        "📈 Weight Analysis", 
        "🔬 Detailed Data", 
        "📋 Raw Data",
        "🗺️ Interactive Map"  # 新增
    ])
    
    # ========== Tab 1: Ranking ==========
    with tab1:
        st.subheader("🏆 Coupling Coordination Ranking")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            display_df = result_df[['country', 'coordination_D', 'level', 'geo_score', 'socio_score']].copy()
            display_df.columns = ['Country', 'Coordination D', 'Level', 'Geo Score', 'Socio Score']
            display_df = display_df.round(3)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Coordination D": st.column_config.ProgressColumn(
                        "Coordination D",
                        help="Coupling coordination degree value (0-1)",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
        
        with col_right:
            level_counts = result_df['level'].value_counts()
            colors = ['#2ecc71', '#27ae60', '#f1c40f', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad', '#34495e']
            fig_pie = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title="Coordination Level Distribution",
                color_discrete_sequence=colors,
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("📊 Coordination D Ranking Bar Chart")
        top_n = st.slider("Display Top N Countries", 5, 30, 15, key="top_n_slider")
        
        top_countries = result_df.head(top_n)
        bar_colors = ['#2ecc71' if d >= 0.6 else '#f39c12' if d >= 0.4 else '#e74c3c' 
                      for d in top_countries['coordination_D']]
        
        fig_bar = go.Figure(go.Bar(
            x=top_countries['coordination_D'],
            y=top_countries['country'],
            orientation='h',
            marker_color=bar_colors,
            text=top_countries['coordination_D'].round(3),
            textposition='outside'
        ))
        fig_bar.update_layout(
            title=f"Top {top_n} Countries Coordination Ranking",
            xaxis_title="Coupling Coordination Degree D",
            yaxis_title="Country",
            height=500,
            xaxis_range=[0, 1]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ========== Tab 2: Scatter Analysis ==========
    with tab2:
        st.subheader("🗺️ Geo-Economic System Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            highlight_country = st.selectbox(
                "Highlight Country",
                ["None"] + list(result_df['country'].values),
                key="highlight_select"
            )
        with col2:
            color_by = st.radio("Color By", ["Coordination Level", "Geo Score", "Socio Score"], horizontal=True, key="color_by_radio")
        
        scatter_df = result_df.copy()
        
        if color_by == "Coordination Level":
            fig_scatter = px.scatter(
                scatter_df,
                x='geo_score',
                y='socio_score',
                size='coordination_D',
                color='level',
                hover_name='country',
                text='country',
                title="Geo-Environmental System vs Socio-Economic System",
                labels={'geo_score': 'Geo-Environmental Score f(G)', 'socio_score': 'Socio-Economic Score g(S)'},
                size_max=30,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            color_col = 'geo_score' if color_by == "Geo Score" else 'socio_score'
            fig_scatter = px.scatter(
                scatter_df,
                x='geo_score',
                y='socio_score',
                size='coordination_D',
                color=color_col,
                color_continuous_scale='RdYlGn',
                hover_name='country',
                text='country',
                title="Geo-Environmental System vs Socio-Economic System",
                labels={'geo_score': 'Geo-Environmental Score f(G)', 'socio_score': 'Socio-Economic Score g(S)'},
                size_max=30
            )
        
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        if highlight_country != "None":
            highlight_data = scatter_df[scatter_df['country'] == highlight_country]
            fig_scatter.add_trace(go.Scatter(
                x=highlight_data['geo_score'],
                y=highlight_data['socio_score'],
                mode='markers+text',
                marker=dict(size=40, symbol='star', color='red'),
                text=highlight_data['country'],
                textposition='top center',
                name=f"⭐ {highlight_country}"
            ))
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Quadrant Analysis
        st.subheader("📐 Quadrant Analysis")
        
        quadrant_df = result_df.copy()
        quadrant_df['quadrant'] = np.where(
            (quadrant_df['geo_score'] >= 0.5) & (quadrant_df['socio_score'] >= 0.5), "🟢 High-High (Coordinated Development)",
            np.where((quadrant_df['geo_score'] >= 0.5) & (quadrant_df['socio_score'] < 0.5), "🟡 High-Low (Geo Advantage)",
            np.where((quadrant_df['geo_score'] < 0.5) & (quadrant_df['socio_score'] >= 0.5), "🟠 Low-High (Economic Advantage)",
                     "🔴 Low-Low (Dual Disadvantage)"))
        )
        
        quadrant_counts = quadrant_df['quadrant'].value_counts()
        
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        cols = [col_q1, col_q2, col_q3, col_q4]
        for i, (quad, count) in enumerate(quadrant_counts.items()):
            if i < 4:
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{quad}</h4>
                        <h2>{count}</h2>
                        <p>Countries</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with st.expander("View Quadrant Country Details"):
            for quad in quadrant_df['quadrant'].unique():
                st.markdown(f"**{quad}**")
                countries_in_quad = quadrant_df[quadrant_df['quadrant'] == quad]['country'].tolist()
                st.write(", ".join(countries_in_quad))
    
    # ========== Tab 3: Weight Analysis ==========
    with tab3:
        st.subheader("⚖️ Indicator Weight Analysis")
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            geo_weights_df = pd.DataFrame({
                'Indicator': results['geo_columns'],
                'Weight': results['geo_weights']
            }).sort_values('Weight', ascending=True)
            
            fig_geo = px.bar(
                geo_weights_df,
                x='Weight',
                y='Indicator',
                orientation='h',
                title='Geo-Environmental System Weights',
                color='Weight',
                color_continuous_scale='Blues',
                text='Weight'
            )
            fig_geo.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_geo.update_layout(height=400)
            st.plotly_chart(fig_geo, use_container_width=True)
        
        with col_w2:
            socio_weights_df = pd.DataFrame({
                'Indicator': results['socio_columns'],
                'Weight': results['socio_weights']
            }).sort_values('Weight', ascending=True)
            
            fig_socio = px.bar(
                socio_weights_df,
                x='Weight',
                y='Indicator',
                orientation='h',
                title='Socio-Economic System Weights',
                color='Weight',
                color_continuous_scale='Reds',
                text='Weight'
            )
            fig_socio.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_socio.update_layout(height=400)
            st.plotly_chart(fig_socio, use_container_width=True)
        
        st.subheader("📊 System Weight Comparison")
        comparison_df = pd.DataFrame({
            'Indicator': results['geo_columns'] + results['socio_columns'],
            'Weight': list(results['geo_weights']) + list(results['socio_weights']),
            'System': ['Geo System'] * len(results['geo_columns']) + ['Socio System'] * len(results['socio_columns'])
        }).sort_values('Weight', ascending=False).head(10)
        
        fig_compare = px.bar(
            comparison_df,
            x='Weight',
            y='Indicator',
            color='System',
            orientation='h',
            title='Top 10 Influential Indicators',
            color_discrete_map={'Geo System': '#3498db', 'Socio System': '#e74c3c'},
            text='Weight'
        )
        fig_compare.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_compare.update_layout(height=450)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # ========== Tab 4: Detailed Data ==========
    with tab4:
        st.subheader("🔬 Detailed Indicator Data")
        
        # Get country list
        country_list = result_df['country'].values.tolist()
        
        # Use session_state to keep selected country
        if 'selected_country' not in st.session_state:
            st.session_state.selected_country = country_list[0] if country_list else None
        
        # Create two-column layout
        col_select, col_empty = st.columns([2, 3])
        
        with col_select:
            # Country selector
            selected_country = st.selectbox(
                "Select Country to View Details",
                country_list,
                index=country_list.index(st.session_state.selected_country) if st.session_state.selected_country in country_list else 0,
                key="country_selector",
                on_change=None
            )
            # Update session_state
            st.session_state.selected_country = selected_country
        
        # Get selected country data
        country_data = result_df[result_df['country'] == selected_country].iloc[0]
        
        # Display metric cards
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("🌍 Coupling Coordination D", f"{country_data['coordination_D']:.4f}")
            st.metric("🔗 Coupling Degree C", f"{country_data['coupling_C']:.4f}")
        with col_d2:
            st.metric("🗻 Geo-Environmental Score f(G)", f"{country_data['geo_score']:.4f}")
            st.metric("💼 Socio-Economic Score g(S)", f"{country_data['socio_score']:.4f}")
        with col_d3:
            st.metric("📊 Coordination Type", country_data['level'])
            rank = int(result_df[result_df['country'] == selected_country].index[0]) + 1
            st.metric("📈 Rank", f"No. {rank}")
        
        # Radar chart
        st.subheader(f"📡 {selected_country} Indicator Radar Chart")
        
        # Get normalized data
        geo_norm_df = results['geo_norm']
        socio_norm_df = results['socio_norm']
        
        if selected_country in geo_norm_df.index:
            geo_values = geo_norm_df.loc[selected_country].values
            socio_values = socio_norm_df.loc[selected_country].values
        else:
            geo_values = geo_norm_df.iloc[0].values
            socio_values = socio_norm_df.iloc[0].values
        
        fig_radar = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Geo-Environmental Indicators", "Socio-Economic Indicators"),
            specs=[[{'type': 'polar'}, {'type': 'polar'}]]
        )
        
        fig_radar.add_trace(go.Scatterpolar(
            r=geo_values,
            theta=results['geo_columns'],
            fill='toself',
            name=selected_country,
            line_color='#3498db',
            fillcolor='rgba(52, 152, 219, 0.3)'
        ), row=1, col=1)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=socio_values,
            theta=results['socio_columns'],
            fill='toself',
            name=selected_country,
            line_color='#e74c3c',
            fillcolor='rgba(231, 76, 60, 0.3)'
        ), row=1, col=2)
        
        fig_radar.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Comparison with average
        st.subheader("📊 Comparison with Average")
        compare_data = pd.DataFrame({
            'Indicator': ['Coordination D', 'Geo Score', 'Socio Score', 'Coupling C'],
            selected_country: [
                country_data['coordination_D'],
                country_data['geo_score'],
                country_data['socio_score'],
                country_data['coupling_C']
            ],
            'Average': [
                result_df['coordination_D'].mean(),
                result_df['geo_score'].mean(),
                result_df['socio_score'].mean(),
                result_df['coupling_C'].mean()
            ]
        })
        
        fig_compare_country = go.Figure()
        fig_compare_country.add_trace(go.Bar(
            name=selected_country,
            x=compare_data['Indicator'],
            y=compare_data[selected_country],
            marker_color='#3498db'
        ))
        fig_compare_country.add_trace(go.Bar(
            name='Average',
            x=compare_data['Indicator'],
            y=compare_data['Average'],
            marker_color='#95a5a6'
        ))
        fig_compare_country.update_layout(
            title=f"{selected_country} vs All Countries Average",
            yaxis_title="Value",
            barmode='group'
        )
        st.plotly_chart(fig_compare_country, use_container_width=True)
    
    # ========== Tab 5: Raw Data ==========
    with tab5:
        st.subheader("📋 Raw Data")
        
        if natural_df is not None and human_df is not None:
            merged_raw = natural_df.merge(human_df, on='country_na', how='outer')
            merged_raw = merged_raw.fillna(0)
            st.dataframe(merged_raw, use_container_width=True)
            
            csv = merged_raw.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Download Raw Data (CSV)",
                data=csv,
                file_name="msr_raw_data.csv",
                mime="text/csv"
            )
        # ========== Tab 6: Interactive Map ==========
    with tab6:
        st.subheader("🗺️ Interactive CCD Map")
        st.markdown("Click on any marker to view detailed CCD information")
        
        # 根据 CCD 值分类
        def classify_ccd(d):
            if d >= 0.7:
                return "High Coordination (D ≥ 0.7)", "#2ecc71"
            elif d >= 0.5:
                return "Medium Coordination (0.5 ≤ D < 0.7)", "#f39c12"
            else:
                return "Low Coordination (D < 0.5)", "#e74c3c"
        
        # 添加分类列
        map_df = result_df.copy()
        map_df['ccd_category'], map_df['color'] = zip(*map_df['coordination_D'].apply(classify_ccd))
        
        # 加载坐标文件
        coord_file = os.path.join(data_path, "countries_coordinates.csv")
        
        if os.path.exists(coord_file):
            coords_df = pd.read_csv(coord_file)
            
            # 合并数据
            map_data = map_df.merge(coords_df, left_on='country', right_on='country', how='inner')
            
            if len(map_data) > 0:
                # 创建地图
                m = folium.Map(
                    location=[20, 0], 
                    zoom_start=2, 
                    tiles='CartoDB positron',
                    control_scale=True
                )
                
                # 添加标记点
                for _, row in map_data.iterrows():
                    # 计算标记大小（基于 CCD 值）
                    radius = 6 + row['coordination_D'] * 12
                    
                    # 创建弹出信息
                    popup_html = f"""
                    <div style="width: 260px; padding: 12px;">
                        <h4 style="color: {row['color']}; margin-bottom: 8px;">{row['country']}</h4>
                        <hr style="margin: 5px 0;">
                        <table style="width: 100%; font-size: 13px;">
                            <tr><td><b>CCD Value:</b></td>
                            <td style="text-align: right;"><b>{row['coordination_D']:.4f}</b></td>
                            </tr>
                            <tr><td><b>Category:</b></td>
                            <td style="text-align: right; color: {row['color']};">{row['ccd_category']}</td>
                            </tr>
                            <tr><td><b>Geo Score:</b></td>
                            <td style="text-align: right;">{row['geo_score']:.3f}</td>
                            </tr>
                            <tr><td><b>Socio Score:</b></td>
                            <td style="text-align: right;">{row['socio_score']:.3f}</td>
                            </tr>
                            <tr><td><b>Coupling C:</b></td>
                            <td style="text-align: right;">{row['coupling_C']:.3f}</td>
                            </tr>
                        </table>
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        popup=folium.Popup(popup_html, max_width=300),
                        color=row['color'],
                        fill=True,
                        fill_color=row['color'],
                        fill_opacity=0.7,
                        weight=2,
                        tooltip=f"{row['country']}: D={row['coordination_D']:.3f}"
                    ).add_to(m)
                
                # 添加图例
                legend_html = '''
                <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000; 
                            background-color: white; padding: 12px 18px; border-radius: 8px; 
                            box-shadow: 0 2px 10px rgba(0,0,0,0.2); border-left: 4px solid #333;
                            font-size: 13px;">
                    <b>📊 CCD Classification</b><br>
                    <span style="color: #2ecc71;">●</span> High Coordination (D ≥ 0.7)<br>
                    <span style="color: #f39c12;">●</span> Medium Coordination (0.5 ≤ D < 0.7)<br>
                    <span style="color: #e74c3c;">●</span> Low Coordination (D < 0.5)<br>
                    <hr style="margin: 6px 0;">
                    <span style="font-size: 11px;">● Marker size = CCD value</span>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
                # 显示地图
                st_folium(m, width=800, height=550, returned_objects=[])
                
                # 显示统计
                st.markdown("---")
                st.subheader("📊 CCD Category Summary")
                
                col1, col2, col3 = st.columns(3)
                counts = map_data['ccd_category'].value_counts()
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:#2ecc71; padding:15px; border-radius:10px; color:white; text-align:center">
                        <h2>{counts.get('High Coordination (D ≥ 0.7)', 0)}</h2>
                        <p>High Coordination</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color:#f39c12; padding:15px; border-radius:10px; color:white; text-align:center">
                        <h2>{counts.get('Medium Coordination (0.5 ≤ D < 0.7)', 0)}</h2>
                        <p>Medium Coordination</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background-color:#e74c3c; padding:15px; border-radius:10px; color:white; text-align:center">
                        <h2>{counts.get('Low Coordination (D < 0.5)', 0)}</h2>
                        <p>Low Coordination</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 显示国家列表
                with st.expander("📋 View Countries by Category"):
                    for category in ['High Coordination (D ≥ 0.7)', 'Medium Coordination (0.5 ≤ D < 0.7)', 'Low Coordination (D < 0.5)']:
                        countries_in_cat = map_data[map_data['ccd_category'] == category]['country'].tolist()
                        if countries_in_cat:
                            st.markdown(f"**{category}** ({len(countries_in_cat)} countries)")
                            st.write(", ".join(countries_in_cat))
                            st.markdown("---")
                
            else:
                st.warning("No matching countries found between results and coordinates file.")
                st.info("Please ensure country names in the coordinates file match the results.")
        
        else:
            # 如果坐标文件不存在，显示提示并提供下载
            st.warning("⚠️ Countries coordinates file not found!")
            
            st.markdown("""
            ### How to set up the interactive map:
            
            1. **Download the template** below
            2. **Fill in coordinates** for each country (get from Google Maps or [latlong.net](https://www.latlong.net/))
            3. **Save as** `countries_coordinates.csv` in your data folder
            4. **Rerun** the analysis
            
            ### Sample coordinate format:
            ```csv
            country,latitude,longitude
            China,35.0,105.0
            India,20.0,77.0
            Indonesia,-0.8,113.2

# ============================================================
# Main Program
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>"🌊" 21st Century Maritime Silk Road</h1>
        <h2>Coupling Coordination Analysis System (CCDM)</h2>
        <p>Geo-Environmental System | Socio-Economic System | Coordinated Development Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/earth-planet.png", width=80)
        st.title("⚙️ Control Panel")
        
        # Data path
        data_path = st.text_input(
            "📁 Data Folder Path",
            value=".",
            help="Folder containing MSR_*.csv files"
        )
        
        # Model parameters
        st.subheader("🎛️ Model Parameters")
        alpha = st.slider("Geo-Environmental System Weight α", 0.0, 1.0, 0.5, 0.1, key="alpha_slider")
        beta = st.slider("Socio-Economic System Weight β", 0.0, 1.0, 0.5, 0.1, key="beta_slider")
        
        # Run button
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        **📖 Instructions**
        - D ≥ 0.7: Good Coordination
        - 0.5 ≤ D < 0.7: Barely Coordination  
        - D < 0.5: Disorder State
        """)
    
    # Use session_state to store analysis results
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    # Main content
    if run_button or st.session_state.analysis_done:
        if run_button:
            st.session_state.analysis_done = True
        
        with st.spinner("Loading data and running analysis..."):
            try:
                # Load data
                natural_df, human_df, countries = load_data(data_path)
                
                if natural_df is None or human_df is None or len(natural_df) == 0:
                    st.error("❌ Data files not found! Please ensure the folder contains MSR_*.csv files")
                    st.info("Expected file name format: MSR_slope.csv, MSR_gdp.csv, etc.")
                    st.session_state.analysis_done = False
                    return
                
                # Prepare model input
                geo_df = natural_df.set_index('country_na').fillna(0)
                socio_df = human_df.set_index('country_na').fillna(0)
                
                # Run model (with caching)
                results, result_df, model = run_ccdm_model(geo_df, socio_df, alpha, beta)
                
                # Store in session_state for recovery after rerun
                st.session_state.results = results
                st.session_state.result_df = result_df
                st.session_state.natural_df = natural_df
                st.session_state.human_df = human_df
                
                # Display success message
                st.success(f"✅ Analysis complete! Total {len(result_df)} countries analyzed")
                
                # Create dashboard
                create_dashboard(results, result_df, natural_df, human_df)
                
                # Download results
                st.markdown("---")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    csv_results = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download Analysis Results (CSV)",
                        data=csv_results,
                        file_name="msr_ccdm_results.csv",
                        mime="text/csv"
                    )
                with col_d2:
                    report = f"""
                    MSR Coupling Coordination Analysis Report
                    ==========================================
                    Total Countries Analyzed: {len(result_df)}
                    Mean Coordination D: {result_df['coordination_D'].mean():.4f}
                    Std Coordination D: {result_df['coordination_D'].std():.4f}
                    Max Coordination D: {result_df.iloc[0]['country']} ({result_df.iloc[0]['coordination_D']:.4f})
                    Min Coordination D: {result_df.iloc[-1]['country']} ({result_df.iloc[-1]['coordination_D']:.4f})
                    
                    Level Distribution:
                    {result_df['level'].value_counts().to_string()}
                    """
                    st.download_button(
                        label="📄 Download Analysis Report (TXT)",
                        data=report,
                        file_name="msr_ccdm_report.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.exception(e)
                st.session_state.analysis_done = False
    else:
        # Initial state
        st.info("👈 Please set the data folder path on the left, then click 'Run Analysis'")
        
        with st.expander("📖 Instructions"):
            st.markdown("""
            ### Data Preparation
            1. Name your data files in the format: `MSR_INDICATOR_NAME.csv`
            2. Each CSV file must contain two columns:
               - `country_na`: Country name
               - `mean`: Indicator mean value
            3. Place all files in the same folder
            
            ### Supported File Examples
            - `MSR_slope.csv` (Slope)
            - `MSR_elevation.csv` (Elevation)
            - `MSR_bathymetry.csv` (Bathymetry)
            - `MSR_evi.csv` (EVI - Vegetation Index)
            - `MSR_gdp.csv` (GDP)
            - `MSR_lsci.csv` (Liner Shipping Connectivity Index)
            - `MSR_population.csv` (Population Density)
            
            ### Output Results
            - Coupling coordination ranking
            - Weight analysis
            - Quadrant analysis
            - Detailed country reports
            """)
        
        if os.path.exists("."):
            csv_files = glob("MSR_*.csv")
            if csv_files:
                st.success(f"Found {len(csv_files)} data files in current directory:")
                for f in csv_files:
                    st.write(f"  - {f}")
            else:
                st.warning("No MSR_*.csv files found in current directory")


if __name__ == "__main__":
    main()
