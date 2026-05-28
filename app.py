import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from glob import glob
import re

# 设置页面配置（必须在最前面）
st.set_page_config(
    page_title="MSR 耦合协调度分析系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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
# CCDM 模型类
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
        if D >= 0.9: return "优质协调"
        if D >= 0.8: return "良好协调"
        if D >= 0.7: return "中级协调"
        if D >= 0.6: return "初级协调"
        if D >= 0.5: return "勉强协调"
        if D >= 0.4: return "濒临失调"
        if D >= 0.3: return "轻度失调"
        if D >= 0.2: return "中度失调"
        return "严重失调"


# ============================================================
# 数据加载函数（使用缓存）
# ============================================================
@st.cache_data
def load_data(folder_path):
    """加载所有MSR CSV文件"""
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
    
    # 合并数据
    natural_df = pd.DataFrame({'country_na': all_countries})
    for df in natural_list:
        natural_df = natural_df.merge(df, on='country_na', how='left')
    
    human_df = pd.DataFrame({'country_na': all_countries})
    for df in human_list:
        human_df = human_df.merge(df, on='country_na', how='left')
    
    return natural_df, human_df, all_countries


@st.cache_data
def run_ccdm_model(geo_df, socio_df, alpha, beta):
    """运行CCDM模型并缓存结果"""
    model = CouplingCoordinationModel(alpha=alpha, beta=beta)
    results = model.fit(geo_df, socio_df)
    
    # 构建结果表
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
# 可视化函数（使用 session_state 保持状态）
# ============================================================
def create_dashboard(results, result_df, natural_df, human_df):
    """创建交互式仪表板"""
    
    # 指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌍 分析国家 & 地区数", len(result_df))
    with col2:
        st.metric("📊 协调度均值", f"{result_df['coordination_D'].mean():.3f}")
    with col3:
        st.metric("📈 最高协调度", f"{result_df['coordination_D'].max():.3f}")
    with col4:
        st.metric("📉 最低协调度", f"{result_df['coordination_D'].min():.3f}")
    
    # Tab布局
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 综合排名", "🗺️ 散点分析", "📈 权重分析", "🔬 详细数据", "📋 原始数据"
    ])
    
    # ========== Tab 1: 综合排名 ==========
    with tab1:
        st.subheader("🏆 耦合协调度排名")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            display_df = result_df[['country', 'coordination_D', 'level', 'geo_score', 'socio_score']].copy()
            display_df.columns = ['国家 & 地区', '协调度D', '等级', '地理得分', '经济得分']
            display_df = display_df.round(3)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "协调度D": st.column_config.ProgressColumn(
                        "协调度D",
                        help="耦合协调度值 (0-1)",
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
                title="协调等级分布",
                color_discrete_sequence=colors,
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("📊 协调度排名条形图")
        top_n = st.slider("显示前N个国家 & 地区", 5, 30, 15, key="top_n_slider")
        
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
            title=f"Top {top_n} 国家 & 地区协调度排名",
            xaxis_title="耦合协调度 D",
            yaxis_title="国家 & 地区",
            height=500,
            xaxis_range=[0, 1]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ========== Tab 2: 散点分析 ==========
    with tab2:
        st.subheader("🗺️ 地理-经济系统分析")
        
        col1, col2 = st.columns(2)
        with col1:
            highlight_country = st.selectbox(
                "高亮显示国家 & 地区",
                ["无"] + list(result_df['country'].values),
                key="highlight_select"
            )
        with col2:
            color_by = st.radio("颜色标记", ["协调等级", "地理得分", "经济得分"], horizontal=True, key="color_by_radio")
        
        scatter_df = result_df.copy()
        
        if color_by == "协调等级":
            fig_scatter = px.scatter(
                scatter_df,
                x='geo_score',
                y='socio_score',
                size='coordination_D',
                color='level',
                hover_name='country',
                text='country',
                title="地理环境系统 vs 社会经济系统",
                labels={'geo_score': '地理环境系统得分 f(G)', 'socio_score': '社会经济系统得分 g(S)'},
                size_max=30,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            color_col = 'geo_score' if color_by == "地理得分" else 'socio_score'
            fig_scatter = px.scatter(
                scatter_df,
                x='geo_score',
                y='socio_score',
                size='coordination_D',
                color=color_col,
                color_continuous_scale='RdYlGn',
                hover_name='country',
                text='country',
                title="地理环境系统 vs 社会经济系统",
                labels={'geo_score': '地理环境系统得分 f(G)', 'socio_score': '社会经济系统得分 g(S)'},
                size_max=30
            )
        
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        if highlight_country != "无":
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
        
        # 四象限分析
        st.subheader("📐 四象限分析")
        
        quadrant_df = result_df.copy()
        quadrant_df['quadrant'] = np.where(
            (quadrant_df['geo_score'] >= 0.5) & (quadrant_df['socio_score'] >= 0.5), "🟢 高-高 (协调发展)",
            np.where((quadrant_df['geo_score'] >= 0.5) & (quadrant_df['socio_score'] < 0.5), "🟡 高-低 (地理优势)",
            np.where((quadrant_df['geo_score'] < 0.5) & (quadrant_df['socio_score'] >= 0.5), "🟠 低-高 (经济优势)",
                     "🔴 低-低 (双重劣势)"))
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
                        <p>个国家 & 地区</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with st.expander("查看四象限国家 & 地区详情"):
            for quad in quadrant_df['quadrant'].unique():
                st.markdown(f"**{quad}**")
                countries_in_quad = quadrant_df[quadrant_df['quadrant'] == quad]['country'].tolist()
                st.write(", ".join(countries_in_quad))
    
    # ========== Tab 3: 权重分析 ==========
    with tab3:
        st.subheader("⚖️ 指标权重分析")
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            geo_weights_df = pd.DataFrame({
                '指标': results['geo_columns'],
                '权重': results['geo_weights']
            }).sort_values('权重', ascending=True)
            
            fig_geo = px.bar(
                geo_weights_df,
                x='权重',
                y='指标',
                orientation='h',
                title='地理环境系统指标权重',
                color='权重',
                color_continuous_scale='Blues',
                text='权重'
            )
            fig_geo.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_geo.update_layout(height=400)
            st.plotly_chart(fig_geo, use_container_width=True)
        
        with col_w2:
            socio_weights_df = pd.DataFrame({
                '指标': results['socio_columns'],
                '权重': results['socio_weights']
            }).sort_values('权重', ascending=True)
            
            fig_socio = px.bar(
                socio_weights_df,
                x='权重',
                y='指标',
                orientation='h',
                title='社会经济系统指标权重',
                color='权重',
                color_continuous_scale='Reds',
                text='权重'
            )
            fig_socio.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_socio.update_layout(height=400)
            st.plotly_chart(fig_socio, use_container_width=True)
        
        st.subheader("📊 系统权重对比")
        comparison_df = pd.DataFrame({
            '指标': results['geo_columns'] + results['socio_columns'],
            '权重': list(results['geo_weights']) + list(results['socio_weights']),
            '系统': ['地理系统'] * len(results['geo_columns']) + ['经济系统'] * len(results['socio_columns'])
        }).sort_values('权重', ascending=False).head(10)
        
        fig_compare = px.bar(
            comparison_df,
            x='权重',
            y='指标',
            color='系统',
            orientation='h',
            title='Top 10 影响指标',
            color_discrete_map={'地理系统': '#3498db', '经济系统': '#e74c3c'},
            text='权重'
        )
        fig_compare.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_compare.update_layout(height=450)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # ========== Tab 4: 详细数据（修复闪退问题）==========
    with tab4:
        st.subheader("🔬 详细指标数据")
        
        # 获取国家列表
        country_list = result_df['country'].values.tolist()
        
        # 使用 session_state 保持选中的国家
        if 'selected_country' not in st.session_state:
            st.session_state.selected_country = country_list[0] if country_list else None
        
        # 创建两列布局
        col_select, col_empty = st.columns([2, 3])
        
        with col_select:
            # 国家选择器
            selected_country = st.selectbox(
                "选择国家 & 地区查看详情",
                country_list,
                index=country_list.index(st.session_state.selected_country) if st.session_state.selected_country in country_list else 0,
                key="country_selector",
                on_change=None
            )
            # 更新 session_state
            st.session_state.selected_country = selected_country
        
        # 获取选中国家的数据
        country_data = result_df[result_df['country'] == selected_country].iloc[0]
        
        # 显示指标卡片
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("🌍 耦合协调度 D", f"{country_data['coordination_D']:.4f}")
            st.metric("🔗 耦合度 C", f"{country_data['coupling_C']:.4f}")
        with col_d2:
            st.metric("🗻 地理环境得分 f(G)", f"{country_data['geo_score']:.4f}")
            st.metric("💼 社会经济得分 g(S)", f"{country_data['socio_score']:.4f}")
        with col_d3:
            st.metric("📊 协调类型", country_data['level'])
            rank = int(result_df[result_df['country'] == selected_country].index[0]) + 1
            st.metric("📈 排名", f"第 {rank} 名")
        
        # 雷达图
        st.subheader(f"📡 {selected_country} 指标雷达图")
        
        # 获取归一化数据
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
            subplot_titles=("地理环境指标", "社会经济指标"),
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
        
        # 与平均值的对比
        st.subheader("📊 与平均值对比")
        compare_data = pd.DataFrame({
            '指标': ['协调度D', '地理得分', '经济得分', '耦合度C'],
            selected_country: [
                country_data['coordination_D'],
                country_data['geo_score'],
                country_data['socio_score'],
                country_data['coupling_C']
            ],
            '平均值': [
                result_df['coordination_D'].mean(),
                result_df['geo_score'].mean(),
                result_df['socio_score'].mean(),
                result_df['coupling_C'].mean()
            ]
        })
        
        fig_compare_country = go.Figure()
        fig_compare_country.add_trace(go.Bar(
            name=selected_country,
            x=compare_data['指标'],
            y=compare_data[selected_country],
            marker_color='#3498db'
        ))
        fig_compare_country.add_trace(go.Bar(
            name='平均值',
            x=compare_data['指标'],
            y=compare_data['平均值'],
            marker_color='#95a5a6'
        ))
        fig_compare_country.update_layout(
            title=f"{selected_country} vs 全部国家 & 地区平均值",
            yaxis_title="数值",
            barmode='group'
        )
        st.plotly_chart(fig_compare_country, use_container_width=True)
    
    # ========== Tab 5: 原始数据 ==========
    with tab5:
        st.subheader("📋 原始数据")
        
        if natural_df is not None and human_df is not None:
            merged_raw = natural_df.merge(human_df, on='country_na', how='outer')
            merged_raw = merged_raw.fillna(0)
            st.dataframe(merged_raw, use_container_width=True)
            
            csv = merged_raw.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载原始数据 (CSV)",
                data=csv,
                file_name="msr_raw_data.csv",
                mime="text/csv"
            )


# ============================================================
# 主程序
# ============================================================
def main():
    # 标题
    st.markdown("""
    <div class="main-header">
        <h1>🌊 21世纪海上丝绸之路</h1>
        <h2>耦合协调度分析系统 (CCDM)</h2>
        <p>地理环境系统 | 社会经济系统 | 协同发展评估</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/earth-planet.png", width=80)
        st.title("⚙️ 控制面板")
        
        # 数据路径
        data_path = st.text_input(
            "📁 数据文件夹路径",
            value=".",
            help="包含 MSR_*.csv 文件的文件夹路径"
        )
        
        # 模型参数
        st.subheader("🎛️ 模型参数")
        alpha = st.slider("地理环境系统权重 α", 0.0, 1.0, 0.5, 0.1, key="alpha_slider")
        beta = st.slider("社会经济系统权重 β", 0.0, 1.0, 0.5, 0.1, key="beta_slider")
        
        # 运行按钮
        run_button = st.button("🚀 运行分析", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        **📖 说明**
        - D ≥ 0.7: 良好协调
        - 0.5 ≤ D < 0.7: 勉强协调  
        - D < 0.5: 失调状态
        """)
    
    # 使用 session_state 存储分析结果
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    # 主内容
    if run_button or st.session_state.analysis_done:
        if run_button:
            st.session_state.analysis_done = True
        
        with st.spinner("正在加载数据并运行分析..."):
            try:
                # 加载数据
                natural_df, human_df, countries = load_data(data_path)
                
                if natural_df is None or human_df is None or len(natural_df) == 0:
                    st.error("❌ 未找到数据文件！请确保文件夹中包含 MSR_*.csv 文件")
                    st.info("期望的文件名格式: MSR_slope.csv, MSR_gdp.csv 等")
                    st.session_state.analysis_done = False
                    return
                
                # 准备模型输入
                geo_df = natural_df.set_index('country_na').fillna(0)
                socio_df = human_df.set_index('country_na').fillna(0)
                
                # 运行模型（使用缓存）
                results, result_df, model = run_ccdm_model(geo_df, socio_df, alpha, beta)
                
                # 存储到 session_state 以便在闪退后恢复
                st.session_state.results = results
                st.session_state.result_df = result_df
                st.session_state.natural_df = natural_df
                st.session_state.human_df = human_df
                
                # 显示成功消息
                st.success(f"✅ 分析完成！共分析 {len(result_df)} 个国家 & 地区")
                
                # 创建仪表板
                create_dashboard(results, result_df, natural_df, human_df)
                
                # 下载结果
                st.markdown("---")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    csv_results = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载分析结果 (CSV)",
                        data=csv_results,
                        file_name="msr_ccdm_results.csv",
                        mime="text/csv"
                    )
                with col_d2:
                    report = f"""
                    MSR 耦合协调度分析报告
                    ========================
                    分析国家 & 地区数: {len(result_df)}
                    协调度均值: {result_df['coordination_D'].mean():.4f}
                    协调度标准差: {result_df['coordination_D'].std():.4f}
                    最高协调度: {result_df.iloc[0]['country']} ({result_df.iloc[0]['coordination_D']:.4f})
                    最低协调度: {result_df.iloc[-1]['country']} ({result_df.iloc[-1]['coordination_D']:.4f})
                    
                    等级分布:
                    {result_df['level'].value_counts().to_string()}
                    """
                    st.download_button(
                        label="📄 下载分析报告 (TXT)",
                        data=report,
                        file_name="msr_ccdm_report.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ 分析出错: {str(e)}")
                st.exception(e)
                st.session_state.analysis_done = False
    else:
        # 初始状态
        st.info("👈 请在左侧设置数据路径，然后点击「运行分析」")
        
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 数据准备
            1. 确保您的数据文件命名格式为: `MSR_指标名.csv`
            2. 每个CSV文件必须包含两列:
               - `country_na`: 国家 & 地区名称
               - `mean`: 指标平均值
            3. 将文件放在同一个文件夹中
            
            ### 支持的文件示例
            - `MSR_slope.csv` (坡度)
            - `MSR_elevation.csv` (高程)
            - `MSR_bathymetry.csv` (水深)
            - `MSR_evi.csv` (植被指数)
            - `MSR_gdp.csv` (GDP)
            - `MSR_lsci.csv` (航运指数)
            - `MSR_population.csv` (人口密度)
            """)
        
        if os.path.exists("."):
            csv_files = glob("MSR_*.csv")
            if csv_files:
                st.success(f"在当前目录找到 {len(csv_files)} 个数据文件:")
                for f in csv_files:
                    st.write(f"  - {f}")
            else:
                st.warning("当前目录未找到 MSR_*.csv 文件")


if __name__ == "__main__":
    main()
