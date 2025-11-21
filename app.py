import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np
from collections import Counter
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="大数据开发实习岗位分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
        .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="metric-container"] label {color: white !important;}
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {color: white !important;}
        h1 {color: #667eea;}
        h2 {color: #764ba2;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 数据加载与清洗函数
# ============================================================================
@st.cache_data
def load_and_clean_data():
    """加载并清洗数据"""
    df = None
    
    # 尝试多个可能的文件路径
    possible_paths = [
        '../Big_data_development_results.csv',  # 本地开发环境
        'Big_data_development_results.csv',     # Streamlit Cloud
        './Big_data_development_results.csv'    # 当前目录
    ]
    
    # 尝试多种编码和路径读取CSV文件
    for file_path in possible_paths:
        for encoding in ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                continue
        if df is not None:
            break
    
    # 如果所有编码和路径都失败
    if df is None:
        st.error("❌ 无法读取数据文件：Big_data_development_results.csv")
        st.info("💡 请确保数据文件在项目根目录或上级目录中")
        return None
    
    expected_cols = ['职位id', '职位标题', '薪资范围', '公司名称', '工作地点', 
                     '所处行业', '学历要求', '每周天数', '实习时长', '福利待遇',
                     '职位描述', '简历要求', '截止日期', '详细地址', '详情页url']
    
    if len(df.columns) == len(expected_cols):
        df.columns = expected_cols
    
    # 薪资清洗
    def parse_salary(salary_str):
        try:
            nums = re.findall(r'\d+', str(salary_str))
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                return int(nums[0]), int(nums[0])
            else:
                return None, None
        except:
            return None, None
    
    df[['最低薪资', '最高薪资']] = df['薪资范围'].apply(lambda x: pd.Series(parse_salary(x)))
    df['平均薪资'] = (df['最低薪资'] + df['最高薪资']) / 2
    df['平均薪资'] = df['平均薪资'].fillna(0).astype(int)
    
    # 城市提取
    df['城市'] = df['工作地点'].astype(str).apply(
        lambda x: x.split('-')[0] if '-' in x else x.split('·')[0] if '·' in x else x
    )
    
    # 技能提取
    TECH_KEYWORDS = ['Hadoop', 'Spark', 'Flink', 'Python', 'Java', 'SQL', 'Kafka', 
                     'Hive', 'HBase', 'Scala', 'ETL', 'MySQL', 'Redis', 'Elasticsearch',
                     'Docker', 'Kubernetes', 'Linux', 'Shell', 'ClickHouse',
                     '数据仓库', '数据湖', '实时计算', '离线计算', 'MapReduce', 'HDFS']
    
    def extract_skills(desc):
        if pd.isna(desc):
            return []
        desc_upper = str(desc).upper()
        return [skill for skill in TECH_KEYWORDS if skill.upper() in desc_upper]
    
    df['技能标签'] = df['职位描述'].apply(extract_skills)
    
    # 学历标准化
    def standardize_education(edu):
        edu_str = str(edu).lower()
        if '博士' in edu_str:
            return '博士'
        elif '硕士' in edu_str or '研究生' in edu_str:
            return '硕士'
        elif '本科' in edu_str or '学士' in edu_str:
            return '本科'
        elif '大专' in edu_str or '专科' in edu_str:
            return '大专'
        else:
            return '不限'
    
    df['学历分类'] = df['学历要求'].apply(standardize_education)
    
    # 实习时长分类
    def classify_duration(duration):
        duration_str = str(duration)
        if '3' in duration_str and '月' in duration_str:
            return '3个月'
        elif '6' in duration_str and '月' in duration_str:
            return '6个月'
        elif '长期' in duration_str or '灵活' in duration_str:
            return '长期实习'
        else:
            return '其他'
    
    df['实习时长分类'] = df['实习时长'].apply(classify_duration)
    
    # 福利标签提取
    def extract_welfare(welfare_str):
        if pd.isna(welfare_str):
            return []
        tags = re.split(r'[,，;；、\s]+', str(welfare_str))
        return [tag.strip() for tag in tags if tag.strip()]
    
    df['福利标签'] = df['福利待遇'].apply(extract_welfare)
    
    # 删除无效数据
    df = df[(df['平均薪资'] > 0) & (df['城市'].notna()) & (df['城市'] != '')]
    
    return df

# 加载数据
with st.spinner('🔄 正在加载数据...'):
    df = load_and_clean_data()

if df is None or df.empty:
    st.stop()

# ============================================================================
# 侧边栏筛选器
# ============================================================================
st.sidebar.title("🔍 数据筛选器")
st.sidebar.markdown("---")

# 城市筛选
st.sidebar.subheader("📍 工作城市")
all_cities = sorted(df['城市'].unique().tolist())
selected_cities = st.sidebar.multiselect(
    "选择城市",
    all_cities,
    default=all_cities[:10] if len(all_cities) > 10 else all_cities
)

# 薪资范围筛选
st.sidebar.subheader("💰 薪资范围")
min_sal = int(df['平均薪资'].min())
max_sal = int(df['平均薪资'].max())
salary_range = st.sidebar.slider("日薪范围（元/天）", min_sal, max_sal, (min_sal, max_sal))

# 学历筛选
st.sidebar.subheader("🎓 学历要求")
all_edu = df['学历分类'].unique().tolist()
selected_edu = st.sidebar.multiselect("选择学历", all_edu, default=all_edu)

# 技能筛选
st.sidebar.subheader("💻 技能要求")
all_skills = []
for skills in df['技能标签']:
    all_skills.extend(skills)
unique_skills = sorted(list(set(all_skills)))
selected_skills = st.sidebar.multiselect("选择技能（AND逻辑）", unique_skills, default=[])

# 实习时长筛选
st.sidebar.subheader("⏰ 实习时长")
all_durations = df['实习时长分类'].unique().tolist()
selected_durations = st.sidebar.multiselect("选择时长", all_durations, default=all_durations)

# 福利筛选
st.sidebar.subheader("🎁 福利待遇")
all_welfare = []
for welfare in df['福利标签']:
    all_welfare.extend(welfare)
unique_welfare = sorted(list(set(all_welfare)))[:20]
selected_welfare = st.sidebar.multiselect("选择福利", unique_welfare, default=[])

st.sidebar.markdown("---")
st.sidebar.info("💡 提示：所有筛选条件为 AND 关系")

# ============================================================================
# 应用筛选
# ============================================================================
filtered_df = df.copy()

if selected_cities:
    filtered_df = filtered_df[filtered_df['城市'].isin(selected_cities)]

if selected_edu:
    filtered_df = filtered_df[filtered_df['学历分类'].isin(selected_edu)]

if selected_durations:
    filtered_df = filtered_df[filtered_df['实习时长分类'].isin(selected_durations)]

filtered_df = filtered_df[
    (filtered_df['平均薪资'] >= salary_range[0]) &
    (filtered_df['平均薪资'] <= salary_range[1])
]

if selected_skills:
    filtered_df = filtered_df[
        filtered_df['技能标签'].apply(lambda x: all(skill in x for skill in selected_skills))
    ]

if selected_welfare:
    filtered_df = filtered_df[
        filtered_df['福利标签'].apply(lambda x: any(w in x for w in selected_welfare))
    ]

# ============================================================================
# 主界面
# ============================================================================
st.title("📊 大数据开发实习岗位分析平台")
st.markdown("### 🎯 帮助学生、求职者、高校就业指导中心、企业HR快速了解市场趋势")

if filtered_df.empty:
    st.warning("⚠️ 当前筛选条件下没有数据，请调整筛选条件")
    st.stop()

# KPI 指标卡
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📋 岗位总数", f"{len(filtered_df)}")
with col2:
    st.metric("💰 平均日薪", f"¥{filtered_df['平均薪资'].mean():.0f}")
with col3:
    st.metric("🏢 招聘企业", f"{filtered_df['公司名称'].nunique()}")
with col4:
    st.metric("🌆 覆盖城市", f"{filtered_df['城市'].nunique()}")
with col5:
    top_city = filtered_df['城市'].mode()[0] if len(filtered_df) > 0 else "无"
    st.metric("🔥 最热城市", top_city)

st.markdown("---")

# Tab 布局
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 综合概览", "💰 薪资分析", "🗺️ 地域分布", "💻 技能需求", "📋 岗位列表"
])

# Tab 1: 综合概览
with tab1:
    st.subheader("📈 综合数据概览")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🎓 学历要求分布")
        edu_counts = filtered_df['学历分类'].value_counts().reset_index()
        edu_counts.columns = ['学历', '数量']
        fig_edu = px.pie(edu_counts, values='数量', names='学历', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_edu.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_edu, use_container_width=True)
    
    with col_b:
        st.markdown("#### ⏰ 实习时长分布")
        duration_counts = filtered_df['实习时长分类'].value_counts().reset_index()
        duration_counts.columns = ['时长', '数量']
        fig_duration = px.bar(duration_counts, x='数量', y='时长', orientation='h',
                             color='数量', color_continuous_scale='Viridis', text='数量')
        fig_duration.update_layout(showlegend=False)
        st.plotly_chart(fig_duration, use_container_width=True)
    
    st.markdown("---")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("#### 🏢 发布岗位最多的公司 TOP10")
        company_counts = filtered_df['公司名称'].value_counts().head(10).reset_index()
        company_counts.columns = ['公司', '岗位数']
        fig_company = px.bar(company_counts, x='岗位数', y='公司', orientation='h',
                            color='岗位数', color_continuous_scale='Blues', text='岗位数')
        fig_company.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_company, use_container_width=True)
    
    with col_d:
        st.markdown("#### 🏭 行业分布 TOP10")
        industry_counts = filtered_df['所处行业'].value_counts().head(10).reset_index()
        industry_counts.columns = ['行业', '数量']
        fig_industry = px.bar(industry_counts, x='数量', y='行业', orientation='h',
                             color='数量', color_continuous_scale='Reds', text='数量')
        fig_industry.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_industry, use_container_width=True)

# Tab 2: 薪资分析
with tab2:
    st.subheader("💰 薪资深度分析")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### 📊 薪资分布直方图")
        fig_hist = px.histogram(filtered_df, x='平均薪资', nbins=40,
                               color_discrete_sequence=['#667eea'],
                               labels={'平均薪资': '日薪（元/天）', 'count': '岗位数量'})
        fig_hist.add_vline(x=filtered_df['平均薪资'].median(), line_dash="dash",
                          line_color="red", annotation_text=f"中位数: ¥{filtered_df['平均薪资'].median():.0f}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_s2:
        st.markdown("#### 📦 薪资箱线图")
        fig_box = px.box(filtered_df, y='平均薪资', points='all',
                        color_discrete_sequence=['#764ba2'],
                        labels={'平均薪资': '日薪（元/天）'})
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### 🎓 不同学历薪资对比")
    fig_edu_salary = px.box(filtered_df, x='学历分类', y='平均薪资', color='学历分类',
                            labels={'平均薪资': '日薪（元/天）', '学历分类': '学历要求'})
    st.plotly_chart(fig_edu_salary, use_container_width=True)
    
    st.markdown("#### 📋 薪资统计摘要")
    salary_stats = filtered_df['平均薪资'].describe().to_frame()
    salary_stats.columns = ['统计值']
    salary_stats.index = ['数量', '平均值', '标准差', '最小值', '25%分位', '中位数', '75%分位', '最大值']
    st.table(salary_stats)

# Tab 3: 地域分布
with tab3:
    st.subheader("🗺️ 地域分布分析")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 📍 城市岗位数量 TOP15")
        city_counts = filtered_df['城市'].value_counts().head(15).reset_index()
        city_counts.columns = ['城市', '岗位数']
        fig_city = px.bar(city_counts, x='城市', y='岗位数', color='岗位数',
                         color_continuous_scale='Teal', text='岗位数')
        st.plotly_chart(fig_city, use_container_width=True)
    
    with col_g2:
        st.markdown("#### 💰 城市平均薪资 TOP15")
        city_salary = filtered_df.groupby('城市')['平均薪资'].mean().sort_values(ascending=False).head(15).reset_index()
        city_salary.columns = ['城市', '平均薪资']
        fig_city_sal = px.bar(city_salary, x='城市', y='平均薪资', color='平均薪资',
                             color_continuous_scale='Oranges', text='平均薪资')
        fig_city_sal.update_traces(texttemplate='¥%{text:.0f}', textposition='outside')
        st.plotly_chart(fig_city_sal, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### 🌆 主要城市薪资分布对比")
    top_cities = filtered_df['城市'].value_counts().head(10).index
    df_top_cities = filtered_df[filtered_df['城市'].isin(top_cities)]
    fig_city_box = px.box(df_top_cities, x='城市', y='平均薪资', color='城市',
                          labels={'平均薪资': '日薪（元/天）'})
    st.plotly_chart(fig_city_box, use_container_width=True)

# Tab 4: 技能需求
with tab4:
    st.subheader("💻 技能需求分析")
    
    all_skills_list = []
    for skills in filtered_df['技能标签']:
        all_skills_list.extend(skills)
    
    if all_skills_list:
        skill_counter = Counter(all_skills_list)
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("#### 📈 技能需求排行榜 TOP20")
            skill_df = pd.DataFrame(skill_counter.most_common(20), columns=['技能', '需求次数'])
            fig_skill = px.bar(skill_df, x='需求次数', y='技能', orientation='h',
                              color='需求次数', color_continuous_scale='Viridis', text='需求次数')
            fig_skill.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_skill, use_container_width=True)
        
        with col_t2:
            st.markdown("#### ☁️ 技能词云")
            fig_wc, ax = plt.subplots(figsize=(10, 8))
            wordcloud = WordCloud(width=800, height=600, background_color='white',
                                 colormap='viridis', relative_scaling=0.5,
                                 min_font_size=12).generate_from_frequencies(skill_counter)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        
        st.markdown("---")
        
        st.markdown("#### 🔗 常见技能组合 TOP10")
        skill_combos = filtered_df['技能标签'].apply(
            lambda x: ', '.join(sorted(x)) if len(x) > 1 else None
        ).dropna()
        
        if not skill_combos.empty:
            combo_counts = skill_combos.value_counts().head(10).reset_index()
            combo_counts.columns = ['技能组合', '出现次数']
            st.table(combo_counts)
    else:
        st.info("当前筛选条件下未提取到技能标签")

# Tab 5: 岗位列表
with tab5:
    st.subheader("📋 岗位详情列表")
    st.info(f"📊 当前筛选条件下共有 **{len(filtered_df)}** 个岗位")
    
    display_df = filtered_df.copy()
    display_df['技能'] = display_df['技能标签'].apply(lambda x: ', '.join(x[:5]) if x else '未提取')
    display_df['福利'] = display_df['福利标签'].apply(lambda x: ', '.join(x[:3]) if x else '未提取')
    
    show_cols = ['职位标题', '公司名称', '城市', '平均薪资', '学历分类', 
                 '实习时长分类', '技能', '福利', '详情页url']
    
    # 使用 st.table 显示（更稳定）
    st.dataframe(display_df[show_cols].head(100), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### 📥 导出数据")
    
    csv = display_df[show_cols].to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下载筛选后的岗位数据（CSV）",
        data=csv,
        file_name=f"大数据开发岗位_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# 页脚
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 大数据开发实习岗位分析平台 | 数据来源：实习僧 | 更新时间：{}</p>
        <p>💡 适用于：学生求职、高校就业指导、企业HR招聘分析</p>
    </div>
""".format(datetime.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)
