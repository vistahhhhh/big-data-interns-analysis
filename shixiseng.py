"""
实习僧大数据开发岗位数据分析平台
面向学生、求职者、高校就业指导中心、企业HR的交互式数据产品
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="实习僧大数据开发岗位分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 数据加载与清洗 ====================

@st.cache_data
def load_and_clean_data(file_path):
    """加载并清洗数据"""
    try:
        df = pd.read_csv(file_path)
        
        # 1. 薪资清洗
        def extract_avg_salary(salary_str):
            if pd.isna(salary_str):
                return np.nan
            numbers = re.findall(r'\d+', str(salary_str))
            if len(numbers) >= 2:
                return int((int(numbers[0]) + int(numbers[1])) / 2)
            elif len(numbers) == 1:
                return int(numbers[0])
            return np.nan
        
        df['avg_salary'] = df['薪资范围'].apply(extract_avg_salary)
        
        # 2. 每周天数清洗
        def clean_days_per_week(days_str):
            if pd.isna(days_str):
                return np.nan
            numbers = re.findall(r'\d+', str(days_str))
            if numbers:
                return f"{numbers[0]}天／周"
            return days_str
        
        df['每周天数'] = df['每周天数'].apply(clean_days_per_week)
        
        # 3. 实习时长清洗
        def clean_duration(duration_str):
            if pd.isna(duration_str):
                return np.nan
            numbers = re.findall(r'\d+', str(duration_str))
            if numbers:
                return f"{numbers[0]}个月"
            return duration_str
        
        df['实习时长'] = df['实习时长'].apply(clean_duration)
        
        def extract_duration_months(duration_str):
            if pd.isna(duration_str):
                return np.nan
            numbers = re.findall(r'\d+', str(duration_str))
            if numbers:
                return int(numbers[0])
            return np.nan
        
        df['duration_months'] = df['实习时长'].apply(extract_duration_months)
        
        # 4. 工作地点清洗
        def extract_city(location_str):
            if pd.isna(location_str):
                return "未知"
            
            location = str(location_str).strip()
            city_mapping = {
                '北京': '北京', '北京市': '北京', '上海': '上海', '上海市': '上海',
                '深圳': '深圳', '深圳市': '深圳', '广州': '广州', '广州市': '广州',
                '杭州': '杭州', '杭州市': '杭州', '成都': '成都', '成都市': '成都',
                '南京': '南京', '南京市': '南京', '武汉': '武汉', '武汉市': '武汉',
                '西安': '西安', '西安市': '西安', '苏州': '苏州', '苏州市': '苏州',
                '重庆': '重庆', '重庆市': '重庆', '天津': '天津', '天津市': '天津',
            }
            
            if location in city_mapping:
                return city_mapping[location]
            
            for city_key in city_mapping.keys():
                if city_key in location:
                    return city_mapping[city_key]
            
            return location
        
        df['城市'] = df['工作地点'].apply(extract_city)
        
        # 5. 技能标签化
        TECH_SKILLS = ['Java', 'Python', 'SQL', 'Hadoop', 'Spark', 'Flink', 
                       'Hive', 'Kafka', 'Scala', 'C++', 'Linux', 'MySQL', 
                       'Redis', 'HBase', 'Elasticsearch', 'Docker', 'Kubernetes']
        
        def extract_skills(description):
            if pd.isna(description):
                return []
            description_upper = str(description).upper()
            matched = []
            for skill in TECH_SKILLS:
                if skill.upper() in description_upper:
                    matched.append(skill)
            return matched
        
        df['matched_skills'] = df['职位描述'].apply(extract_skills)
        
        # 6. 福利标签化
        def extract_welfare_tags(welfare_str):
            if pd.isna(welfare_str):
                return []
            
            tags = re.split(r'[,，、；;\s]+', str(welfare_str))
            welfare_mapping = {
                '转正': '转正机会', '转正机会': '转正机会', '留用机会': '转正机会',
                '房补': '房补', '住房补贴': '房补', '餐补': '餐补', '饭补': '餐补',
                '下午茶': '下午茶', '零食': '下午茶', '周末双休': '周末双休',
                '双休': '周末双休', '五险一金': '五险一金', '五险': '五险一金',
                '交通补助': '交通补助', '交通补贴': '交通补助', '节日福利': '节日福利',
                '年终奖': '年终奖', '奖金': '年终奖', '弹性工作': '弹性工作',
                '团建': '团建活动', '带薪年假': '带薪年假', '定期体检': '定期体检',
            }
            
            standardized_tags = []
            for tag in tags:
                tag = tag.strip()
                if tag and len(tag) > 0:
                    mapped_tag = welfare_mapping.get(tag, tag)
                    if mapped_tag not in standardized_tags:
                        standardized_tags.append(mapped_tag)
            return standardized_tags
        
        df['welfare_tags'] = df['福利待遇'].apply(extract_welfare_tags)
        df['截止日期'] = pd.to_datetime(df['截止日期'], errors='coerce')
        
        return df
    
    except FileNotFoundError:
        st.error(f"❌ 文件未找到: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ 数据加载失败: {str(e)}")
        st.stop()


def filter_data(df, cities, education, duration, salary_range, required_skills, welfare_prefs):
    """根据用户选择的条件筛选数据"""
    filtered_df = df.copy()
    
    if cities and len(cities) > 0:
        filtered_df = filtered_df[filtered_df['城市'].isin(cities)]
    
    if education != "全部":
        education_hierarchy = {
            '不限': ['不限', '大专', '本科', '硕士', '博士'],
            '大专': ['大专', '本科', '硕士', '博士'],
            '本科': ['本科', '硕士', '博士'],
            '硕士': ['硕士', '博士'],
            '博士': ['博士']
        }
        if education in education_hierarchy:
            filtered_df = filtered_df[filtered_df['学历要求'].isin(education_hierarchy[education])]
    
    if duration != "全部":
        duration_num = int(re.findall(r'\d+', duration)[0])
        filtered_df = filtered_df[filtered_df['duration_months'] >= duration_num]
    
    filtered_df = filtered_df[
        (filtered_df['avg_salary'] >= salary_range[0]) & 
        (filtered_df['avg_salary'] <= salary_range[1])
    ]
    
    if required_skills and len(required_skills) > 0:
        def has_required_skills(skills_list):
            return all(skill in skills_list for skill in required_skills)
        filtered_df = filtered_df[filtered_df['matched_skills'].apply(has_required_skills)]
    
    if welfare_prefs and len(welfare_prefs) > 0:
        def has_welfare(welfare_list):
            return any(welfare in welfare_list for welfare in welfare_prefs)
        filtered_df = filtered_df[filtered_df['welfare_tags'].apply(has_welfare)]
    
    return filtered_df


def main():
    st.title("📊 实习僧大数据开发岗位分析平台")
    st.markdown("---")
    
    DATA_PATH = "Big_data_development_results.csv"
    df = load_and_clean_data(DATA_PATH)
    
    # 侧边栏筛选器
    st.sidebar.header("🔍 筛选条件")
    
    all_cities = sorted(df['城市'].unique().tolist())
    selected_cities = st.sidebar.multiselect("选择城市", options=all_cities, default=[])
    
    education_options = ['全部', '不限', '大专', '本科', '硕士', '博士']
    selected_education = st.sidebar.selectbox("学历要求", options=education_options, index=0)
    
    duration_options = ['全部'] + sorted(df['实习时长'].dropna().unique().tolist())
    selected_duration = st.sidebar.selectbox("实习时长", options=duration_options, index=0)
    
    min_salary = int(df['avg_salary'].min())
    max_salary = int(df['avg_salary'].max())
    salary_range = st.sidebar.slider("日薪范围（元/天）", min_value=min_salary, max_value=max_salary, 
                                     value=(min_salary, max_salary), step=10)
    
    all_skills = sorted(list(set([skill for skills in df['matched_skills'] for skill in skills])))
    selected_skills = st.sidebar.multiselect("必备技能", options=all_skills, default=[])
    
    all_welfare = sorted(list(set([tag for tags in df['welfare_tags'] for tag in tags])))
    selected_welfare = st.sidebar.multiselect("福利偏好", options=all_welfare, default=[])
    
    filtered_df = filter_data(df, selected_cities, selected_education, selected_duration, 
                             salary_range, selected_skills, selected_welfare)
    
    # 检查筛选后是否有数据
    if len(filtered_df) == 0:
        st.header("📈 核心指标")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("岗位总数", "0", delta="占比 0.0%")
        with col2:
            st.metric("平均日薪", "¥0", delta="暂无数据")
        with col3:
            st.metric("覆盖城市", "0", delta=f"总计 {df['城市'].nunique()} 个")
        with col4:
            st.metric("招聘企业", "0", delta=f"总计 {df['公司名称'].nunique()} 家")
        
        st.markdown("---")
        
        # 显示友好的提示信息
        st.warning("⚠️ 没有找到符合筛选条件的目标岗位")
        st.info("""
        **建议：**
        - 🔍 尝试放宽筛选条件（如扩大薪资范围、选择更多城市）
        - 📚 减少必备技能的数量要求
        - 🎓 调整学历或实习时长要求
        - 🎁 减少福利偏好的限制
        """)
        
        # 显示当前筛选条件
        st.subheader("当前筛选条件：")
        filter_info = []
        if selected_cities:
            filter_info.append(f"- **城市**: {', '.join(selected_cities)}")
        if selected_education != "全部":
            filter_info.append(f"- **学历**: {selected_education}")
        if selected_duration != "全部":
            filter_info.append(f"- **实习时长**: {selected_duration}")
        filter_info.append(f"- **薪资范围**: ¥{salary_range[0]} - ¥{salary_range[1]}/天")
        if selected_skills:
            filter_info.append(f"- **必备技能**: {', '.join(selected_skills)}")
        if selected_welfare:
            filter_info.append(f"- **福利偏好**: {', '.join(selected_welfare)}")
        
        st.markdown('\n'.join(filter_info))
        return
    
    # KPI 指标卡（有数据时）
    st.header("📈 核心指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("岗位总数", f"{len(filtered_df):,}", delta=f"占比 {len(filtered_df)/len(df)*100:.1f}%")
    with col2:
        avg_sal = filtered_df['avg_salary'].mean()
        st.metric("平均日薪", f"¥{avg_sal:.0f}", delta=f"中位数 ¥{filtered_df['avg_salary'].median():.0f}")
    with col3:
        st.metric("覆盖城市", f"{filtered_df['城市'].nunique()}", delta=f"总计 {df['城市'].nunique()} 个")
    with col4:
        st.metric("招聘企业", f"{filtered_df['公司名称'].nunique()}", delta=f"总计 {df['公司名称'].nunique()} 家")
    
    st.markdown("---")
    
    # ==================== 创建分页标签 ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 薪资与城市分析", 
        "🛠️ 技能与学历分析", 
        "🏢 企业与岗位推荐",
        "🏭 行业与趋势分析",
        "📋 数据详情表"
    ])
    
    # ==================== 第1页：薪资与城市分析 ====================
    with tab1:
        st.header("💰 薪资分布分析")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=filtered_df['avg_salary'], name='日薪分布', 
                                     marker_color='lightseagreen', boxmean='sd'))
            fig_box.update_layout(title="薪资箱线图", yaxis_title="日薪（元/天）", height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            fig_hist = px.histogram(filtered_df, x='avg_salary', nbins=30, title="薪资分布直方图",
                                   labels={'avg_salary': '日薪（元/天）'})
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        st.header("🌍 城市岗位热力分析")
        city_stats = filtered_df.groupby('城市').agg({'职位id': 'count', 'avg_salary': 'mean'}).reset_index()
        city_stats.columns = ['城市', '岗位数量', '平均薪资']
        city_stats = city_stats.sort_values('岗位数量', ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_city_count = px.bar(city_stats, x='城市', y='岗位数量', title="各城市岗位数量 TOP20",
                                    color='岗位数量', color_continuous_scale='Blues')
            fig_city_count.update_layout(height=400)
            st.plotly_chart(fig_city_count, use_container_width=True)
        
        with col2:
            fig_city_salary = px.bar(city_stats, x='城市', y='平均薪资', title="各城市平均薪资 TOP20",
                                     color='平均薪资', color_continuous_scale='Reds')
            fig_city_salary.update_layout(height=400)
            st.plotly_chart(fig_city_salary, use_container_width=True)
    
    # ==================== 第2页：技能与学历分析 ====================
    with tab2:
        st.header("🛠️ 技能需求分析")
        all_skills_list = [skill for skills in filtered_df['matched_skills'] for skill in skills]
        skill_counter = Counter(all_skills_list)
        skill_df = pd.DataFrame(skill_counter.most_common(20), columns=['技能', '出现次数'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_skills = px.bar(skill_df, x='出现次数', y='技能', orientation='h', title="高频技能 TOP20",
                               color='出现次数', color_continuous_scale='Viridis')
            fig_skills.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)
        
        with col2:
            if len(skill_df) > 0:
                fig_wordcloud = px.scatter(skill_df, x=np.random.randn(len(skill_df)), 
                                          y=np.random.randn(len(skill_df)), size='出现次数', text='技能',
                                          title="技能词云", color='出现次数', size_max=60)
                fig_wordcloud.update_traces(textposition='middle center')
                fig_wordcloud.update_layout(height=500, xaxis={'visible': False}, yaxis={'visible': False})
                st.plotly_chart(fig_wordcloud, use_container_width=True)
        
        st.markdown("---")
        
        st.header("🎓 学历要求分布")
        education_stats = filtered_df['学历要求'].value_counts().reset_index()
        education_stats.columns = ['学历', '数量']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_edu_pie = px.pie(education_stats, values='数量', names='学历', title="学历要求占比", hole=0.4)
            fig_edu_pie.update_layout(height=400)
            st.plotly_chart(fig_edu_pie, use_container_width=True)
        
        with col2:
            fig_edu_bar = px.bar(education_stats, x='学历', y='数量', title="学历要求数量分布", color='数量')
            fig_edu_bar.update_layout(height=400)
            st.plotly_chart(fig_edu_bar, use_container_width=True)
    
    # ==================== 第3页：企业与岗位推荐 ====================
    with tab3:
        st.header("🏢 热门招聘企业 TOP10")
        company_stats = filtered_df['公司名称'].value_counts().head(10).reset_index()
        company_stats.columns = ['公司', '岗位数量']
        
        fig_company = px.bar(company_stats, x='岗位数量', y='公司', orientation='h', 
                            title="发布岗位最多的公司 TOP10", color='岗位数量')
        fig_company.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_company, use_container_width=True)
        
        st.markdown("---")
        
        st.header("💼 推荐岗位 TOP10")
        st.markdown("**根据薪资、技能匹配度和福利综合推荐**")
        
        # 计算推荐分数：薪资权重50%，技能数量权重30%，福利数量权重20%
        recommend_df = filtered_df.copy()
        recommend_df['技能数量'] = recommend_df['matched_skills'].apply(len)
        recommend_df['福利数量'] = recommend_df['welfare_tags'].apply(len)
        
        # 归一化处理
        if recommend_df['avg_salary'].max() > 0:
            recommend_df['薪资得分'] = recommend_df['avg_salary'] / recommend_df['avg_salary'].max() * 50
        else:
            recommend_df['薪资得分'] = 0
            
        if recommend_df['技能数量'].max() > 0:
            recommend_df['技能得分'] = recommend_df['技能数量'] / recommend_df['技能数量'].max() * 30
        else:
            recommend_df['技能得分'] = 0
            
        if recommend_df['福利数量'].max() > 0:
            recommend_df['福利得分'] = recommend_df['福利数量'] / recommend_df['福利数量'].max() * 20
        else:
            recommend_df['福利得分'] = 0
        
        recommend_df['推荐分数'] = recommend_df['薪资得分'] + recommend_df['技能得分'] + recommend_df['福利得分']
        
        # 按推荐分数排序，取前10
        top_jobs = recommend_df.nlargest(10, '推荐分数')[[
            '职位标题', '公司名称', 'avg_salary', '城市', '学历要求', '实习时长',
            'matched_skills', 'welfare_tags', '推荐分数', '详情页url'
        ]].copy()
        
        # 显示推荐岗位卡片
        for idx, row in top_jobs.iterrows():
            with st.expander(f"⭐ {row['职位标题']} - {row['公司名称']} (推荐分数: {row['推荐分数']:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**💰 日薪**: ¥{row['avg_salary']:.0f}/天")
                    st.markdown(f"**📍 城市**: {row['城市']}")
                    st.markdown(f"**🎓 学历**: {row['学历要求']}")
                    st.markdown(f"**⏱️ 时长**: {row['实习时长']}")
                
                with col2:
                    skills_str = ', '.join(row['matched_skills']) if row['matched_skills'] else '未标注'
                    st.markdown(f"**🛠️ 技能要求**: {skills_str}")
                    
                    welfare_str = ', '.join(row['welfare_tags']) if row['welfare_tags'] else '未标注'
                    st.markdown(f"**🎁 福利待遇**: {welfare_str}")
                
                st.markdown(f"**🔗 [查看详情]({row['详情页url']})**")
    
    # ==================== 第4页：行业与趋势分析 ====================
    with tab4:
        st.header("📅 岗位发布时间趋势")
        filtered_df_with_date = filtered_df[filtered_df['截止日期'].notna()].copy()
        
        if len(filtered_df_with_date) > 0:
            filtered_df_with_date['月份'] = filtered_df_with_date['截止日期'].dt.to_period('M').astype(str)
            time_stats = filtered_df_with_date.groupby('月份').size().reset_index(name='岗位数量')
            time_stats = time_stats.sort_values('月份')
            
            fig_time = px.line(time_stats, x='月份', y='岗位数量', title="近一年岗位发布趋势", markers=True)
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("暂无有效的时间数据")
        
        st.markdown("---")
        
        st.header("🏭 行业分布分析")
        industry_stats = filtered_df['所处行业'].value_counts().head(15).reset_index()
        industry_stats.columns = ['行业', '数量']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sunburst = px.sunburst(industry_stats, path=['行业'], values='数量', 
                                       title="行业分布旭日图", color='数量')
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            fig_treemap = px.treemap(industry_stats, path=['行业'], values='数量', 
                                    title="行业分布树状图", color='数量')
            fig_treemap.update_layout(height=500)
            st.plotly_chart(fig_treemap, use_container_width=True)
    
    # ==================== 第5页：数据详情表 ====================
    with tab5:
        st.header("📋 岗位详情数据表")
        st.markdown(f"**共 {len(filtered_df)} 条岗位信息**")
        
        # 准备显示数据
        display_df = filtered_df[['职位标题', '公司名称', 'avg_salary', '城市', '学历要求', 
                                  '实习时长', 'matched_skills', 'welfare_tags', '详情页url']].copy()
        display_df.columns = ['职位标题', '公司', '日薪(元)', '城市', '学历', '实习时长', '技能要求', '福利', '详情页']
        
        # 处理列表类型的列
        display_df['技能要求'] = display_df['技能要求'].apply(lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else '未标注')
        display_df['福利'] = display_df['福利'].apply(lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else '未标注')
        
        # 重置索引
        display_df = display_df.reset_index(drop=True)
        
        # 添加分页功能
        st.markdown("---")
        
        # 每页显示的行数
        rows_per_page = st.selectbox("每页显示行数", [10, 25, 50, 100, 200], index=2)
        
        # 计算总页数
        total_pages = (len(display_df) - 1) // rows_per_page + 1
        
        # 页码选择
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page_number = st.number_input(
                f"页码 (共 {total_pages} 页)", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                step=1
            )
        
        # 计算当前页的数据范围
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(display_df))
        
        # 显示当前页的数据
        st.markdown(f"**显示第 {start_idx + 1} - {end_idx} 条，共 {len(display_df)} 条**")
        
        # 使用st.data_editor显示数据（可编辑表格，更稳定）
        st.data_editor(
            display_df.iloc[start_idx:end_idx],
            use_container_width=True,
            num_rows="fixed",
            disabled=True,
            hide_index=False
        )
        
        st.markdown("---")
        
        # 导出功能
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出全部数据为CSV",
                data=csv,
                file_name="filtered_jobs.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # 显示统计信息
            st.metric("数据总行数", len(display_df))


if __name__ == "__main__":
    main()
