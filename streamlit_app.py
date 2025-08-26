# --- 라이브러리 임포트 ---
# 필요한 라이브러리들을 불러옵니다.
import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy import stats

import plotly.graph_objects as go
import plotly.express as px

# --- 0. 유틸리티 및 통계 계산 함수 ---

def format_p_value(p):
    """p-값을 논문 형식에 맞게 별표(*)로 변환하는 함수"""
    if p < 0.001:
        return f"{p:.3f} (***)"
    elif p < 0.01:
        return f"{p:.3f} (**)"
    elif p < 0.05:
        return f"{p:.3f} (*)"
    else:
        return f"{p:.3f}"

def calculate_paired_stats(pre_data, post_data):
    """대응표본 t-검정, 효과크기(Cohen's dz), 95% 신뢰구간을 계산하는 함수"""
    # 결측치가 있는 쌍을 제거 (Pairwise deletion)
    temp_df = pd.DataFrame({'pre': pre_data, 'post': post_data}).dropna()
    pre = temp_df['pre']
    post = temp_df['post']
    
    if len(pre) < 2:  # 데이터가 부족하면 계산 불가
        return {}

    # 통계량 계산
    t_stat, p_value = stats.ttest_rel(pre, post)
    diff = post - pre
    n = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    
    # Cohen's dz 계산
    cohen_dz = mean_diff / sd_diff if sd_diff != 0 else 0
    
    # 95% 신뢰구간 계산
    se_diff = sd_diff / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_diff - t_critical * se_diff
    ci_high = mean_diff + t_critical * se_diff
    
    return {
        '사전 평균': np.mean(pre),
        '사후 평균': np.mean(post),
        '사전 표준편차': np.std(pre, ddof=1),
        '사후 표준편차': np.std(post, ddof=1),
        '평균 차이': mean_diff,
        't-값': t_stat,
        'p-값': p_value,
        "Cohen's dz": cohen_dz,
        '95% CI 하한': ci_low,
        '95% CI 상한': ci_high,
    }

# --- 1. 앱 기본 설정 ---
st.set_page_config(layout="wide", page_title="대응표본 t-검정 분석기")
st.title("📄 교육연구대회용 사전-사후 데이터 분석")
st.subheader("대응표본 t-검정, 효과크기, 시각화 (문항/요인 단위 분석 지원)")
st.write("---")

# --- 2. 사이드바: 파일 업로드 및 분석 옵션 설정 ---
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요.", type=['csv'])
    
    # 파일이 업로드되면 분석 옵션 표시
    if uploaded_file:
        st.header("2. 분석 옵션 설정")
        
        # 분석 단위 선택
        analysis_unit = st.radio(
            "분석 단위 선택",
            ('문항 단위', '요인 단위'),
            help="'문항 단위'는 개별 변수를 그대로 사용합니다. '요인 단위'는 여러 문항을 합산/평균하여 새로운 요인 변수를 만들어 분석합니다."
        )

        # 요인 단위 분석 시 추가 옵션
        if analysis_unit == '요인 단위':
            agg_method = st.radio(
                "요인 점수 집계 방식",
                ('평균(mean)', '합산(sum)'),
                horizontal=True,
                help="요인을 구성하는 하위 문항들의 점수를 평균낼지 합산할지 선택합니다."
            )

            use_reverse = st.checkbox("역문항 처리 사용")
            if use_reverse:
                max_score = st.number_input("리커트 척도 최대 점수", min_value=1, value=5, step=1)
                reverse_items_str = st.text_area(
                    "역문항 목록 (쉼표로 구분)",
                    placeholder="예: 사전3, 사전5, 사후3, 사후5"
                )

            st.subheader("요인-문항 매핑 입력")
            factor_map_str = st.text_area(
                "분석할 요인 정보를 JSON 형식으로 입력하세요.",
                height=250,
                placeholder='''{
    "주도성": {
        "pre": ["사전1", "사전2", "사전3"],
        "post": ["사후1", "사후2", "사후3"]
    },
    "AI리터러시": {
        "pre": ["사전4", "사전5"],
        "post": ["사후4", "사후5"]
    }
}'''
            )

# --- 3. 메인 화면 ---
if uploaded_file is not None:
    # 데이터 로드
    try:
        df_original = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        df = df_original.copy() # 원본 데이터 보존
        st.header("📋 업로드된 데이터 미리보기")
        st.dataframe(df.head())
        options = df.columns.tolist()
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # --- 문항 단위 분석 UI ---
    if analysis_unit == '문항 단위':
        st.header("🔍 분석할 변수 선택 (문항 단위)")
        col1, col2 = st.columns(2)
        with col1:
            pre_vars = st.multiselect('사전 검사 변수 선택', options, key="pre_item")
        with col2:
            post_vars = st.multiselect('사후 검사 변수 선택', options, key="post_item")
        
        if pre_vars and post_vars and len(pre_vars) != len(post_vars):
            st.warning("⚠️ 사전-사후 검사 변수의 개수를 동일하게 맞춰주세요.")

    # --- '분석 실행' 버튼 ---
    st.write("---")
    if st.button('🚀 분석 실행', type="primary"):
        results = []
        df_for_plotting = pd.DataFrame() # 시각화용 데이터프레임
        
        # --- 요인 단위 분석 로직 ---
        if analysis_unit == '요인 단위':
            try:
                if not factor_map_str:
                    st.error("요인-문항 매핑 정보를 입력해주세요.")
                    st.stop()
                factor_map = json.loads(factor_map_str)

                # 1. (선택적) 역문항 처리
                if use_reverse and reverse_items_str:
                    reverse_items = [item.strip() for item in reverse_items_str.split(',')]
                    for item in reverse_items:
                        if item in df.columns:
                            df[item] = (max_score + 1) - df[item]
                        else:
                            st.warning(f"'{item}' 컬럼이 데이터에 없어 역코딩을 건너뜁니다.")
                
                # 2. 요인 점수 계산
                factor_scores = {}
                analysis_pairs = {}
                
                for factor, items in factor_map.items():
                    pre_cols, post_cols = items.get('pre', []), items.get('post', [])
                    
                    if not pre_cols or not post_cols:
                        st.error(f"'{factor}' 요인에 'pre' 또는 'post' 문항 목록이 없습니다.")
                        continue
                    
                    # 요인 점수 계산 (Pairwise deletion 적용)
                    factor_df_temp = df[pre_cols + post_cols].dropna()

                    if agg_method == '평균(mean)':
                        pre_score = factor_df_temp[pre_cols].mean(axis=1)
                        post_score = factor_df_temp[post_cols].mean(axis=1)
                    else: # '합산(sum)'
                        pre_score = factor_df_temp[pre_cols].sum(axis=1)
                        post_score = factor_df_temp[post_cols].sum(axis=1)

                    # 3. 통계 분석 수행
                    stats_result = calculate_paired_stats(pre_score, post_score)
                    if stats_result:
                        stats_result['요인'] = factor
                        results.append(stats_result)
                        
                        # 시각화용 데이터 저장
                        plot_temp_df = pd.DataFrame({
                            f"{factor}_사전": pre_score,
                            f"{factor}_사후": post_score
                        })
                        df_for_plotting = pd.concat([df_for_plotting, plot_temp_df], axis=1)

            except json.JSONDecodeError:
                st.error("JSON 형식이 올바르지 않습니다. 예시를 참고하여 수정해주세요.")
            except Exception as e:
                st.error(f"요인 분석 중 오류가 발생했습니다: {e}")

        # --- 문항 단위 분석 로직 ---
        else: # analysis_unit == '문항 단위'
            if not pre_vars or not post_vars or len(pre_vars) != len(post_vars):
                st.error("분석을 실행하려면 사전-사후 변수 쌍을 올바르게 선택해주세요.")
            else:
                for pre_var, post_var in zip(pre_vars, post_vars):
                    stats_result = calculate_paired_stats(df[pre_var], df[post_var])
                    if stats_result:
                        stats_result['사전 변수'] = pre_var
                        stats_result['사후 변수'] = post_var
                        results.append(stats_result)
                df_for_plotting = df
        
        # --- 4. 분석 결과 표시 ---
        if results:
            st.header("📊 분석 결과 요약")
            results_df = pd.DataFrame(results)

            # 결과 테이블 컬럼 순서 정리
            if analysis_unit == '요인 단위':
                cols_order = ['요인', '사전 평균', '사후 평균', '평균 차이', '사전 표준편차', '사후 표준편차', 
                              't-값', 'p-값', "Cohen's dz", '95% CI 하한', '95% CI 상한']
            else: # 문항 단위
                cols_order = ['사전 변수', '사후 변수', '사전 평균', '사후 평균', '평균 차이', '사전 표준편차', '사후 표준편차', 
                              't-값', 'p-값', "Cohen's dz", '95% CI 하한', '95% CI 상한']
            
            # p-값 포맷팅 및 유의성 컬럼 추가
            results_df['유의성'] = results_df['p-값'].apply(format_p_value)
            results_df = results_df[[col for col in cols_order if col in results_df.columns] + ['유의성']]
            
            # 소수점 포맷팅
            format_dict = {col: '{:.3f}' for col in results_df.columns if results_df[col].dtype == 'float64'}
            st.dataframe(results_df.style.format(format_dict))
            
            # CSV 다운로드 버튼
            csv_data = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 결과 테이블 CSV로 다운로드",
                data=csv_data,
                file_name="t-test_analysis_results.csv",
                mime="text/csv",
            )
            st.info("p-값 형식: p < .05 (*), p < .01 (**), p < .001 (***)")

            # --- 5. 데이터 시각화 ---
            st.header("📈 시각화 자료")
            
            # 분석 단위에 따라 반복할 대상 설정
            if analysis_unit == '요인 단위':
                plot_vars = [(f"{r['요인']}_사전", f"{r['요인']}_사후") for _, r in results_df.iterrows()]
            else: # 문항 단위
                plot_vars = zip(pre_vars, post_vars)
            
            for pre_var, post_var in plot_vars:
                st.subheader(f"'{pre_var.replace('_사전','').replace('_사후','').strip()}' 분석")
                
                # --- [수정됨] 시각화 레이아웃: 세로 배치 ---
                # st.columns를 제거하여 차트를 순서대로 세로로 배치합니다.
                
                # 1. Raincloud Plot
                plot_df_long = pd.DataFrame({
                    '점수': pd.concat([df_for_plotting[pre_var], df_for_plotting[post_var]], ignore_index=True),
                    '시점': [f'사전'] * len(df_for_plotting) + [f'사후'] * len(df_for_plotting)
                })
                fig_rain = px.violin(
                    plot_df_long, y='점수', x='시점', color='시점',
                    box=True, points='all', color_discrete_map={'사전': 'blue', '사후': 'orange'}
                )
                fig_rain.update_layout(
                    title_text='<b>사전-사후 분포 비교 (Raincloud Plot)</b>',
                    xaxis_title='검사 시점', yaxis_title='점수', showlegend=False
                )
                st.plotly_chart(fig_rain, use_container_width=True)

                # 2. Paired Line Plot (개별 변화 추이)
                fig_line = go.Figure()
                
                colors = px.colors.qualitative.Plotly
                
                for i in range(len(df_for_plotting)):
                    color = colors[i % len(colors)]
                    
                    fig_line.add_trace(go.Scatter(
                        x=['사전', '사후'],
                        y=[df_for_plotting.loc[i, pre_var], df_for_plotting.loc[i, post_var]],
                        mode='lines+markers',
                        line=dict(color=color, width=1.5),
                        marker=dict(color=color),
                        showlegend=False
                    ))
                fig_line.update_layout(
                    title_text='<b>개별 데이터 변화 추이</b>',
                    xaxis_title='검사 시점',
                    yaxis_title='점수'
                )
                st.plotly_chart(fig_line, use_container_width=True)
                
                st.write("---") # 각 요인별 분석 결과 구분을 위한 라인
        
        elif not results and st.session_state.get('button_clicked'):
             st.warning("분석을 완료했으나 유효한 결과가 없습니다. 데이터나 설정을 확인해주세요.")

else:
    st.info("👈 사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요.")

# --- 6. Footer ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: grey;">
        © 2025 이대형. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)