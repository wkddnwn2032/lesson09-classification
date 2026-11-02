import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/wkddnwn2032/data/refs/heads/main/stroke.csv"


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.caption("뇌졸중 데이터의 분포와 변수 사이의 관계를 탐색합니다.")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# 숫자형 데이터 변환
data = df.copy()

numeric_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "stroke"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 뇌졸중 여부 열을 먼저 생성
data["뇌졸중 여부"] = data["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

# 그다음 분석할 데이터 추출
valid_stroke = data[data["stroke"].isin([0, 1])].copy()


# --------------------------------------------------
# 1. 나이와 평균 혈당 분포
# --------------------------------------------------

st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        data.dropna(subset=["age"]),
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"}
    )
    fig.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        data.dropna(subset=["avg_glucose_level"]),
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )
    fig.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )
    st.plotly_chart(fig, use_container_width=True)


st.divider()


# --------------------------------------------------
# 2. 뇌졸중 여부에 따른 상자그림
# --------------------------------------------------

st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        valid_stroke.dropna(subset=["age"]),
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        points="outliers",
        title="뇌졸중 여부에 따른 나이"
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="나이"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        valid_stroke.dropna(subset=["avg_glucose_level"]),
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        points="outliers",
        title="뇌졸중 여부에 따른 평균 혈당"
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="평균 혈당"
    )
    st.plotly_chart(fig, use_container_width=True)


# 두 그룹 평균값
st.subheader("두 그룹의 평균값")

mean_rows = []

for value, name in [
    (0, "뇌졸중 없음"),
    (1, "뇌졸중 있음")
]:
    group = valid_stroke[valid_stroke["stroke"] == value]

    mean_rows.append({
        "그룹": name,
        "사람 수": len(group),
        "평균 나이": group["age"].mean(),
        "평균 혈당": group["avg_glucose_level"].mean()
    })

mean_df = pd.DataFrame(mean_rows)

mean_df["평균 나이"] = mean_df["평균 나이"].round(2)
mean_df["평균 혈당"] = mean_df["평균 혈당"].round(2)

st.dataframe(
    mean_df,
    hide_index=True,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 3. 고혈압과 심장병에 따른 뇌졸중 비율
# --------------------------------------------------

st.header("3. 건강 상태에 따른 뇌졸중 비율")


def make_rate_table(column, labels):

    subset = valid_stroke.dropna(subset=[column]).copy()

    subset["상태"] = subset[column].map(labels)

    result = (
        subset.groupby("상태")
        .agg(
            전체_인원=("stroke", "size"),
            뇌졸중_인원=("stroke", "sum")
        )
        .reset_index()
    )

    result = result.rename(columns={
        "전체_인원": "전체 인원",
        "뇌졸중_인원": "뇌졸중 인원"
    })

    result["뇌졸중 비율(%)"] = (
        result["뇌졸중 인원"]
        / result["전체 인원"] * 100
    ).round(2)

    return result


col1, col2 = st.columns(2)

with col1:

    hypertension_df = make_rate_table(
        "hypertension",
        {
            0: "고혈압 없음",
            1: "고혈압 있음"
        }
    )

    fig = px.bar(
        hypertension_df,
        x="상태",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 여부에 따른 뇌졸중 비율"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        hypertension_df,
        hide_index=True,
        use_container_width=True
    )


with col2:

    heart_df = make_rate_table(
        "heart_disease",
        {
            0: "심장병 없음",
            1: "심장병 있음"
        }
    )

    fig = px.bar(
        heart_df,
        x="상태",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 여부에 따른 뇌졸중 비율"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        heart_df,
        hide_index=True,
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# 4. BMI 빈 값과 전체 뇌졸중 비율 비교
# --------------------------------------------------

st.header("4. BMI 빈 값과 뇌졸중 비율")

missing_bmi = data[data["bmi"].isna()]

missing_valid = missing_bmi[
    missing_bmi["stroke"].isin([0, 1])
]


def calculate_stroke_rate(series):

    series = series[series.isin([0, 1])]

    if len(series) == 0:
        return 0

    return (series == 1).mean() * 100


bmi_df = pd.DataFrame([
    {
        "구분": "BMI 빈 값인 사람",
        "사람 수": len(missing_valid),
        "뇌졸중 인원": int(
            (missing_valid["stroke"] == 1).sum()
        ),
        "뇌졸중 비율(%)": round(
            calculate_stroke_rate(missing_valid["stroke"]), 2
        )
    },
    {
        "구분": "전체 사람",
        "사람 수": len(valid_stroke),
        "뇌졸중 인원": int(
            (valid_stroke["stroke"] == 1).sum()
        ),
        "뇌졸중 비율(%)": round(
            calculate_stroke_rate(valid_stroke["stroke"]), 2
        )
    }
])

st.dataframe(
    bmi_df,
    hide_index=True,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 5. 흡연 상태별 사람 수
# --------------------------------------------------

st.header("5. 흡연 상태별 사람 수")

smoking_df = (
    data["smoking_status"]
    .fillna("빈 값")
    .astype(str)
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_df,
    hide_index=True,
    use_container_width=True
)


st.divider()

st.caption(
    "이 페이지는 데이터 탐색을 위한 학습용 분석입니다. "
    "변수 사이의 차이만으로 뇌졸중의 원인이나 개인의 위험도를 "
    "단정할 수 없습니다."
)
