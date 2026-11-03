
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/wkddnwn2032/data/refs/heads/main/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📋 채점표")
st.write(
    "테스트 데이터를 이용해 분류 모델의 예측 결과를 채점합니다. "
    "혼동행렬과 여러 평가지표를 비교해 봅니다."
)

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()

required_columns = ["id", "stroke"] + FEATURES
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "필요한 열이 없습니다: " + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
# --------------------------------------------------

st.header("1. 입력 속성 선택")

selected_features = st.multiselect(
    "예측에 사용할 속성",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("속성을 최소 두 개 이상 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 4. 데이터 정렬 및 분리
# --------------------------------------------------

st.header("2. 데이터 분리 및 전처리")

data = df.copy()

for col in selected_features + ["stroke", "id"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["stroke"])
data = data[data["stroke"].isin([0, 1])].copy()

# 번호 순으로 정렬
data["_sort_id"] = data["id"]

data = data.sort_values(
    "_sort_id",
    kind="stable",
    na_position="last"
).reset_index(drop=True)

# 10명씩 묶어 앞의 3명을 테스트용으로 지정
positions = np.arange(len(data))
test_mask = positions % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

st.write(
    f"전체 데이터: **{len(data):,}명**"
)

st.write(
    f"훈련 데이터: **{len(train_data):,}명** / "
    f"테스트 데이터: **{len(test_data):,}명**"
)

if len(test_data) != 1533:
    st.warning(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 {len(test_data):,}명입니다. "
        "데이터 개수와 정렬 상태를 확인하세요."
    )

if y_train.nunique() < 2:
    st.error("훈련 데이터에 뇌졸중 여부 두 종류가 모두 있어야 합니다.")
    st.stop()


# --------------------------------------------------
# 5. BMI 결측값 처리 및 표준화
# --------------------------------------------------

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터의 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    train_missing = X_train["bmi"].isna().sum()
    test_missing = X_test["bmi"].isna().sum()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"BMI 빈 값은 훈련 데이터 중앙값 "
        f"**{bmi_median:.2f}**로 채웠습니다. "
        f"(훈련용 {train_missing}명, 테스트용 {test_missing}명)"
    )

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 빈 값이 있습니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()

# 훈련 데이터만으로 표준화 기준 계산
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success("데이터 분리와 전처리를 완료했습니다.")


# --------------------------------------------------
# 6. 모델 학습
# --------------------------------------------------

st.header("3. 분류 모델 학습")

# 로지스틱 회귀
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train_scaled,
    y_train
)

# 테스트 데이터 예측
logistic_pred = logistic_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test_scaled)

# 기준 모델: 훈련 데이터에서 많은 쪽으로만 예측
majority_class = int(y_train.mode().iloc[0])

baseline_pred = np.full(
    len(y_test),
    majority_class,
    dtype=int
)

st.success("세 모델의 예측 준비가 완료되었습니다.")


# --------------------------------------------------
# 7. 혼동행렬 및 평가지표 계산 함수
# --------------------------------------------------

def confusion_values(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def calculate_metrics(counts):

    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    total = tn + fp + fn + tp

    # 정확도
    accuracy = (tp + tn) / total if total else None

    # 재현율 = TP / 실제 뇌졸중 인원
    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    # 정밀도 = TP / 뇌졸중이라고 예측한 인원
    predicted_positive = tp + fp

    precision = (
        tp / predicted_positive
        if predicted_positive > 0
        else None
    )

    # F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)
    if precision is None or recall is None:
        f1 = None

    elif precision + recall == 0:
        f1 = 0.0

    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def metric_text(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 8. 혼동행렬 표 만들기
# --------------------------------------------------

def confusion_table(counts):

    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    actual_negative = tn + fp
    actual_positive = fn + tp

    predicted_negative = tn + fn
    predicted_positive = fp + tp

    total = tn + fp + fn + tp

    table = pd.DataFrame(
        {
            "예측: 뇌졸중 아님 (0)": [
                f"TN: {tn:,}명",
                f"FN: {fn:,}명",
                f"{predicted_negative:,}명"
            ],

            "예측: 뇌졸중 (1)": [
                f"FP: {fp:,}명",
                f"TP: {tp:,}명",
                f"{predicted_positive:,}명"
            ],

            "행 합계": [
                f"{actual_negative:,}명",
                f"{actual_positive:,}명",
                f"{total:,}명"
            ]
        },

        index=[
            "실제: 뇌졸중 아님 (0)",
            "실제: 뇌졸중 (1)",
            "열 합계"
        ]
    )

    return table


model_results = {

    "로지스틱 회귀(확률로 답하는 모델)": {
        "prediction": logistic_pred,
        "counts": confusion_values(y_test, logistic_pred)
    },

    "의사결정트리(질문으로 답하는 모델)": {
        "prediction": tree_pred,
        "counts": confusion_values(y_test, tree_pred)
    },

    "훈련용에서 많은 쪽으로만 답하는 모델": {
        "prediction": baseline_pred,
        "counts": confusion_values(y_test, baseline_pred)
    }
}


# --------------------------------------------------
# 9. 혼동행렬 보여 주기
# --------------------------------------------------

st.header("4. 테스트 데이터 혼동행렬")

test_positive_count = int((y_test == 1).sum())

st.write(
    f"**테스트용 사람 수: {len(y_test):,}명** "
    f"| **실제 뇌졸중인 사람: {test_positive_count:,}명**"
)

st.caption(
    "행은 실제 결과, 열은 예측 결과입니다."
)

st.markdown(
    """
    - **TP**: 실제 뇌졸중이고 뇌졸중이라고 예측한 사람
    - **TN**: 실제 뇌졸중이 아니고 아니라고 예측한 사람
    - **FP**: 실제로는 아니지만 뇌졸중이라고 예측한 사람
    - **FN**: 실제 뇌졸중이지만 아니라고 예측한 사람
    """
)

for model_name, result in model_results.items():

    st.subheader(model_name)

    st.dataframe(
        confusion_table(result["counts"]),
        use_container_width=True
    )


# --------------------------------------------------
# 10. 평가지표 비교
# --------------------------------------------------

st.header("5. 모델별 평가지표 비교")

metric_rows = []

for model_name, result in model_results.items():

    metrics = calculate_metrics(result["counts"])

    metric_rows.append({
        "모델": model_name,
        "정확도": metric_text(metrics["정확도"]),
        "재현율": metric_text(metrics["재현율"]),
        "정밀도": metric_text(metrics["정밀도"]),
        "F1": metric_text(metrics["F1"])
    })

metrics_df = pd.DataFrame(metric_rows)

st.dataframe(
    metrics_df,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "정확도 = (TP+TN)/전체 인원"
)

st.caption(
    "재현율 = TP/(TP+FN), 정밀도 = TP/(TP+FP)"
)

st.caption(
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)

st.caption(
    "뇌졸중이라고 예측한 사람이 0명이면 "
    "정밀도와 F1은 '계산할 수 없음'으로 표시합니다."
)


# --------------------------------------------------
# 11. 놓친 환자와 헛짚은 사람 목록
# --------------------------------------------------

st.divider()

st.header("6. 놓친 환자와 헛짚은 사람 확인")

model_choice = st.selectbox(
    "목록을 확인할 모델 선택",
    options=list(model_results.keys())
)

chosen_prediction = model_results[model_choice]["prediction"]

chosen_counts = model_results[model_choice]["counts"]

# FN: 실제 뇌졸중(1)이지만 0이라고 예측
fn_mask = (
    (y_test.to_numpy() == 1)
    & (chosen_prediction == 0)
)

# FP: 실제 뇌졸중이 아니지만 1이라고 예측
fp_mask = (
    (y_test.to_numpy() == 0)
    & (chosen_prediction == 1)
)

missed_people = test_data.loc[fn_mask].copy()
false_alarm_people = test_data.loc[fp_mask].copy()


# 목록을 보기 좋은 형태로 변환
def make_person_list(person_data):

    result = pd.DataFrame()

    result["번호"] = person_data["id"]
    result["나이"] = person_data["age"]
    result["평균 혈당"] = person_data["avg_glucose_level"]

    result["고혈압 여부"] = person_data["hypertension"].map({
        0: "없음",
        1: "있음"
    })

    result["심장병 여부"] = person_data["heart_disease"].map({
        0: "없음",
        1: "있음"
    })

    result = result.sort_values(
        "나이",
        ascending=False,
        na_position="last"
    )

    return result


missed_list = make_person_list(missed_people)

false_alarm_list = make_person_list(false_alarm_people)


# --------------------------------------------------
# 12. 두 목록 나란히 표시
# --------------------------------------------------

left_col, right_col = st.columns(2)

with left_col:

    st.subheader(
        f"놓친 환자 (FN): {len(missed_list):,}명"
    )

    st.caption(
        "실제 뇌졸중 환자이지만 "
        "뇌졸중이 아니라고 예측한 사람"
    )

    if len(missed_list) > 0:

        st.dataframe(
            missed_list.head(20),
            hide_index=True,
            use_container_width=True
        )

        if len(missed_list) > 20:
            st.caption(
                f"전체 {len(missed_list):,}명 중 "
                "나이가 많은 순서로 앞의 20명만 표시합니다."
            )

    else:
        st.info("놓친 환자가 없습니다.")


with right_col:

    st.subheader(
        f"헛짚은 사람 (FP): {len(false_alarm_list):,}명"
    )

    st.caption(
        "실제 뇌졸중 환자가 아니지만 "
        "뇌졸중이라고 예측한 사람"
    )

    if len(false_alarm_list) > 0:

        st.dataframe(
            false_alarm_list.head(20),
            hide_index=True,
            use_container_width=True
        )

        if len(false_alarm_list) > 20:
            st.caption(
                f"전체 {len(false_alarm_list):,}명 중 "
                "나이가 많은 순서로 앞의 20명만 표시합니다."
            )

    else:
        st.info("헛짚은 사람이 없습니다.")


# --------------------------------------------------
# 13. 혼동행렬과 목록 인원 비교
# --------------------------------------------------

st.info(
    f"목록 인원 확인: "
    f"놓친 환자 {len(missed_list):,}명 = "
    f"혼동행렬 FN {chosen_counts['FN']:,}명, "
    f"헛짚은 사람 {len(false_alarm_list):,}명 = "
    f"혼동행렬 FP {chosen_counts['FP']:,}명입니다."
)


# --------------------------------------------------
# 14. 마무리
# --------------------------------------------------

st.divider()

st.caption(
    "이 페이지는 분류 모델 평가 방법을 학습하기 위한 실습입니다. "
    "실제 의료 진단이나 환자 선별에 사용할 수 없습니다."
)
