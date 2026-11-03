
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정 및 데이터
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="⚖️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "wkddnwn2032/data/refs/heads/main/stroke.csv"
)

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("⚖️ 성능 높이기")
st.write(
    "뇌졸중을 겪은 사람은 전체에서 매우 적습니다. "
    "두 클래스의 가중치를 조정하기 전과 후를 비교해 "
    "모델의 예측 성능이 어떻게 달라지는지 살펴봅니다."
)

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()

required_columns = ["id", "stroke"] + FEATURES
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()


# --------------------------------------------------
# 2. 입력 속성 선택
# --------------------------------------------------

st.header("1. 입력 속성 선택")

selected_features = st.multiselect(
    "예측에 사용할 속성을 선택하세요.",
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
# 3. 데이터 정렬 및 분리
# --------------------------------------------------

st.header("2. 데이터 분리 및 전처리")

data = df.copy()

for col in selected_features + ["stroke", "id"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["stroke"])
data = data[data["stroke"].isin([0, 1])].copy()

data["_sort_id"] = data["id"]

data = data.sort_values(
    "_sort_id",
    kind="stable",
    na_position="last"
).reset_index(drop=True)

positions = np.arange(len(data))
test_mask = positions % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

st.write(
    f"전체 데이터 **{len(data):,}명** · "
    f"훈련 데이터 **{len(train_data):,}명** · "
    f"테스트 데이터 **{len(test_data):,}명**"
)

if len(test_data) != 1533:
    st.warning(
        f"테스트 데이터가 요청한 1,533명이 아니라 "
        f"현재 {len(test_data):,}명입니다. 데이터 행 수와 정렬을 확인하세요."
    )

if y_train.nunique() < 2:
    st.error("훈련 데이터에 뇌졸중 여부 두 종류가 모두 있어야 합니다.")
    st.stop()

# BMI를 선택했을 때만 훈련 데이터 중앙값으로 결측값 대체
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터의 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    train_missing_bmi = int(X_train["bmi"].isna().sum())
    test_missing_bmi = int(X_test["bmi"].isna().sum())

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"BMI 결측값은 훈련 데이터 중앙값 **{bmi_median:.2f}**로 채웠습니다. "
        f"(훈련용 {train_missing_bmi}명, 테스트용 {test_missing_bmi}명)"
    )

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 결측값이 있습니다. "
        "해당 데이터를 확인해 주세요."
    )
    st.stop()

# 훈련 데이터만으로 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success("훈련용과 테스트용 데이터 분리를 완료했습니다.")


# --------------------------------------------------
# 4. 가중치 조정 전·후 모델 학습
# --------------------------------------------------

st.header("3. 가중치 조정 전·후 모델 학습")

# 가중치 조정 전
logistic_before = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight=None
)
logistic_before.fit(X_train_scaled, y_train)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight=None
)
tree_before.fit(X_train_scaled, y_train)

# 가중치 조정 후
logistic_after = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)
logistic_after.fit(X_train_scaled, y_train)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)
tree_after.fit(X_train_scaled, y_train)

predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "before": logistic_before.predict(X_test_scaled),
        "after": logistic_after.predict(X_test_scaled),
        "before_model": logistic_before,
        "after_model": logistic_after
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "before": tree_before.predict(X_test_scaled),
        "after": tree_after.predict(X_test_scaled),
        "before_model": tree_before,
        "after_model": tree_after
    }
}

st.success(
    "두 모델을 가중치 조정 전(class_weight=None)과 "
    "조정 후(class_weight='balanced')로 각각 학습했습니다."
)

st.caption(
    "가중치 조정은 훈련 과정에서 드문 뇌졸중 클래스의 오차를 "
    "상대적으로 더 크게 반영하는 방법입니다. 테스트 데이터는 조정에 사용하지 않습니다."
)


# --------------------------------------------------
# 5. 혼동행렬 및 지표 계산
# --------------------------------------------------

def confusion_values(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def calculate_metrics(counts):
    tn, fp = counts["TN"], counts["FP"]
    fn, tp = counts["FN"], counts["TP"]

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else None

    actual_positive = tp + fn
    recall = tp / actual_positive if actual_positive else None

    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def percent(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


def confusion_dataframe(counts):
    tn, fp = counts["TN"], counts["FP"]
    fn, tp = counts["FN"], counts["TP"]

    return pd.DataFrame(
        [
            [
                f"TN {tn:,}명",
                f"FP {fp:,}명",
                f"{tn + fp:,}명"
            ],
            [
                f"FN {fn:,}명",
                f"TP {tp:,}명",
                f"{fn + tp:,}명"
            ],
            [
                f"{tn + fn:,}명",
                f"{fp + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        ],
        index=[
            "실제: 뇌졸중 아님 (0)",
            "실제: 뇌졸중 (1)",
            "열 합계"
        ],
        columns=[
            "예측: 뇌졸중 아님 (0)",
            "예측: 뇌졸중 (1)",
            "행 합계"
        ]
    )


def combined_comparison(before_counts, after_counts):
    before_metrics = calculate_metrics(before_counts)
    after_metrics = calculate_metrics(after_counts)

    rows = [
        ["혼동행렬 - 실제 아님 / 예측 아님",
         f"TN {before_counts['TN']:,}명",
         f"TN {after_counts['TN']:,}명"],

        ["혼동행렬 - 실제 아님 / 예측 뇌졸중",
         f"FP {before_counts['FP']:,}명",
         f"FP {after_counts['FP']:,}명"],

        ["혼동행렬 - 실제 뇌졸중 / 예측 아님",
         f"FN {before_counts['FN']:,}명",
         f"FN {after_counts['FN']:,}명"],

        ["혼동행렬 - 실제 뇌졸중 / 예측 뇌졸중",
         f"TP {before_counts['TP']:,}명",
         f"TP {after_counts['TP']:,}명"],

        ["정확도", percent(before_metrics["정확도"]),
         percent(after_metrics["정확도"])],

        ["재현율", percent(before_metrics["재현율"]),
         percent(after_metrics["재현율"])],

        ["정밀도", percent(before_metrics["정밀도"]),
         percent(after_metrics["정밀도"])],

        ["F1", percent(before_metrics["F1"]),
         percent(after_metrics["F1"])]
    ]

    return pd.DataFrame(
        rows,
        columns=["평가 항목", "가중치 조정 전", "가중치 조정 후"]
    )


# --------------------------------------------------
# 6. 두 모델의 전·후 비교표
# --------------------------------------------------

st.header("4. 가중치 조정 전·후 성능 비교")

st.write(
    f"테스트 데이터 **{len(y_test):,}명**, "
    f"실제 뇌졸중 환자 **{int((y_test == 1).sum()):,}명**을 기준으로 평가합니다."
)

for model_name, result in predictions.items():
    before_counts = confusion_values(y_test, result["before"])
    after_counts = confusion_values(y_test, result["after"])

    st.subheader(model_name)

    st.dataframe(
        combined_comparison(before_counts, after_counts),
        hide_index=True,
        use_container_width=True
    )

    before_found = before_counts["TP"]
    after_found = after_counts["TP"]
    before_missed = before_counts["FN"]
    after_missed = after_counts["FN"]
    before_false = before_counts["FP"]
    after_false = after_counts["FP"]

    st.markdown(
        f"**찾아낸 환자:** {before_found:,}명 → {after_found:,}명"
        f"　|　**놓친 환자:** {before_missed:,}명 → {after_missed:,}명"
        f"　|　**헛짚은 사람:** {before_false:,}명 → {after_false:,}명"
    )


# --------------------------------------------------
# 7. 산점도 축 선택
# --------------------------------------------------

st.divider()
st.header("5. 가중치 조정 전·후 0.5 경계 비교")

st.write(
    "테스트 데이터의 실제 뇌졸중 여부를 점 색으로 표시하고, "
    "로지스틱 회귀 모델의 예측 확률이 0.5가 되는 경계를 겹쳐 봅니다."
)

axis_features = selected_features

if len(axis_features) == 2:
    col_x, col_y = st.columns(2)

    with col_x:
        x_feature = st.selectbox(
            "가로축",
            axis_features,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    remaining_y = [f for f in axis_features if f != x_feature]

    with col_y:
        y_feature = st.selectbox(
            "세로축",
            remaining_y,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

else:
    st.caption("선택한 속성이 3개 이상이므로 3차원 산점도를 사용합니다.")

    default_axes = axis_features[:3]
    axis_cols = st.columns(3)

    with axis_cols[0]:
        x_feature = st.selectbox(
            "X축",
            axis_features,
            index=axis_features.index(default_axes[0]),
            format_func=lambda x: FEATURE_NAMES[x],
            key="axis_x_3d"
        )

    y_options = [f for f in axis_features if f != x_feature]

    with axis_cols[1]:
        y_default = default_axes[1] if default_axes[1] in y_options else y_options[0]

        y_feature = st.selectbox(
            "Y축",
            y_options,
            index=y_options.index(y_default),
            format_func=lambda x: FEATURE_NAMES[x],
            key="axis_y_3d"
        )

    z_options = [
        f for f in axis_features
        if f not in [x_feature, y_feature]
    ]

    with axis_cols[2]:
        z_default = default_axes[2] if default_axes[2] in z_options else z_options[0]

        z_feature = st.selectbox(
            "Z축",
            z_options,
            index=z_options.index(z_default),
            format_func=lambda x: FEATURE_NAMES[x],
            key="axis_z_3d"
        )


# --------------------------------------------------
# 8. 경계 계산용 함수
# --------------------------------------------------

def make_axis_grid(feature, values, count=100):
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))

    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0)
        low -= padding
        high += padding

    return np.linspace(low, high, count)


def probability_for_frame(model, frame):
    scaled = scaler.transform(frame[selected_features])
    return model.predict_proba(scaled)[:, 1]


def fixed_feature_values(axis_features_used):
    fixed = {}

    for feature in selected_features:
        if feature not in axis_features_used:
            fixed[feature] = float(X_test[feature].median())

    return fixed


# --------------------------------------------------
# 9. 2차원 산점도 및 경계선
# --------------------------------------------------

if len(selected_features) == 2:

    x_values = make_axis_grid(x_feature, X_test[x_feature])
    y_values = make_axis_grid(y_feature, X_test[y_feature])

    xx, yy = np.meshgrid(x_values, y_values)

    grid_frame = pd.DataFrame({
        feature: np.full(
            xx.size,
            float(X_test[feature].median())
        )
        for feature in selected_features
    })

    grid_frame[x_feature] = xx.ravel()
    grid_frame[y_feature] = yy.ravel()

    before_probability = probability_for_frame(
        logistic_before,
        grid_frame
    ).reshape(xx.shape)

    after_probability = probability_for_frame(
        logistic_after,
        grid_frame
    ).reshape(xx.shape)

    fig = go.Figure()

    # 실제 뇌졸중 여부
    for label, color in [
        (0, "#2878B5"),
        (1, "#D62728")
    ]:
        mask = y_test.to_numpy() == label

        fig.add_trace(
            go.Scatter(
                x=X_test.loc[mask, x_feature],
                y=X_test.loc[mask, y_feature],
                mode="markers",
                name=(
                    "실제 뇌졸중"
                    if label == 1
                    else "실제 뇌졸중 아님"
                ),
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.65
                ),
                hovertemplate=(
                    f"{FEATURE_NAMES[x_feature]}: %{{x}}<br>"
                    f"{FEATURE_NAMES[y_feature]}: %{{y}}<extra></extra>"
                )
            )
        )

    # 가중치 조정 전 0.5 경계선
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=before_probability,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none",
                showlabels=False
            ),
            line=dict(color="#F39C12", width=3),
            showscale=False,
            name="가중치 조정 전 0.5 경계",
            hoverinfo="skip"
        )
    )

    # 가중치 조정 후 0.5 경계선
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=after_probability,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none",
                showlabels=False
            ),
            line=dict(color="#8E44AD", width=3, dash="dash"),
            showscale=False,
            name="가중치 조정 후 0.5 경계",
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        title="실제 결과와 로지스틱 회귀의 0.5 경계선",
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        legend_title="범례",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    # 경계선이 표시 범위 안에 있는지 확인
    before_cross = (
        float(np.nanmin(before_probability)) <= 0.5
        <= float(np.nanmax(before_probability))
    )

    after_cross = (
        float(np.nanmin(after_probability)) <= 0.5
        <= float(np.nanmax(after_probability))
    )

    if not before_cross:
        st.info("가중치 조정 전 0.5 경계선은 그림 범위 밖에 있습니다.")

    if not after_cross:
        st.info("가중치 조정 후 0.5 경계선은 그림 범위 밖에 있습니다.")

    fixed = fixed_feature_values([x_feature, y_feature])

    if fixed:
        fixed_text = ", ".join(
            f"{FEATURE_NAMES[k]}={v:.2f}"
            for k, v in fixed.items()
        )

        st.caption(
            "두 축 이외의 속성은 테스트 데이터 중앙값에 고정했습니다: "
            + fixed_text
        )


# --------------------------------------------------
# 10. 3차원 산점도 및 경계 평면
# --------------------------------------------------

else:

    x_values = make_axis_grid(
        x_feature,
        X_test[x_feature],
        35
    )

    y_values = make_axis_grid(
        y_feature,
        X_test[y_feature],
        35
    )

    xx, yy = np.meshgrid(x_values, y_values)

    fig = go.Figure()

    # 실제 결과 산점도
    for label, color in [
        (0, "#2878B5"),
        (1, "#D62728")
    ]:
        mask = y_test.to_numpy() == label

        fig.add_trace(
            go.Scatter3d(
                x=X_test.loc[mask, x_feature],
                y=X_test.loc[mask, y_feature],
                z=X_test.loc[mask, z_feature],
                mode="markers",
                name=(
                    "실제 뇌졸중"
                    if label == 1
                    else "실제 뇌졸중 아님"
                ),
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.65
                )
            )
        )

    fixed = fixed_feature_values([
        x_feature,
        y_feature,
        z_feature
    ])

    # 로지스틱 회귀의 0.5 경계 평면 계산
    plane_results = []

    for model, label, color in [
        (
            logistic_before,
            "가중치 조정 전 0.5 평면",
            "#F39C12"
        ),
        (
            logistic_after,
            "가중치 조정 후 0.5 평면",
            "#8E44AD"
        )
    ]:

        coef = model.coef_[0]
        intercept = float(model.intercept_[0])

        coef_map = {
            feature: coef[i] / scaler.scale_[i]
            for i, feature in enumerate(selected_features)
        }

        adjusted_intercept = intercept - sum(
            coef[i] * scaler.mean_[i] / scaler.scale_[i]
            for i in range(len(selected_features))
        )

        for feature, value in fixed.items():
            adjusted_intercept += coef_map[feature] * value

        ax = coef_map[x_feature]
        ay = coef_map[y_feature]
        az = coef_map[z_feature]

        if np.isclose(az, 0.0):
            plane_results.append((label, None, color))
            continue

        # ax*x + ay*y + az*z + intercept = 0
        zz = -(
            ax * xx + ay * yy + adjusted_intercept
        ) / az

        z_min = float(X_test[z_feature].min())
        z_max = float(X_test[z_feature].max())

        # 표시 범위 안의 평면만 표시
        zz_plot = np.where(
            (zz >= z_min) & (zz <= z_max),
            zz,
            np.nan
        )

        visible = np.isfinite(zz_plot).any()

        if visible:
            fig.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=zz_plot,
                    name=label,
                    showscale=False,
                    opacity=0.35,
                    colorscale=[
                        [0, color],
                        [1, color]
                    ],
                    hoverinfo="skip"
                )
            )

        plane_results.append((label, visible, color))

    fig.update_layout(
        title="3차원 산점도와 로지스틱 회귀의 0.5 경계 평면",
        scene=dict(
            xaxis_title=FEATURE_NAMES[x_feature],
            yaxis_title=FEATURE_NAMES[y_feature],
            zaxis_title=FEATURE_NAMES[z_feature]
        ),
        legend_title="범례",
        height=750
    )

    st.plotly_chart(fig, use_container_width=True)

    for label, visible, color in plane_results:

        if visible is None:
            st.info(
                f"{label}은 선택한 Z축 방향으로 평면을 나타낼 수 없습니다. "
                "해당 축의 계수가 0에 가깝습니다."
            )

        elif not visible:
            st.info(f"{label}은 그림 범위 밖에 있습니다.")

    if fixed:
        fixed_text = ", ".join(
            f"{FEATURE_NAMES[k]}={v:.2f}"
            for k, v in fixed.items()
        )

        st.caption(
            "세 축 이외의 속성은 테스트 데이터 중앙값에 고정했습니다: "
            + fixed_text
        )


# --------------------------------------------------
# 11. 마무리 안내
# --------------------------------------------------

st.divider()

st.caption(
    "가중치 조정은 뇌졸중 환자를 더 잘 찾아내도록 도울 수 있지만, "
    "헛짚는 사람도 늘어날 수 있습니다. 정확도 하나만 보지 말고 "
    "재현율·정밀도·F1과 놓친 환자 및 헛짚은 사람 수를 함께 비교하세요. "
    "이 실습 결과는 의료 진단이나 실제 환자 선별에 사용할 수 없습니다."
)
