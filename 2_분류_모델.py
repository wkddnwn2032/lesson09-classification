
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
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


st.title("🤖 분류 모델")
st.write(
    "로지스틱 회귀와 의사결정트리를 이용하여 "
    "뇌졸중 발생 여부를 예측하는 분류 모델을 만들어 봅니다."
)

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
# --------------------------------------------------

st.header("1. 입력 속성 선택")

st.write(
    "뇌졸중 여부를 예측할 때 사용할 속성을 선택하세요. "
    "처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다."
)

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
# 4. 데이터 정렬 및 학습 / 테스트 분리
# --------------------------------------------------

st.header("2. 학습 데이터와 테스트 데이터 분리")

data = df.copy()

for col in selected_features + ["stroke"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["stroke"])
data = data[data["stroke"].isin([0, 1])]

# ID 번호 순으로 정렬
data["_sort_id"] = pd.to_numeric(data["id"], errors="coerce")
data = data.sort_values(
    "_sort_id",
    kind="stable"
).reset_index(drop=True)

# 10명씩 묶었을 때 앞의 3명을 테스트용으로 지정
positions = np.arange(len(data))
test_mask = positions % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

# 테스트 데이터가 학습에 들어가지 않았는지 확인
st.write(
    f"전체 데이터: **{len(data):,}명**"
)
st.write(
    f"학습 데이터: **{len(train_data):,}명** / "
    f"테스트 데이터: **{len(test_data):,}명**"
)

if len(test_data) != 1533:
    st.warning(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 테스트 데이터는 {len(test_data):,}명입니다. "
        "데이터 개수와 정렬 상태를 확인하세요."
    )

if y_train.nunique() < 2:
    st.error("학습 데이터에 뇌졸중 여부 두 종류가 모두 있어야 합니다.")
    st.stop()


# --------------------------------------------------
# 5. BMI 결측값 처리
# --------------------------------------------------

st.header("3. 결측값 처리 및 표준화")

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습 데이터의 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    missing_train = X_train["bmi"].isna().sum()
    missing_test = X_test["bmi"].isna().sum()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"BMI 빈 값은 학습 데이터의 중앙값인 "
        f"**{bmi_median:.2f}**로 채웠습니다. "
        f"(학습 데이터 {missing_train}명, 테스트 데이터 {missing_test}명)"
    )

# BMI 이외에 결측값이 있는지 확인
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 빈 값이 있습니다. "
        "해당 데이터의 결측값을 확인해 주세요."
    )
    st.stop()

# 표준화는 학습 데이터로만 계산
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success(
    "학습 데이터로 표준화 기준을 계산하고 "
    "테스트 데이터에는 같은 기준을 적용했습니다."
)


# --------------------------------------------------
# 6. 모델 학습
# --------------------------------------------------

st.header("4. 분류 모델 학습")

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

st.success("두 가지 분류 모델의 학습이 완료되었습니다.")


# --------------------------------------------------
# 7. 정확도 계산
# --------------------------------------------------

st.header("5. 모델 정확도 비교")

# 로지스틱 회귀
logistic_train_acc = logistic_model.score(
    X_train_scaled, y_train
)

logistic_test_acc = logistic_model.score(
    X_test_scaled, y_test
)

# 의사결정트리
tree_train_acc = tree_model.score(
    X_train_scaled, y_train
)

tree_test_acc = tree_model.score(
    X_test_scaled, y_test
)

# 가장 많은 클래스로만 예측하는 기준 모델
majority_class = y_train.mode()[0]

baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()


def accuracy_card(title, train_acc, test_acc):

    st.subheader(title)

    st.metric(
        label="테스트 데이터 정확도",
        value=f"{test_acc * 100:.2f}%"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.caption(
            f"훈련 데이터 정확도: {train_acc * 100:.2f}%"
        )

    with col2:
        st.caption(
            f"테스트 데이터 정확도: {test_acc * 100:.2f}%"
        )


card1, card2, card3 = st.columns(3)

with card1:
    accuracy_card(
        "로지스틱 회귀\n(확률로 답하는 모델)",
        logistic_train_acc,
        logistic_test_acc
    )

with card2:
    accuracy_card(
        "의사결정트리\n(질문으로 답하는 모델)",
        tree_train_acc,
        tree_test_acc
    )

with card3:
    accuracy_card(
        "기준 모델\n(많은 쪽으로만 답하는 모델)",
        baseline_train_acc,
        baseline_test_acc
    )

st.caption(
    "정확도는 전체 예측 중 실제 정답과 일치한 비율입니다. "
    "기준 모델은 입력 속성을 사용하지 않고 학습 데이터에서 "
    "더 많은 쪽의 결과만 예측합니다."
)


# --------------------------------------------------
# 8. 두 속성을 선택하여 결정 경계 시각화
# --------------------------------------------------

st.divider()
st.header("6. 모델의 예측 영역과 실제 데이터")

st.write(
    "선택한 속성 중 두 개를 골라 가로축과 세로축으로 설정합니다. "
    "점의 색은 테스트 데이터의 실제 뇌졸중 여부를 나타냅니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        selected_features,
        format_func=lambda x: FEATURE_NAMES[x],
        index=0
    )

with axis_col2:
    y_options = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        y_options,
        format_func=lambda x: FEATURE_NAMES[x],
        index=0
    )

x_index = selected_features.index(x_feature)
y_index = selected_features.index(y_feature)

# 테스트 데이터의 실제 값
plot_data = X_test.copy()

plot_data["실제 결과"] = y_test.map({
    0: "뇌졸중 아님",
    1: "뇌졸중"
}).values

# 두 축이 아닌 나머지 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

st.write("**그림을 그릴 때 고정한 나머지 속성의 값**")

if fixed_values:
    for feature, value in fixed_values.items():
        st.write(f"- {FEATURE_NAMES[feature]}: {value:.3f}")
else:
    st.write("고정할 나머지 속성이 없습니다.")

# 테스트 데이터 범위를 기준으로 격자 생성
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_margin = (x_max - x_min) * 0.05
y_margin = (y_max - y_min) * 0.05

if x_margin == 0:
    x_margin = 0.5

if y_margin == 0:
    y_margin = 0.5

x_range = np.linspace(
    x_min - x_margin,
    x_max + x_margin,
    180
)

y_range = np.linspace(
    y_min - y_margin,
    y_max + y_margin,
    180
)

xx, yy = np.meshgrid(x_range, y_range)

grid_data = pd.DataFrame(
    np.tile(
        X_test.median().values,
        (xx.size, 1)
    ),
    columns=selected_features
)

grid_data[x_feature] = xx.ravel()
grid_data[y_feature] = yy.ravel()

# 테스트 중앙값을 기준으로 고정한 값 기록
for feature, value in fixed_values.items():
    grid_data[feature] = value

# 학습 때 사용한 표준화 기준 적용
grid_scaled = scaler.transform(grid_data)

# 로지스틱 회귀의 뇌졸중 확률
probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1].reshape(xx.shape)

# 의사결정트리의 예측 결과
tree_regions = tree_model.predict(
    grid_scaled
).reshape(xx.shape)

# --------------------------------------------------
# 9. Plotly 산점도
# --------------------------------------------------

fig = go.Figure()

# 의사결정트리 예측 영역을 옅은 색으로 표시
fig.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=tree_regions,
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill"
        ),
        colorscale=[
            [0, "rgba(70,130,180,0.12)"],
            [1, "rgba(255,100,100,0.12)"]
        ],
        showscale=False,
        line=dict(width=0),
        name="의사결정트리 예측 영역",
        hoverinfo="skip"
    )
)

# 로지스틱 회귀의 0.5 확률 경계선이 존재하는지 확인
has_boundary = (
    probability.min() <= 0.5
    and probability.max() >= 0.5
)

if has_boundary:

    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=probability,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none",
                showlabels=True
            ),
            line=dict(
                color="black",
                width=3
            ),
            showscale=False,
            name="로지스틱 회귀 0.5 경계선",
            hoverinfo="skip"
        )
    )

# 실제 테스트 데이터 표시
for label, color in [
    ("뇌졸중 아님", "royalblue"),
    ("뇌졸중", "crimson")
]:

    subset = plot_data[
        plot_data["실제 결과"] == label
    ]

    fig.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=f"실제: {label}",
            marker=dict(
                color=color,
                size=7,
                opacity=0.7
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y}}<br>"
                f"실제 결과: {label}<extra></extra>"
            )
        )
    )

fig.update_layout(
    title="테스트 데이터와 두 모델의 예측 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=650,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

if not has_boundary:
    st.warning(
        "로지스틱 회귀의 뇌졸중 확률 0.5 경계선이 "
        "현재 그림의 범위 안에 나타나지 않습니다."
    )
else:
    st.info(
        "검은 선은 로지스틱 회귀 모델이 뇌졸중 확률 0.5를 기준으로 "
        "예측을 나누는 경계선입니다."
    )

st.caption(
    "배경의 옅은 파란색과 붉은색 영역은 의사결정트리의 예측 영역입니다. "
    "점의 색은 모델 예측이 아닌 실제 테스트 데이터의 결과입니다."
)


# --------------------------------------------------
# 10. 의사결정트리의 질문 구조
# --------------------------------------------------

st.divider()
st.header("7. 의사결정트리가 던지는 질문")

st.write(
    "의사결정트리는 질문을 최대 세 번까지 던집니다. "
    "각 마디에는 해당 마디에 도달한 훈련 데이터의 사람 수와 "
    "실제 뇌졸중 인원 및 비율을 표시합니다."
)

tree_structure = tree_model.tree_

# 훈련 데이터가 각 노드에 도달한 횟수 계산
decision_path = tree_model.decision_path(X_train_scaled)

node_people = np.asarray(
    decision_path.sum(axis=0)
).ravel().astype(int)

node_positive = np.zeros(
    tree_structure.node_count,
    dtype=int
)

for node in range(tree_structure.node_count):

    node_mask = decision_path[:, node].toarray().ravel().astype(bool)

    node_positive[node] = int(
        y_train.iloc[np.where(node_mask)[0]].sum()
    )


# DOT 문자열 생성
dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, margin=0.15];',
    'edge [fontname="sans-serif", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
asked_features = set()


def build_tree_dot(node):

    global leaf_count, negative_leaf_count

    people = int(node_people[node])
    positive = int(node_positive[node])

    rate = positive / people * 100 if people > 0 else 0

    is_leaf = (
        tree_structure.children_left[node]
        == tree_structure.children_right[node]
    )

    if is_leaf:

        leaf_count += 1

        # 해당 마디에서 모델이 선택한 답
        class_index = np.argmax(
            tree_structure.value[node][0]
        )

        prediction = int(tree_model.classes_[class_index])

        if prediction == 1:
            answer = "뇌졸중"
            fill = "#F7B7B7"
            font = "#7A1010"
        else:
            answer = "뇌졸중 아님"
            fill = "#B9D8F5"
            font = "#103B70"
            negative_leaf_count += 1

        label = (
            f"답: {answer}\\n"
            f"훈련용: {people}명\\n"
            f"실제 뇌졸중: {positive}명\\n"
            f"실제 비율: {rate:.1f}%"
        )

        dot_lines.append(
            f'{node} [label="{label}", '
            f'fillcolor="{fill}", fontcolor="{font}"];'
        )

        return

    # 질문이 있는 마디
    feature_index = tree_structure.feature[node]
    feature = selected_features[feature_index]

    asked_features.add(feature)

    # 표준화된 임계값을 원래 단위로 변환
    scaled_threshold = tree_structure.threshold[node]

    original_threshold = (
        scaled_threshold * scaler.scale_[feature_index]
        + scaler.mean_[feature_index]
    )

    # 질문 내용
    question = (
        f"{FEATURE_NAMES[feature]} "
        f"≤ {original_threshold:.2f} ?"
    )

    label = (
        f"{question}\\n"
        f"훈련용: {people}명\\n"
        f"실제 뇌졸중: {positive}명\\n"
        f"실제 비율: {rate:.1f}%"
    )

    dot_lines.append(
        f'{node} [label="{label}", fillcolor="#FFF0B3"];'
    )

    left = tree_structure.children_left[node]
    right = tree_structure.children_right[node]

    dot_lines.append(
        f'{node} -> {left} [label="예"];'
    )

    dot_lines.append(
        f'{node} -> {right} [label="아니요"];'
    )

    build_tree_dot(left)
    build_tree_dot(right)


build_tree_dot(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 11. 의사결정트리 분석 결과
# --------------------------------------------------

st.subheader("의사결정트리 요약")

st.write(
    f"- **답을 내리는 마디의 개수:** {leaf_count}칸"
)

st.write(
    f"- **뇌졸중 아님이라고 답하는 마디:** "
    f"{negative_leaf_count}칸"
)

if asked_features:

    used_names = [
        FEATURE_NAMES[f]
        for f in selected_features
        if f in asked_features
    ]

    st.write(
        "- **이 나무가 실제로 질문한 속성:** "
        + ", ".join(used_names)
    )

else:
    st.write(
        "- **이 나무가 실제로 질문한 속성:** 없음"
    )

st.divider()

st.caption(
    "주의: 이 모델은 학습용 데이터 분석 실습을 위한 것입니다. "
    "실제 의료 진단이나 개인의 뇌졸중 발생 가능성을 판단하는 데 "
    "사용할 수 없습니다."
)
