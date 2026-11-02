import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/wkddnwn2032/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_실습실_기록.json"

COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]

# --------------------------------------------------
# 기록 파일 읽기 / 저장
# --------------------------------------------------
def load_notes():
    """저장된 우리말 뜻과 데이터 출처를 JSON 파일에서 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return {
            "meanings": saved.get("meanings", {}),
            "source": saved.get("source", ""),
        }
    except (OSError, json.JSONDecodeError):
        st.warning("기록 파일을 읽지 못했습니다. 새 기록으로 시작합니다.")
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    """내_기록 폴더에 UTF-8 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meanings": meanings, "source": source}
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


@st.cache_data
def load_data():
    """CSV를 UTF-8로 읽습니다. bmi의 빈 값은 그대로 유지합니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 각 열의 의미를 직접 정리해 보는 데이터 탐색 실습")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

missing_columns = [column for column in COLUMNS if column not in df.columns]
if missing_columns:
    st.error(f"CSV에서 필요한 열을 찾을 수 없습니다: {', '.join(missing_columns)}")
    st.stop()

# stroke가 숫자형 또는 문자열이어도 1을 올바르게 셉니다.
stroke_values = pd.to_numeric(df["stroke"], errors="coerce")
stroke_count = int((stroke_values == 1).sum())
stroke_ratio = stroke_count / len(df) * 100 if len(df) else 0

# --------------------------------------------------
# 첫 화면: 데이터 소개 및 요약 카드
# --------------------------------------------------
st.subheader("데이터 한눈에 보기")
st.write(
    "이 데이터는 여러 사람의 인구학적 정보와 건강 관련 항목을 담고 있습니다. "
    "`stroke` 열은 뇌졸중 발생 여부를 나타내는 값으로 사용됩니다. "
    "아래 표와 원본 데이터에서 열 이름, 값의 종류, 빈 값 등을 확인해 보세요."
)

metric_cols = st.columns(4)
metrics = [
    ("전체 사람 수", f"{len(df):,}명"),
    ("열 개수", f"{len(df.columns):,}개"),
    ("stroke가 1인 사람 수", f"{stroke_count:,}명"),
    ("stroke가 1인 비율", f"{stroke_ratio:.2f}%"),
]
for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.metric(label, value)

st.divider()

# --------------------------------------------------
# 열 정보 표: 우리말 뜻은 사용자가 직접 작성
# --------------------------------------------------
st.subheader("열 이름과 데이터 정보")
st.caption("‘우리말 뜻’ 칸을 교재를 참고해 채운 뒤, 표 아래의 저장 버튼을 눌러 주세요.")

saved_notes = load_notes()
saved_meanings = saved_notes["meanings"]

rows = []
for column in df.columns:
    rows.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": int(df[column].nunique(dropna=True)),
        "빈 값 개수": int(df[column].isna().sum()),
    })

info_df = pd.DataFrame(rows, columns=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"])
edited_info = st.data_editor(
    info_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
 disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 각 열 이름의 뜻을 직접 적어 보세요.",
            width="large",
        ),
        "값의 종류": st.column_config.NumberColumn(
            "값의 종류", help="빈 값을 제외한 서로 다른 값의 개수"
        ),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
    key="column_info_editor",
)

st.subheader("데이터 출처")
st.caption("교재에 제시된 데이터 출처를 아래에 직접 입력해 저장하세요.")
source_text = st.text_area(
    "데이터 출처",
    value=saved_notes["source"],
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
    height=100,
    key="source_text",
    label_visibility="collapsed",
)

if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    meanings_to_save = {}
    for _, row in edited_info.iterrows():
        meanings_to_save[str(row["열 이름"])] = str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""
    try:
        save_notes(meanings_to_save, source_text)
        st.success(f"저장했습니다: {RECORD_FILE}")
    except OSError as error:
        st.error("기록 파일을 저장하지 못했습니다. 실행 환경의 폴더 쓰기 권한을 확인해 주세요.")
        st.code(str(error))

st.divider()

# --------------------------------------------------
# 원본 데이터 첫 다섯 줄
# --------------------------------------------------
st.subheader("원본 데이터 미리 보기")
st.caption("CSV의 첫 다섯 행을 원본 열 이름과 값 그대로 표시합니다. 빈 값은 비어 있는 상태로 보입니다.")
st.dataframe(df.head(5), hide_index=True, use_container_width=True)

st.divider()
st.caption("실습 안내: 이 화면은 데이터 탐색용이며, 의료 진단이나 개인의 뇌졸중 위험 판정에 사용하기 위한 도구가 아닙니다.")
