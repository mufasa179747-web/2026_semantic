import streamlit as st
from transformers import pipeline

# 設定網頁標題
st.set_page_config(page_title="中文產品評論情緒分析器", page_icon="📊")
st.title("📊 中文產品評論情緒分析器")
st.write("輸入產品評論，系統會自動評估 1 到 5 等級的情緒星等！")

# 載入情緒分析模型 (使用快取避免每次重新載入)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="LiYuan/amazon-review-sentiment-analysis")

sentiment_analyzer = load_model()

def analyze_review_stars(text):
    # 執行情緒預測
    result = sentiment_analyzer(text)[0]
    label = result['label']
    try:
        stars = int(label.split()[0])
    except (IndexError, ValueError):
        stars = 3
    return stars, result['score']

# 建立使用者輸入介面
user_input = st.text_area("請在下方輸入產品評論文字：", placeholder="例如：這款產品真的太棒了，完全符合我的預期！")

# 按鈕與分析邏輯
if st.button("開始進行情緒分析"):
    if user_input.strip():
        with st.spinner("分析中，請稍候..."):
            stars, score = analyze_review_stars(user_input)
        
        # 呈現預測結果
        st.subheader("分析結果")
        col1, col2 = st.columns(2)
        
        with col1:
            # 以數字與星號呈現
            st.metric(label="預估情緒等級", value=f"{stars} / 5 顆星")
            st.write("⭐️" * stars + "☆" * (5 - stars))
            
        with col2:
            st.metric(label="模型信心指數", value=f"{score:.2%}")
            
        # 根據星等給予對應的中文狀態提示
        if stars >= 4:
            st.success("這是一則正面的好評！")
        elif stars == 3:
            st.warning("這是一則中立偏普通的評價。")
        else:
            st.error("這是一則負面或需要改進的差評。")
    else:
        st.warning("請輸入有效的評語文字後再進行分析。")
