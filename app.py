import os
import time
import streamlit as st
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
from generation_model import LLMIntegration
import base64

st.set_page_config(page_title="CAG Chatbot", layout="wide", page_icon="🤖")

# Load API Key
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY", st.secrets.get("general", {}).get("HUGGINGFACE_API_KEY"))
if not api_key:
    st.error("🚫 API key missing. Add it to the secrets manager or .env file.")
    st.stop()



def set_background(image_path):
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Call the function in your script
set_background("img.jpg")


# Initialize LLM
llm_system = LLMIntegration(api_key=api_key)

# Initialize session state variables
if "stats" not in st.session_state:
    st.session_state.stats = {"hits": 0, "misses": 0, "response_times": [], "queries": []}



# Sidebar Controls
with st.sidebar:
    st.header("🔧 Settings")
    cache_size = st.slider("Cache Size", 50, 500, 100)
    similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8)
    if st.button("🗑️ Clear Cache"):
        llm_system.cache_manager.clear_cache()
        st.session_state.stats = {"hits": 0, "misses": 0, "response_times": [], "queries": []}
        st.success("✔️ Cache cleared successfully!")

# Main Interface
st.title("🤖 Cache Augmented Generation Chatbot")
st.write("Smart caching for enhanced responses.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💡 Chat")
    query = st.text_input("📝 Enter your query:")
    if query:
        start_time = time.time()
        cached_response = llm_system.cache_manager.get_from_cache(llm_system.cache_manager.normalize_key(query))
        
        if cached_response:
            response = cached_response
            st.session_state.stats["hits"] += 1
            st.success("📌 Cache Hit!")
        else:
            response = llm_system.generate_response(query)
            st.session_state.stats["misses"] += 1
            st.warning("⚠️ Cache Miss. Fetching from LLM...")
        
        response_time = time.time() - start_time
        st.session_state.stats["response_times"].append(response_time)
        st.session_state.stats["queries"].append({"query": query, "response": response, "time": response_time})
        
        st.success(f"💬 {response}")
        st.info(f"⏳ Response Time: {response_time:.2f} seconds")

    with st.expander("📜 Query History"):
        for entry in st.session_state.stats["queries"][-5:]:
            st.write(f"**🔍 Query:** {entry['query']}")
            st.write(f"**🗨️ Response:** {entry['response']}")
            st.write(f"⏱️ Time Taken: {entry['time']:.2f} seconds")
            st.write("---")

with col2:
    st.subheader("📊 Cache Statistics")
    total_queries = st.session_state.stats["hits"] + st.session_state.stats["misses"]
    hit_ratio = (st.session_state.stats["hits"] / total_queries) * 100 if total_queries else 0
    miss_ratio = (st.session_state.stats["misses"] / total_queries) * 100 if total_queries else 0
    
    st.metric("🎯 Hits", st.session_state.stats["hits"])
    st.metric("❌ Misses", st.session_state.stats["misses"])
    st.progress(hit_ratio / 100, text=f"📈 Cache Hit Ratio: {hit_ratio:.2f}%")
    st.progress(miss_ratio / 100, text=f"📉 Cache Miss Ratio: {miss_ratio:.2f}%")
    
    if st.session_state.stats["response_times"]:
        fig = px.line(
            x=[entry["query"] for entry in st.session_state.stats["queries"]],
            y=st.session_state.stats["response_times"],
            title="📊 Response Time Trend",
            labels={"x": "Query", "y": "Response Time (s)"}
        )
        st.plotly_chart(fig, use_container_width=True)
