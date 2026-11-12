import streamlit as st
from pathlib import Path
from model import generate_story

st.set_page_config(
    page_title="Story Studio",
    page_icon="📖",
    layout="centered",
)

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )

load_css()

st.markdown(
    """
    <div class="page-heading">
        <div class="eyebrow">AI WRITING SPACE</div>
        <h1>Story Studio</h1>
        <p>Give your story a beginning. Let AI help with what comes next.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">YOUR STORY</div>', unsafe_allow_html=True)

prompt = st.text_area(
    "Story opening",
    placeholder="One rainy evening, a student discovered a mysterious box...",
    height=180,
    label_visibility="collapsed",
)

col1, col2 = st.columns(2)
with col1:
    length_choice = st.selectbox("LENGTH", ["Short", "Long"])
with col2:
    temperature = st.slider("CREATIVITY", 0.2, 1.5, 0.8, 0.1)

max_new_tokens = 45 if length_choice == "Short" else 110

generate = st.button("Continue story  →", type="primary", use_container_width=True)

if generate:
    if not prompt.strip():
        st.warning("Add a story opening before continuing.")
    else:
        with st.spinner("Writing your continuation..."):
            try:
                continuation = generate_story(prompt, max_new_tokens, temperature)
                st.markdown('<div class="result-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">YOUR ORIGINAL OPENING</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="story-original">{prompt.strip()}</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label generated-label">AI CONTINUATION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="story-output">{continuation}</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error(str(exc))

st.markdown(
    '<div class="footer-note">Made for stories, ideas, and what-ifs.</div>',
    unsafe_allow_html=True,
)
