from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from app.core.config import PROJECT_ROOT, get_settings
from app.services.predictor import PredictorService
from src.data.target_normalization import normalize_target_frame


BANNER_PATH = PROJECT_ROOT / "Churn Prediction Engine.png"
AUTHOR_IMAGE_PATH = PROJECT_ROOT / "frontend" / "assets" / "okon_prince.png"
TESTING_PACK_PATH = PROJECT_ROOT / "artifacts" / "sample_outputs" / "model_testing_pack.csv"
APP_PAGES = ["Overview", "Predict", "Batch Scoring", "Model Insights", "About"]
YELLOW_SCALE = ["#3B2D00", "#7A5C00", "#D7A700", "#F4C430", "#FFE27A"]
DARK_THEME = {
    "bg": "#050505",
    "bg_soft": "#0d0d0d",
    "card": "rgba(18, 18, 18, 0.94)",
    "card_soft": "rgba(25, 25, 25, 0.96)",
    "ink": "#fff5bf",
    "muted": "#d8c77c",
    "accent": "#f4c430",
    "accent_soft": "rgba(244, 196, 48, 0.14)",
    "border": "rgba(244, 196, 48, 0.32)",
    "shadow": "rgba(244, 196, 48, 0.12)",
    "sidebar": "linear-gradient(180deg, #0a0a0a 0%, #131313 100%)",
    "app_bg": """
        radial-gradient(circle at top right, rgba(244, 196, 48, 0.18), transparent 22%),
        radial-gradient(circle at bottom left, rgba(244, 196, 48, 0.10), transparent 28%),
        linear-gradient(180deg, #050505 0%, #090909 100%)
    """,
    "chart_bg": "#111111",
    "chart_text": "#fff5bf",
    "widget_bg": "rgba(20, 20, 20, 0.96)",
    "widget_bg_soft": "rgba(30, 30, 30, 0.98)",
    "widget_ink": "#fff5bf",
    "caption_ink": "#d8c77c",
    "code_bg": "rgba(244, 196, 48, 0.14)",
    "code_ink": "#fff5bf",
    "glass_bg": "rgba(13, 13, 13, 0.28)",
    "glass_bg_strong": "rgba(18, 18, 18, 0.54)",
    "glass_shadow": "rgba(0, 0, 0, 0.35)",
    "primary_button_bg": "#f4c430",
    "primary_button_ink": "#050505",
    "primary_button_hover_bg": "#ffd54f",
    "score_low": "#ffde59",
    "score_medium": "#f4c430",
    "score_high": "#ff9f1c",
    "toolbar_bg": "rgba(22, 22, 22, 0.88)",
    "toolbar_ink": "#fff5bf",
    "toolbar_filter": "none",
    "toolbar_hover": "rgba(244, 196, 48, 0.14)",
    "toolbar_hover_ink": "#f4c430",
    "toolbar_shadow": "rgba(0, 0, 0, 0.35)",
}
LIGHT_THEME = {
    "bg": "#fffdf5",
    "bg_soft": "#fff7cf",
    "card": "rgba(255, 252, 237, 0.96)",
    "card_soft": "rgba(255, 248, 214, 0.98)",
    "ink": "#221b00",
    "muted": "#7a6400",
    "accent": "#b8860b",
    "accent_soft": "rgba(184, 134, 11, 0.12)",
    "border": "rgba(184, 134, 11, 0.28)",
    "shadow": "rgba(184, 134, 11, 0.16)",
    "sidebar": "linear-gradient(180deg, #fff7cf 0%, #fff0b8 100%)",
    "app_bg": """
        radial-gradient(circle at top right, rgba(244, 196, 48, 0.18), transparent 22%),
        radial-gradient(circle at bottom left, rgba(244, 196, 48, 0.12), transparent 28%),
        linear-gradient(180deg, #fffdf5 0%, #fff7de 100%)
    """,
    "chart_bg": "#fff8e6",
    "chart_text": "#050505",
    "widget_bg": "#ffffff",
    "widget_bg_soft": "#fffaf0",
    "widget_ink": "#050505",
    "caption_ink": "#050505",
    "code_bg": "#fff7de",
    "code_ink": "#050505",
    "glass_bg": "rgba(255, 255, 255, 0.22)",
    "glass_bg_strong": "rgba(255, 252, 237, 0.48)",
    "glass_shadow": "rgba(184, 134, 11, 0.18)",
    "primary_button_bg": "#050505",
    "primary_button_ink": "#ffffff",
    "primary_button_hover_bg": "#1a1a1a",
    "score_low": "#807000",
    "score_medium": "#b8860b",
    "score_high": "#d97706",
    "toolbar_bg": "rgba(255, 249, 226, 0.96)",
    "toolbar_ink": "#050505",
    "toolbar_filter": "brightness(0) saturate(100%)",
    "toolbar_hover": "rgba(184, 134, 11, 0.10)",
    "toolbar_hover_ink": "#050505",
    "toolbar_shadow": "rgba(184, 134, 11, 0.18)",
}


st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor() -> PredictorService:
    return PredictorService()


@st.cache_data
def load_training_data() -> pd.DataFrame:
    settings = get_settings()
    raw_frame = pd.read_csv(settings.raw_data_dir / "train.csv")
    return normalize_target_frame(raw_frame, settings.target_column, settings.target_normalization_map)


@st.cache_data
def load_metrics() -> dict:
    settings = get_settings()
    with settings.metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_testing_pack() -> pd.DataFrame:
    if TESTING_PACK_PATH.exists():
        return pd.read_csv(TESTING_PACK_PATH)
    return pd.DataFrame()


@st.cache_data
def encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def inject_styles(theme_mode: str) -> None:
    theme = DARK_THEME if theme_mode == "Dark" else LIGHT_THEME
    css = """
        <style>
        :root {{
            --bg: {bg};
            --bg-soft: {bg_soft};
            --card: {card};
            --card-soft: {card_soft};
            --ink: {ink};
            --muted: {muted};
            --accent: {accent};
            --accent-soft: {accent_soft};
            --border: {border};
            --shadow: {shadow};
            --widget-bg: {widget_bg};
            --widget-bg-soft: {widget_bg_soft};
            --widget-ink: {widget_ink};
            --caption-ink: {caption_ink};
            --code-bg: {code_bg};
            --code-ink: {code_ink};
            --glass-bg: {glass_bg};
            --glass-bg-strong: {glass_bg_strong};
            --glass-shadow: {glass_shadow};
            --primary-button-bg: {primary_button_bg};
            --primary-button-ink: {primary_button_ink};
            --primary-button-hover-bg: {primary_button_hover_bg};
        }}
        .stApp {{
            background: {app_bg};
            color: var(--ink);
        }}
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}
        [data-testid="stAppViewContainer"] > .main {{
            margin-left: 0 !important;
        }}
        .block-container {{
            padding-top: 1.75rem !important;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stToolbar"],
        [data-testid="stHeaderActionElements"] {{
            background: {toolbar_bg};
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.2rem 0.4rem;
            box-shadow: 0 10px 24px {toolbar_shadow};
            backdrop-filter: blur(12px);
        }}
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] a,
        [data-testid="stHeader"] [role="button"],
        header button,
        header a,
        header [role="button"] {{
            color: {toolbar_ink} !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
            filter: {toolbar_filter} !important;
            -webkit-filter: {toolbar_filter} !important;
            -webkit-text-fill-color: {toolbar_ink} !important;
        }}
        [data-testid="stToolbar"] *,
        [data-testid="stHeaderActionElements"] *,
        button[data-testid^="baseButton-header"],
        button[data-testid^="baseButton-headerNoPadding"],
        [data-testid="stToolbar"] a,
        [data-testid="stHeaderActionElements"] a {{
            color: {toolbar_ink} !important;
            fill: {toolbar_ink} !important;
            stroke: {toolbar_ink} !important;
        }}
        [data-testid="stHeader"] svg,
        [data-testid="stHeader"] svg *,
        [data-testid="stHeader"] path,
        [data-testid="stHeader"] circle,
        [data-testid="stHeader"] rect,
        [data-testid="stHeader"] line,
        [data-testid="stHeader"] polyline,
        [data-testid="stHeader"] polygon {{
            color: {toolbar_ink} !important;
            fill: {toolbar_ink} !important;
            stroke: {toolbar_ink} !important;
            opacity: 1 !important;
        }}
        [data-testid="stToolbar"] svg,
        [data-testid="stHeaderActionElements"] svg,
        [data-testid="stToolbar"] path,
        [data-testid="stHeaderActionElements"] path,
        button[data-testid^="baseButton-header"] svg,
        button[data-testid^="baseButton-headerNoPadding"] svg,
        button[data-testid^="baseButton-header"] path,
        button[data-testid^="baseButton-headerNoPadding"] path {{
            color: {toolbar_ink} !important;
            fill: {toolbar_ink} !important;
            stroke: {toolbar_ink} !important;
        }}
        [data-testid="stToolbar"] button:hover,
        [data-testid="stHeaderActionElements"] button:hover,
        button[data-testid^="baseButton-header"]:hover,
        button[data-testid^="baseButton-headerNoPadding"]:hover,
        [data-testid="stToolbar"] a:hover,
        [data-testid="stHeaderActionElements"] a:hover,
        [data-testid="stHeader"] button:hover,
        [data-testid="stHeader"] a:hover,
        [data-testid="stHeader"] [role="button"]:hover {{
            background: {toolbar_hover} !important;
            color: {toolbar_hover_ink} !important;
        }}
        [data-testid="stToolbar"] button:hover *,
        [data-testid="stHeaderActionElements"] button:hover *,
        button[data-testid^="baseButton-header"]:hover *,
        button[data-testid^="baseButton-headerNoPadding"]:hover *,
        [data-testid="stToolbar"] a:hover *,
        [data-testid="stHeaderActionElements"] a:hover *,
        [data-testid="stToolbar"] button:hover svg,
        [data-testid="stHeaderActionElements"] button:hover svg,
        [data-testid="stToolbar"] button:hover path,
        [data-testid="stHeaderActionElements"] button:hover path,
        button[data-testid^="baseButton-header"]:hover svg,
        button[data-testid^="baseButton-headerNoPadding"]:hover svg,
        button[data-testid^="baseButton-header"]:hover path,
        button[data-testid^="baseButton-headerNoPadding"]:hover path,
        [data-testid="stHeader"] button:hover svg,
        [data-testid="stHeader"] a:hover svg,
        [data-testid="stHeader"] [role="button"]:hover svg,
        [data-testid="stHeader"] button:hover svg *,
        [data-testid="stHeader"] a:hover svg *,
        [data-testid="stHeader"] [role="button"]:hover svg *,
        [data-testid="stHeader"] button:hover path,
        [data-testid="stHeader"] a:hover path,
        [data-testid="stHeader"] [role="button"]:hover path {{
            color: {toolbar_hover_ink} !important;
            fill: {toolbar_hover_ink} !important;
            stroke: {toolbar_hover_ink} !important;
        }}
        [data-testid="stSidebar"] {{
            background: {sidebar};
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--ink);
        }}
        h1, h2, h3, h4, h5 {{
            color: var(--accent);
            letter-spacing: 0.02em;
        }}
        .hero-card, .content-card, .author-card, .footer-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: 0 18px 40px var(--shadow);
        }}
        .hero-card {{
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }}
        .content-card {{
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
        }}
        .content-card ul {{
            margin: 0.65rem 0 0;
            padding-left: 1.25rem;
        }}
        .content-card li {{
            margin-bottom: 0.55rem;
        }}
        .content-card li:last-child {{
            margin-bottom: 0;
        }}
        .metric-card {{
            background: var(--card-soft);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
            min-height: 130px;
        }}
        .metric-title {{
            font-size: 0.86rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.35rem;
        }}
        .metric-note {{
            font-size: 0.95rem;
            color: var(--ink);
        }}
        .project-banner {{
            width: 100%;
            border-radius: 22px;
            border: 1px solid var(--border);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.32);
            margin-bottom: 1rem;
        }}
        .app-shell {{
            position: relative;
            margin-bottom: 2.8rem;
        }}
        .floating-appearance-panel {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            left: auto;
            z-index: 8;
            display: inline-flex;
            align-items: center;
            gap: 0.28rem;
            padding: 0.26rem;
            border-radius: 999px;
            background: var(--glass-bg-strong);
            border: 1px solid var(--border);
            box-shadow: 0 14px 28px var(--glass-shadow);
            backdrop-filter: blur(18px);
            pointer-events: auto;
        }}
        .appearance-link {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.6rem;
            height: 2.6rem;
            min-width: 2.6rem;
            padding: 0;
            border-radius: 999px;
            color: var(--widget-ink);
            text-decoration: none !important;
            border: 1px solid transparent;
            -webkit-text-fill-color: var(--widget-ink);
            transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
        }}
        .appearance-link svg {{
            width: 1.08rem;
            height: 1.08rem;
            color: currentColor;
            fill: none;
            stroke: currentColor;
            stroke-width: 1.8;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}
        .appearance-link:link,
        .appearance-link:visited,
        .appearance-link:hover,
        .appearance-link:active {{
            text-decoration: none !important;
        }}
        .appearance-link:hover {{
            background: var(--glass-bg);
            color: var(--accent);
        }}
        .appearance-link.active {{
            background: var(--glass-bg);
            color: var(--accent);
            border-color: var(--border);
        }}
        .banner-shell {{
            position: relative;
            margin-bottom: 4rem;
        }}
        .banner-shell .project-banner {{
            margin-bottom: 0;
        }}
        .banner-nav {{
            position: absolute;
            left: 50%;
            bottom: -1.55rem;
            transform: translateX(-50%);
            width: min(94%, 1220px);
            display: flex;
            justify-content: center;
            gap: 0.55rem;
            flex-wrap: nowrap;
            overflow-x: auto;
            overflow-y: visible;
            padding: 0 0.35rem 0.2rem;
            scrollbar-width: none;
            z-index: 4;
        }}
        .banner-nav::-webkit-scrollbar {{
            display: none;
        }}
        .glass-nav-button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 0;
            flex: 1 1 0;
            max-width: 10.6rem;
            padding: 0.68rem 0.85rem;
            border-radius: 999px;
            color: var(--widget-ink);
            text-decoration: none !important;
            background: var(--glass-bg);
            border: 1px solid var(--border);
            box-shadow: 0 16px 36px var(--glass-shadow);
            backdrop-filter: blur(18px);
            white-space: nowrap;
            font-size: 0.94rem;
            line-height: 1.1;
            -webkit-text-fill-color: var(--widget-ink);
            transition: transform 0.18s ease, background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
        }}
        .glass-nav-button:link,
        .glass-nav-button:visited,
        .glass-nav-button:hover,
        .glass-nav-button:active {{
            text-decoration: none !important;
        }}
        .glass-nav-button:hover {{
            transform: translateY(-1px);
            background: var(--glass-bg-strong);
            color: var(--accent);
        }}
        .glass-nav-button.active {{
            background: var(--glass-bg-strong);
            color: var(--accent);
            border-color: var(--accent);
        }}
        .recommendation-card {{
            background: linear-gradient(90deg, rgba(244, 196, 48, 0.12), rgba(244, 196, 48, 0.03));
            border-left: 5px solid var(--accent);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.7rem;
            color: var(--ink);
        }}
        .author-grid {{
            display: grid;
            grid-template-columns: minmax(240px, 300px) 1fr;
            gap: 1.6rem;
            align-items: center;
        }}
        .author-photo-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
        }}
        .author-photo {{
            width: 100%;
            max-width: 280px;
            border-radius: 24px;
            border: 2px solid var(--accent);
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.35);
        }}
        .author-name {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.25rem;
        }}
        .author-role {{
            font-size: 1rem;
            color: var(--muted);
            margin-bottom: 1rem;
        }}
        .footer-card {{
            padding: 1rem 1.2rem;
            text-align: center;
            margin-top: 1.5rem;
            color: var(--muted);
        }}
        .footer-card p {{
            margin: 0.18rem 0;
        }}
        .caption-note {{
            color: var(--caption-ink);
            font-size: 0.95rem;
        }}
        .stMarkdown code,
        p code,
        li code {{
            background: var(--code-bg) !important;
            color: var(--code-ink) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.5rem !important;
            padding: 0.12rem 0.38rem !important;
            -webkit-text-fill-color: var(--code-ink) !important;
        }}
        .stMarkdown pre,
        .stMarkdown pre code {{
            background: var(--widget-bg-soft) !important;
            color: var(--code-ink) !important;
            border-color: var(--border) !important;
            -webkit-text-fill-color: var(--code-ink) !important;
        }}
        .stDownloadButton > button,
        .stButton > button,
        button[data-testid^="baseButton-secondary"] {{
            background: var(--widget-bg) !important;
            color: var(--widget-ink) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 10px 24px var(--shadow) !important;
        }}
        .stDownloadButton > button:hover,
        .stButton > button:hover,
        button[data-testid^="baseButton-secondary"]:hover {{
            background: var(--widget-bg-soft) !important;
            color: var(--widget-ink) !important;
        }}
        button[data-testid^="baseButton-primary"] {{
            background: var(--primary-button-bg) !important;
            color: var(--primary-button-ink) !important;
            border: 1px solid var(--primary-button-bg) !important;
            box-shadow: 0 10px 24px var(--shadow) !important;
        }}
        .stFormSubmitButton button,
        .stFormSubmitButton button *,
        button[data-testid="stBaseButton-primaryFormSubmit"],
        button[data-testid="stBaseButton-primaryFormSubmit"] *,
        button[data-testid="stBaseButton-secondaryFormSubmit"],
        button[data-testid="stBaseButton-secondaryFormSubmit"] *,
        button[data-testid="stBaseButton-tertiaryFormSubmit"],
        button[data-testid="stBaseButton-tertiaryFormSubmit"] * {{
            color: var(--primary-button-ink) !important;
            -webkit-text-fill-color: var(--primary-button-ink) !important;
        }}
        .stFormSubmitButton button,
        button[data-testid="stBaseButton-primaryFormSubmit"] {{
            background: var(--primary-button-bg) !important;
            border: 1px solid var(--primary-button-bg) !important;
            box-shadow: 0 10px 24px var(--shadow) !important;
        }}
        button[data-testid^="baseButton-primary"]:hover {{
            background: var(--primary-button-hover-bg) !important;
            color: var(--primary-button-ink) !important;
            border: 1px solid var(--primary-button-hover-bg) !important;
        }}
        .stFormSubmitButton button:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover {{
            background: var(--primary-button-hover-bg) !important;
            color: var(--primary-button-ink) !important;
            border: 1px solid var(--primary-button-hover-bg) !important;
        }}
        [data-testid="stWidgetLabel"] {{
            color: var(--widget-ink) !important;
        }}
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {{
            background: var(--widget-bg) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 10px 24px var(--shadow) !important;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] p,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] span {{
            color: var(--widget-ink) !important;
            -webkit-text-fill-color: var(--widget-ink) !important;
        }}
        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="base-input"] input::placeholder,
        div[data-baseweb="select"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {{
            color: var(--widget-ink) !important;
            -webkit-text-fill-color: var(--widget-ink) !important;
            opacity: 0.68 !important;
        }}
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg,
        div[data-baseweb="base-input"] svg {{
            color: var(--widget-ink) !important;
            fill: var(--widget-ink) !important;
            stroke: var(--widget-ink) !important;
        }}
        div[data-baseweb="input"] button,
        div[data-baseweb="base-input"] button {{
            background: var(--widget-bg-soft) !important;
            color: var(--widget-ink) !important;
            border-left: 1px solid var(--border) !important;
        }}
        div[data-baseweb="input"] button:hover,
        div[data-baseweb="base-input"] button:hover {{
            background: var(--widget-bg) !important;
            color: var(--widget-ink) !important;
        }}
        [role="listbox"] {{
            background: var(--widget-bg) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 14px 32px var(--shadow) !important;
        }}
        [role="option"] {{
            background: var(--widget-bg) !important;
            color: var(--widget-ink) !important;
        }}
        [role="option"]:hover {{
            background: var(--widget-bg-soft) !important;
            color: var(--widget-ink) !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--widget-bg) !important;
            border: 1px solid var(--border) !important;
            color: var(--widget-ink) !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: var(--widget-ink) !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--widget-bg-soft) !important;
            color: var(--widget-ink) !important;
            border: 1px solid var(--border) !important;
        }}
        @media (max-width: 900px) {{
            .floating-appearance-panel {{
                top: 0.75rem;
                right: 0.75rem;
            }}
            .banner-nav {{
                width: calc(100% - 1rem);
                bottom: -1.8rem;
            }}
            .glass-nav-button {{
                flex: 0 0 auto;
                max-width: none;
                padding: 0.62rem 0.8rem;
                font-size: 0.9rem;
            }}
        }}
        </style>
    """.format_map(theme)
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def inject_header_runtime_fix(theme_mode: str) -> None:
    toolbar_ink = "#050505" if theme_mode == "Light" else "#fff5bf"
    toolbar_hover = "#050505" if theme_mode == "Light" else "#f4c430"
    toolbar_filter = "brightness(0) saturate(100%)" if theme_mode == "Light" else "none"
    components.html(
        f"""
        <script>
        (function() {{
            const ink = "{toolbar_ink}";
            const hoverInk = "{toolbar_hover}";
            const iconFilter = "{toolbar_filter}";

            function styleControl(el) {{
                if (!el) return;
                el.style.color = ink;
                el.style.opacity = "1";
                el.style.filter = iconFilter;
                el.style.webkitFilter = iconFilter;
                el.style.webkitTextFillColor = ink;

                const iconNodes = el.querySelectorAll("svg, svg *, path, circle, rect, line, polyline, polygon, img");
                iconNodes.forEach((node) => {{
                    node.style.color = ink;
                    node.style.fill = ink;
                    node.style.stroke = ink;
                    node.style.opacity = "1";
                    node.style.filter = iconFilter;
                    node.style.webkitFilter = iconFilter;
                }});

                el.onmouseenter = () => {{
                    el.style.color = hoverInk;
                    el.style.filter = iconFilter;
                    el.style.webkitFilter = iconFilter;
                    el.style.webkitTextFillColor = hoverInk;
                    iconNodes.forEach((node) => {{
                        node.style.color = hoverInk;
                        node.style.fill = hoverInk;
                        node.style.stroke = hoverInk;
                        node.style.opacity = "1";
                        node.style.filter = iconFilter;
                        node.style.webkitFilter = iconFilter;
                    }});
                }};

                el.onmouseleave = () => {{
                    el.style.color = ink;
                    el.style.filter = iconFilter;
                    el.style.webkitFilter = iconFilter;
                    el.style.webkitTextFillColor = ink;
                    iconNodes.forEach((node) => {{
                        node.style.color = ink;
                        node.style.fill = ink;
                        node.style.stroke = ink;
                        node.style.opacity = "1";
                        node.style.filter = iconFilter;
                        node.style.webkitFilter = iconFilter;
                    }});
                }};
            }}

            function getHeaderControls(doc) {{
                const selectors = [
                    "header button",
                    "header a",
                    "header [role='button']",
                    "[data-testid*='Header'] button",
                    "[data-testid*='Header'] a",
                    "[data-testid*='Toolbar'] button",
                    "[data-testid*='Toolbar'] a",
                ];

                const seen = new Set();
                const matches = [];

                selectors.forEach((selector) => {{
                    doc.querySelectorAll(selector).forEach((el) => {{
                        const rect = el.getBoundingClientRect();
                        if (rect.top < 140 && rect.right > (window.parent.innerWidth || window.innerWidth) - 520) {{
                            if (!seen.has(el)) {{
                                seen.add(el);
                                matches.push(el);
                            }}
                        }}
                    }});
                }});

                doc.querySelectorAll("button, a, [role='button']").forEach((el) => {{
                    const rect = el.getBoundingClientRect();
                    if (rect.top < 140 && rect.right > (window.parent.innerWidth || window.innerWidth) - 520) {{
                        const hasIcon = el.querySelector("svg, img");
                        if (hasIcon && !seen.has(el)) {{
                            seen.add(el);
                            matches.push(el);
                        }}
                    }}
                }});

                return matches;
            }}

            function applyFix() {{
                let doc = document;
                let parentDoc = null;
                try {{
                    if (window.parent && window.parent.document) {{
                        doc = window.parent.document;
                        parentDoc = window.parent.document;
                    }}
                }} catch (error) {{
                    doc = document;
                    parentDoc = null;
                }}

                getHeaderControls(doc).forEach(styleControl);

            }}

            applyFix();
            setTimeout(applyFix, 250);
            setTimeout(applyFix, 1000);
            setTimeout(applyFix, 2500);

            const observer = new MutationObserver(() => applyFix());
            observer.observe(document.body, {{ childList: true, subtree: true }});

            try {{
                if (window.parent && window.parent.document && window.parent.document.body) {{
                    const parentObserver = new MutationObserver(() => applyFix());
                    parentObserver.observe(window.parent.document.body, {{ childList: true, subtree: true }});
                }}
            }} catch (error) {{
                console.debug("Header runtime fix parent access unavailable.");
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def current_theme(theme_mode: str) -> dict[str, str]:
    return DARK_THEME if theme_mode == "Dark" else LIGHT_THEME


def apply_plotly_theme(fig, theme: dict[str, str]):
    fig.update_layout(
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        font_color=theme["chart_text"],
        title_font_color=theme["chart_text"],
        legend_font_color=theme["chart_text"],
        hoverlabel={"font": {"color": theme["chart_text"]}},
        xaxis={
            "tickfont": {"color": theme["chart_text"]},
            "title": {"font": {"color": theme["chart_text"]}},
        },
        yaxis={
            "tickfont": {"color": theme["chart_text"]},
            "title": {"font": {"color": theme["chart_text"]}},
        },
        coloraxis_colorbar={
            "tickfont": {"color": theme["chart_text"]},
            "title": {"font": {"color": theme["chart_text"]}},
        },
    )
    try:
        fig.update_traces(textfont_color=theme["chart_text"])
    except ValueError:
        pass
    fig.update_annotations(font_color=theme["chart_text"])
    return fig


def render_banner() -> None:
    if BANNER_PATH.exists():
        banner_b64 = encode_image(str(BANNER_PATH))
        st.markdown(
            f"<img class='project-banner' src='data:image/png;base64,{banner_b64}' alt='Churn Prediction Engine banner' />",
            unsafe_allow_html=True,
        )


def _query_value(value: str | list[str] | None, default: str) -> str:
    if isinstance(value, list):
        return value[0] if value else default
    if value is None:
        return default
    return str(value)


def app_shell_state() -> tuple[str, str]:
    page = _query_value(st.query_params.get("page"), "Overview")
    theme_mode = _query_value(st.query_params.get("theme"), "Dark")
    if page not in APP_PAGES:
        page = "Overview"
    if theme_mode not in {"Dark", "Light"}:
        theme_mode = "Dark"
    return page, theme_mode


def app_href(page: str, theme_mode: str) -> str:
    return f"?{urlencode({'page': page, 'theme': theme_mode})}"


def appearance_icon(mode: str) -> str:
    if mode == "Dark":
        return (
            "<svg viewBox='0 0 24 24' aria-hidden='true' focusable='false'>"
            "<path d='M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z'></path>"
            "</svg>"
        )
    return (
        "<svg viewBox='0 0 24 24' aria-hidden='true' focusable='false'>"
        "<circle cx='12' cy='12' r='4'></circle>"
        "<path d='M12 2v2.2M12 19.8V22M4.9 4.9l1.6 1.6M17.5 17.5l1.6 1.6M2 12h2.2M19.8 12H22M4.9 19.1l1.6-1.6M17.5 6.5l1.6-1.6'></path>"
        "</svg>"
    )


def render_app_shell(current_page: str, theme_mode: str) -> None:
    appearance_links = []
    for option in ["Dark", "Light"]:
        active_class = "active" if option == theme_mode else ""
        appearance_links.append(
            f"<a class='appearance-link {active_class}' href='{app_href(current_page, option)}' target='_self' aria-label='Switch to {option} mode'>{appearance_icon(option)}</a>"
        )

    nav_links = []
    for page in APP_PAGES:
        active_class = "active" if page == current_page else ""
        nav_links.append(
            f"<a class='glass-nav-button {active_class}' href='{app_href(page, theme_mode)}' target='_self'>{page}</a>"
        )

    if BANNER_PATH.exists():
        banner_b64 = encode_image(str(BANNER_PATH))
        banner_visual = (
            f"<img class='project-banner' src='data:image/png;base64,{banner_b64}' alt='Churn Prediction Engine banner' />"
        )
    else:
        banner_visual = "<div class='project-banner' style='height:240px;background:var(--glass-bg-strong);'></div>"

    st.markdown(
        f"""
        <div class="app-shell">
            <div class="floating-appearance-panel">
                {''.join(appearance_links)}
            </div>
            <div class="banner-shell">
                {banner_visual}
                <div class="banner-nav">
                    {''.join(nav_links)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_summary_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="content-card">
            <h3 style="margin-bottom:0.35rem;">{title}</h3>
            <p style="font-size:1.45rem;font-weight:700;color:var(--accent);margin-bottom:0.35rem;">{value}</p>
            <p style="margin-bottom:0;">{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-card">
            <p>© Okon Prince, 2026</p>
            <p>This project is built using a customer churn dataset hosted on Kaggle by ohmammamia Karina.</p>
            <p>The project is released under the MIT License.</p>
            <p>For enquiries, please contact okonp07@gmail.com</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_page(predictor: PredictorService, metrics_payload: dict) -> None:
    model_info = predictor.model_info()
    validation_metrics = model_info["validation_metrics"]
    task_detection = model_info["task_detection"]
    normalization_info = model_info.get("target_normalization") or {}
    feature_summary = model_info["feature_summary"]
    training_report = metrics_payload["training_validation_report"]
    test_report = metrics_payload["test_validation_report"]
    top_features = [item["base_feature"] for item in model_info["global_feature_importance"][:3]]
    source_classes = normalization_info.get("source_classes", task_detection.get("source_classes", task_detection["classes"]))
    normalized_classes = normalization_info.get("normalized_classes", task_detection["classes"])

    st.markdown(
        """
        <div class="hero-card">
            <p>
                This platform translates customer behavior into practical retention intelligence. It scores churn risk,
                estimates confidence, surfaces business-friendly drivers, and recommends concrete actions teams can take
                before valuable customers quietly disappear.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Best Model", model_info["best_model_name"], "Selected after multi-model benchmarking.")
    with metric_columns[1]:
        render_metric_card("Weighted F1", f"{validation_metrics.get('f1_weighted', 0):.3f}", "Balances performance across the churn tiers.")
    with metric_columns[2]:
        render_metric_card("Accuracy", f"{validation_metrics.get('accuracy', 0):.3f}", "Holdout validation accuracy on unseen records.")
    with metric_columns[3]:
        render_metric_card(
            "QWK",
            f"{validation_metrics.get('quadratic_weighted_kappa', validation_metrics.get('r2', 0)):.3f}",
            "Ordinal alignment between predicted and true risk levels.",
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="content-card">
            <h3>Why this system matters</h3>
            <p>
                Churn is costly because customers rarely announce that they are leaving. They reduce engagement,
                complain more often, spend less, or simply stop returning. This solution helps teams detect those
                warning signs early enough to intervene with meaningful, personalized retention actions.
            </p>
            <h3>How the solution works</h3>
            <p>
                The system validates incoming records, removes leakage-prone identifiers, engineers behavior and
                lifecycle features, evaluates several candidate models, saves the best-performing artifact, and uses
                that same artifact consistently for API and Streamlit inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <h3>Detected modeling strategy</h3>
            <p>{}</p>
        </div>
        """.format(task_detection["strategy_details"]["notes"]),
        unsafe_allow_html=True,
    )

    task_type_label = (
        "Ordinal Multiclass"
        if task_detection["task_type"] == "ordinal_multiclass_classification"
        else task_detection["task_type"].replace("_", " ").title()
    )
    st.markdown(
        f"""
        <div class="content-card">
            <h3>At a Glance</h3>
            <ul>
                <li><strong>Task Type:</strong> {task_type_label} with {len(task_detection['classes'])} ordered classes detected in the target.</li>
                <li><strong>Business Score Scale:</strong> The deployed system reports churn from {normalized_classes[0]} (lowest risk) to {normalized_classes[-1]} (highest risk).</li>
                <li><strong>Candidate Models:</strong> {metrics_payload["candidate_model_count"]} models were benchmarked under the same validation protocol.</li>
                <li><strong>Top Drivers:</strong> {", ".join(top_features)} were the most influential features in the final model.</li>
                <li><strong>Reference Date:</strong> {feature_summary["reference_date"]} was used to compute tenure consistently across training and inference.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <h3>Model Summary</h3>
            <p>
                The final XGBoost model was chosen because it delivered the strongest balance of predictive performance
                and ordinal consistency on unseen validation data. The model preserves the ordered meaning of churn risk
                tiers instead of flattening them into a simple binary label.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Key Insights")
    st.markdown(
        f"""
        - The target was automatically detected as an ordinal multiclass problem on the normalized classes `{task_detection['classes']}`. These classes are ordered from lower churn risk to higher churn risk, so a score closer to `1` is more desirable and a score closer to `5` means the customer is at much higher risk of churning.
        - The original dataset encoded churn as `{source_classes}`. To make the deployed score easier for business users to interpret, the pipeline normalizes those labels to `{normalized_classes}` by merging the raw `-1` tier into score `1`.
        - In practical terms, `1` represents the safest customers, `2` and `3` indicate increasing concern, `4` signals high risk, and `5` marks the highest churn warning level.
        - `XGBoost` was selected from `{metrics_payload['candidate_model_count']}` candidate models after model comparison.
        - Validation performance was strong with `Weighted F1 = {validation_metrics.get('f1_weighted', 0):.3f}`, `Accuracy = {validation_metrics.get('accuracy', 0):.3f}`, and `QWK = {validation_metrics.get('quadratic_weighted_kappa', 0):.3f}`.
        - The most important risk signals were `{top_features[0]}`, `{top_features[1]}`, and `{top_features[2]}`, which align with customer value, feedback, and membership behavior.
        - The largest missing-data areas were `region_category ({training_report['null_summary']['region_category']})`, `points_in_wallet ({training_report['null_summary']['points_in_wallet']})`, and `preferred_offer_types ({training_report['null_summary']['preferred_offer_types']})`, and the pipeline handles them safely during training and scoring.
        - Training and inference share the same engineered feature logic, including tenure, activity, complaints, wallet behavior, and engagement segmentation.
        """,
    )

    with st.expander("Technical Details"):
        detail_columns = st.columns(2)
        with detail_columns[0]:
            st.markdown("**Validation Metrics**")
            st.json(validation_metrics)
            st.markdown("**Task Detection**")
            st.json(task_detection)
            if normalization_info:
                st.markdown("**Target Normalization**")
                st.json(normalization_info)
        with detail_columns[1]:
            st.markdown("**Training Data Validation**")
            st.json(training_report)
            st.markdown("**Test Data Validation**")
            st.json(test_report)
        st.markdown("**Feature Summary**")
        st.json(feature_summary)


def render_prediction_page(predictor: PredictorService, theme_mode: str) -> None:
    theme = current_theme(theme_mode)
    st.subheader("Manual Customer Prediction")
    st.markdown("<p class='caption-note'>Enter a full customer profile and the system will return risk, confidence, key drivers, and retention recommendations.</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        payload = {
            "customer_id": col1.text_input("Customer ID", value="demo-customer-001"),
            "Name": col1.text_input("Name", value="Ava Johnson"),
            "age": col1.number_input("Age", min_value=10, max_value=90, value=35),
            "gender": col1.selectbox("Gender", ["F", "M", "Unknown"], index=0),
            "security_no": col1.text_input("Security No", value="SEC-DEMO-001"),
            "region_category": col2.selectbox("Region", ["Town", "City", "Village", "Missing"], index=1),
            "membership_category": col2.selectbox(
                "Membership",
                [
                    "No Membership",
                    "Basic Membership",
                    "Silver Membership",
                    "Gold Membership",
                    "Premium Membership",
                    "Platinum Membership",
                ],
                index=4,
            ),
            "joining_date": col2.date_input("Joining Date", value=pd.Timestamp("2017-05-10")).isoformat(),
            "joined_through_referral": col2.selectbox("Joined via referral", ["Yes", "No", "Unknown"], index=1),
            "referral_id": col2.text_input("Referral ID", value="CID-REF-001"),
            "preferred_offer_types": col2.selectbox(
                "Offer Preference",
                ["Gift Vouchers/Coupons", "Credit/Debit Card Offers", "Without Offers", "Missing"],
                index=0,
            ),
            "medium_of_operation": col3.selectbox("Primary Device", ["Desktop", "Smartphone", "Both", "Missing"], index=1),
            "internet_option": col3.selectbox("Internet Type", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"], index=0),
            "last_visit_time": col3.text_input("Last Visit Time", value="21:15:00"),
            "days_since_last_login": col3.number_input("Days Since Last Login", min_value=0.0, max_value=90.0, value=19.0),
            "avg_time_spent": col3.number_input("Average Time Spent", min_value=0.0, max_value=5000.0, value=48.0),
            "avg_transaction_value": st.number_input(
                "Average Transaction Value", min_value=0.0, max_value=100000.0, value=12500.0
            ),
            "avg_frequency_login_days": st.text_input("Avg Frequency Login Days", value="12"),
            "points_in_wallet": st.number_input("Points In Wallet", min_value=0.0, max_value=5000.0, value=240.0),
            "used_special_discount": st.selectbox("Used Special Discount", ["Yes", "No"], index=0),
            "offer_application_preference": st.selectbox("Offer Application Preference", ["Yes", "No"], index=1),
            "past_complaint": st.selectbox("Past Complaint", ["Yes", "No"], index=0),
            "complaint_status": st.selectbox(
                "Complaint Status",
                ["Unsolved", "Solved", "Solved in Follow-up", "Not Applicable", "No Information Available"],
                index=0,
            ),
            "feedback": st.selectbox(
                "Feedback",
                [
                    "Poor Customer Service",
                    "Poor Product Quality",
                    "Poor Website",
                    "Too many ads",
                    "Reasonable Price",
                    "Products always in Stock",
                    "Quality Customer Care",
                    "User Friendly Website",
                    "No reason specified",
                ],
                index=0,
            ),
        }
        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        result = predictor.predict_record(payload)
        score_color = (
            theme["score_low"]
            if result["risk_band"] == "Low"
            else theme["score_medium"]
            if result["risk_band"] == "Medium"
            else theme["score_high"]
        )
        st.markdown(
            f"""
            <div class="content-card">
                <h3>Prediction Result</h3>
                <p><strong>Predicted Class:</strong> {result['predicted_class']} ({result['predicted_label']})</p>
                <p><strong>Risk Score:</strong> <span style="color:{score_color};">{result['risk_score']:.3f}</span></p>
                <p><strong>Risk Band:</strong> {result['risk_band']}</p>
                <p><strong>Confidence:</strong> {result['confidence'] if result['confidence'] is not None else 'N/A'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Top Risk Drivers")
        for driver in result["top_risk_drivers"]:
            st.write(f"- {driver}")

        st.subheader("Retention Recommendations")
        for recommendation in result["recommendations"]:
            st.markdown(f"<div class='recommendation-card'>{recommendation}</div>", unsafe_allow_html=True)

        st.subheader("Probability Breakdown")
        probability_frame = pd.DataFrame(
            {
                "class": list(result["probability_breakdown"].keys()),
                "probability": list(result["probability_breakdown"].values()),
            }
        )
        st.plotly_chart(
            apply_plotly_theme(
                px.bar(
                probability_frame,
                x="class",
                y="probability",
                color="probability",
                color_continuous_scale=YELLOW_SCALE,
                title="Per-Class Probability Distribution",
                ),
                theme,
            ),
            use_container_width=True,
        )


def render_batch_page(predictor: PredictorService) -> None:
    st.subheader("Batch Scoring")
    st.markdown(
        "<p class='caption-note'>Upload a CSV of customer records or download the built-in 10-row testing pack to try the model immediately.</p>",
        unsafe_allow_html=True,
    )

    testing_pack = load_testing_pack()
    if not testing_pack.empty:
        action_col, preview_col = st.columns([1, 1.4])
        with action_col:
            st.download_button(
                label="Download Model Testing Pack",
                data=testing_pack.to_csv(index=False).encode("utf-8"),
                file_name="model_testing_pack.csv",
                mime="text/csv",
            )
        with preview_col:
            st.markdown("<p class='caption-note'>Testing pack contains 10 complete customer profiles with all required fields.</p>", unsafe_allow_html=True)
        with st.expander("Preview testing pack"):
            st.dataframe(testing_pack, use_container_width=True)

    upload = st.file_uploader("Upload a CSV file with customer records", type=["csv"])
    if upload is not None:
        frame = pd.read_csv(upload)
        predictions = predictor.predict_batch(frame)
        prediction_frame = pd.DataFrame(predictions)
        st.dataframe(prediction_frame, use_container_width=True)
        st.download_button(
            label="Download Scored Results",
            data=prediction_frame.to_csv(index=False).encode("utf-8"),
            file_name="batch_churn_predictions.csv",
            mime="text/csv",
        )


def render_insights_page(predictor: PredictorService, theme_mode: str) -> None:
    theme = current_theme(theme_mode)
    st.subheader("Model Insights")
    train_df = load_training_data()
    model_info = predictor.model_info()
    importance_frame = pd.DataFrame(model_info["global_feature_importance"])
    validation_metrics = model_info["validation_metrics"]
    class_labels = [str(item) for item in model_info["task_detection"]["classes"]]

    histogram = px.histogram(
        train_df,
        x="churn_risk_score",
        color="churn_risk_score",
        title="Observed Churn Risk Distribution (Normalized 1-5 Scale)",
        color_discrete_sequence=["#f4c430", "#eab308", "#ca8a04", "#fde68a", "#facc15", "#f59e0b"],
    )
    apply_plotly_theme(histogram, theme)
    st.plotly_chart(histogram, use_container_width=True)
    st.markdown(
        """
        <div class="content-card">
            <h3>What this chart means</h3>
            <p>
                This chart shows how customers are currently spread across the churn risk classes in the training
                data. Taller bars mean more customers fall into that risk level. A non-technical viewer can read this
                as the model's starting view of the business: which churn levels are common, which are rare, and
                whether the system is mostly dealing with low-risk, medium-risk, or high-risk customers on the final
                1-to-5 score scale.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    importance_chart = px.bar(
        importance_frame.head(12),
        x="importance",
        y="base_feature",
        orientation="h",
        color="importance",
        color_continuous_scale=YELLOW_SCALE,
        title="Top Global Feature Drivers",
    )
    apply_plotly_theme(importance_chart, theme)
    st.plotly_chart(importance_chart, use_container_width=True)
    st.markdown(
        """
        <div class="content-card">
            <h3>What this chart means</h3>
            <p>
                This chart shows which customer signals influenced the model the most overall. Longer bars mean the
                model relied more heavily on that factor when estimating churn risk. For a business audience, this is
                the quickest way to see what the system pays attention to most, such as membership behavior, feedback,
                complaints, or engagement. These are importance signals, not proof of cause, but they are strong clues
                for where teams should focus retention action.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confusion_matrix_values = validation_metrics.get("confusion_matrix", [])
    if confusion_matrix_values:
        confusion_df = pd.DataFrame(confusion_matrix_values, index=class_labels, columns=class_labels)
        confusion_chart = px.imshow(
            confusion_df,
            text_auto=True,
            color_continuous_scale=YELLOW_SCALE,
            aspect="auto",
            title="Validation Confusion Matrix",
        )
        apply_plotly_theme(confusion_chart, theme)
        confusion_chart.update_layout(
            coloraxis_colorbar_title="Count",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
        )
        st.plotly_chart(confusion_chart, use_container_width=True)
        st.markdown(
            """
            <div class="content-card">
                <h3>What this chart means</h3>
                <p>
                    This grid compares the true churn class of each customer with the class predicted by the model.
                    The strongest performance appears along the diagonal, where predicted and actual classes match.
                    Numbers away from that diagonal are the model's mistakes. In simple terms, this chart helps a
                    lay person see whether the system is correctly separating safer customers from riskier ones and
                    which risk bands the model still tends to confuse.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        actual_counts = confusion_df.sum(axis=1).reset_index()
        actual_counts.columns = ["class", "count"]
        actual_counts["type"] = "Actual"
        predicted_counts = confusion_df.sum(axis=0).reset_index()
        predicted_counts.columns = ["class", "count"]
        predicted_counts["type"] = "Predicted"
        support_df = pd.concat([actual_counts, predicted_counts], ignore_index=True)
        support_chart = px.bar(
            support_df,
            x="class",
            y="count",
            color="type",
            barmode="group",
            color_discrete_sequence=["#f4c430", "#7a5c00"],
            title="Actual vs Predicted Class Support",
        )
        apply_plotly_theme(support_chart, theme)
        support_chart.update_layout(
            xaxis_title="Risk Class",
            yaxis_title="Count",
        )
        st.plotly_chart(support_chart, use_container_width=True)
        st.markdown(
            """
            <div class="content-card">
                <h3>What this chart means</h3>
                <p>
                    This chart compares the number of customers that truly belong to each risk class with the number
                    the model predicted for each class. When the paired bars are close, the system is capturing the
                    business's real risk mix well. When they are far apart, the model is over-predicting or
                    under-predicting some customer groups. That makes this chart especially useful for spotting bias
                    toward certain churn levels.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="content-card">
            <h3>Next Steps</h3>
            <p>
                The best path to stronger performance is to improve both signal quality and decision discipline. That
                means expanding the data with richer engagement trends, transaction history patterns, service-touch
                history, and more recent behavior snapshots; running deeper hyperparameter tuning with stricter
                cross-validation so gains are real and not overfitted; testing additional ordinal-aware or calibrated
                ensemble approaches; and improving class-balance handling for the rarest risk tiers. Beyond model
                accuracy, the system can be made better by adding drift monitoring, scheduled retraining, retention
                campaign outcome tracking, human review loops for high-risk cases, and feedback collection to learn
                which recommendations actually prevent churn in production.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page() -> None:
    st.subheader("About the project")

    st.markdown(
        """
        <div class="content-card">
            <h3>Why churn prediction matters to businesses</h3>
            <p>
                Customer churn is one of the most expensive silent failures in business. Revenue does not usually
                disappear all at once. It leaks away gradually as customers log in less often, spend less, complain
                more frequently, or disengage after a poor experience. By the time the loss becomes obvious, the
                relationship is often already damaged.
            </p>
            <p>
                A strong churn prediction system gives decision-makers earlier visibility into that decline. It helps
                teams prioritize outreach, reduce avoidable customer loss, target incentives more intelligently, and
                improve service recovery before dissatisfaction becomes permanent.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <h3>How this solution was built</h3>
            <p>
                The project was engineered as a full end-to-end machine learning system rather than a notebook-only
                prototype. It begins with structured data validation, schema checks, missing-value review, and leakage
                controls to ensure the model does not learn from personally identifying information such as customer
                IDs, names, security numbers, or referral identifiers.
            </p>
            <p>
                After validation, the pipeline performs production-safe preprocessing and feature engineering. That
                includes tenure features from joining dates, visit-time signals from last activity timestamps, cleaned
                login-frequency values, complaint indicators, dissatisfaction flags, loyalty-related wallet features,
                engagement segments, and spend segments. These engineered features are used consistently in both
                training and inference through the same saved pipeline objects.
            </p>
            <p>
                Multiple candidate models were then benchmarked on the same transformed data. Because the target is an
                ordered risk score rather than a simple yes-or-no churn label, the system automatically detected the
                task as ordinal multiclass classification and used business-relevant validation metrics such as weighted
                F1, macro F1, quadratic weighted kappa, ordinal MAE, one-vs-rest ROC-AUC, and PR-AUC to guide model
                selection. The final model was persisted alongside metadata, metrics, plots, and sample outputs so the
                deployed application can explain and reuse the exact trained artifact.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <h3>Key assumptions behind the model</h3>
            <p>
                The model assumes that historical customer behavior contains stable warning patterns that can be used to
                estimate future churn risk. It assumes the provided churn risk score is a valid ordered business target,
                and this implementation further assumes the raw `-1` tier can be safely merged into score `1` so the
                deployed system exposes a simpler 1-to-5 risk scale. It also assumes that engagement and complaint
                signals are informative proxies for loyalty health, that the dataset snapshot is representative enough
                for deployment-style scoring, and that the strongest interventions come from combining statistical model
                output with deterministic business rules rather than returning a bare number.
            </p>
            <p>
                A further assumption is that human-readable explanations matter. In practice, business users need to
                understand why a customer is risky, not just that they are risky. This is why the app translates model
                drivers into retention language such as low engagement, long time since last login, unresolved
                complaints, weak loyalty participation, and negative experience signals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <h3>How this project is useful to humanity</h3>
            <p>
                At a human level, churn prediction is really about reducing avoidable relationship breakdowns between
                people and the services they rely on. Better retention systems can lead to faster support recovery,
                better user experience, less irrelevant blanket marketing, and more targeted assistance for customers
                who are at risk of abandonment because of friction or dissatisfaction.
            </p>
            <p>
                For organizations, that translates into healthier revenue, smarter resource allocation, and more
                responsible decision support. For customers, it can mean better service, more relevant interventions,
                and fewer ignored complaints. This project solves the practical problem of identifying who needs help,
                when they need it, and what action is most likely to keep the relationship healthy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if AUTHOR_IMAGE_PATH.exists():
        author_image_b64 = encode_image(str(AUTHOR_IMAGE_PATH))
        author_image_html = f"""
        <div class="author-photo-wrap">
            <img class="author-photo" src="data:image/png;base64,{author_image_b64}" alt="Okon Prince" />
        </div>
        """
    else:
        author_image_html = "<div class='author-photo-wrap'><p>Author image unavailable.</p></div>"

    st.markdown(
        f"""
        <div class="author-card content-card">
            <div class="author-grid">
                <div>{author_image_html}</div>
                <div>
                    <h3>About the Author</h3>
                    <div class="author-name">Okon Prince</div>
                    <div class="author-role">AI Engineer &amp; Data Scientist | Senior Data Scientist at MIVA Open University</div>
                    <p>
                        I design and deploy end-to-end data systems that turn raw data into production-ready
                        intelligence.
                    </p>
                    <p>
                        My core stack includes Python, Streamlit, BigQuery, Supabase, Hugging Face, PySpark, SQL,
                        Machine Learning, LLMs, and Transformers.
                    </p>
                    <p>
                        My work spans risk scoring systems, A/B testing, traditional and AI-powered dashboards, RAG
                        pipelines, predictive analytics, LLM-based solutions, and AI research.
                    </p>
                    <p>
                        Currently, I work as a Senior Data Scientist in the department of Research and Development at
                        MIVA Open University, where I carry out AI / ML research and build intelligent systems that
                        drive analytics, decision support, and scalable AI innovation.
                    </p>
                    <p><em>I believe: models are trained, systems are engineered and impact is delivered.</em></p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    page, theme_mode = app_shell_state()
    inject_styles(theme_mode)
    inject_header_runtime_fix(theme_mode)
    render_app_shell(page, theme_mode)
    predictor = load_predictor()
    metrics_payload = load_metrics()

    if page == "Overview":
        render_overview_page(predictor, metrics_payload)
    elif page == "Predict":
        render_prediction_page(predictor, theme_mode)
    elif page == "Batch Scoring":
        render_batch_page(predictor)
    elif page == "Model Insights":
        render_insights_page(predictor, theme_mode)
    else:
        render_about_page()

    render_footer()


if __name__ == "__main__":
    main()
