"""
AUDIOVISUAL IDENTIFICATION CAPACITY TEST
Author: Elna Grasa, 2026, Barcelona
Description: Test to measure human's audiovisual identification capacity of AI content
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle, Arc
import csv

# 1. Basic configuration
st.set_page_config(page_title="AI Perception Test", layout="centered")

# --- Paths (absolute, based on file location to avoid CWD issues in Docker) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "csv")
IMAGE_LIST_FILE = os.path.join(DATA_DIR, "image_list.csv")
RESULTS_FILE = os.path.join(DATA_DIR, "results.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Global CSS with original image size and blocking "pull-to-refresh"
st.markdown("""
    <style>
    /* Prevent page refresh when scrolling down */
    html, body, [data-testid="stAppViewContainer"], .main {
        overscroll-behavior-y: none !important;
        overscroll-behavior: none !important;
        touch-action: manipulation !important;
    }
    /* Original margins and sizes */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        top: 0px !important;
    }
    header[data-testid="stHeader"] {
        height: 0px !important;
        background: transparent !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Read image bank from CSV
def load_image_bank():
    image_bank = []
    try:
        with open(IMAGE_LIST_FILE, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_bank.append({
                    "file": row["file"].strip(),
                    "solution": row["solution"].strip().upper()  # normalize to AI / REAL
                })
    except FileNotFoundError:
        st.error(f"File {IMAGE_LIST_FILE} not found")
        return []
    return image_bank

# Gauge aesthetic function in scale -10 to +10
def draw_aesthetic_gauge(accuracy):
    """
    Draws the semicircle graduated scale from -10 to +10.
    accuracy: float between 0 and 100 (absolute accuracy percentage)
    """
    random_performance = (accuracy - 50) * 2
    scale_value = random_performance / 10
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor='#FAFAFA')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.30, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    def percent_to_angle(p):
        return 180 - (p / 100) * 180

    # Colors: strong red → red → orange → yellow-green → medium green → strong green
    zones = [
        (0, 20,  '#C62828', "High\nConfusion"),
        (20, 40, '#E53935', "Moderate\nConfusion"),
        (40, 50, '#FB8C00', "Slightly\nWorse\nthan Random"),
        (50, 60, '#C0CA33', "Slightly\nBetter\nthan Random"),
        (60, 80, '#66BB6A', "Moderate\nAbility"),
        (80, 100,'#2E7D32', "High\nAbility")
    ]

    r_outer, r_inner = 1.0, 0.62
    for start, end, color, label in zones:
        wedge = Wedge(
            (0, 0), r_outer,
            percent_to_angle(end), percent_to_angle(start),
            width=r_outer - r_inner,
            facecolor=color, edgecolor='white', linewidth=2.0, alpha=0.95
        )
        ax.add_patch(wedge)
        mid_percent = (start + end) / 2
        mid_angle = np.deg2rad(percent_to_angle(mid_percent))
        if 40 <= start < 60:
            r_text = 0.92
        else:
            r_text = 0.81
        text_color = 'white' if start < 40 or start >= 80 else '#1A1A1A'
        ax.text(
            r_text * np.cos(mid_angle), r_text * np.sin(mid_angle), label,
            ha='center', va='center', fontsize=9.5,
            color=text_color,
            fontweight='bold', linespacing=1.1
        )

    ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=180, color='#333333', lw=2.5))
    ax.add_patch(Arc((0, 0), 2 * r_inner, 2 * r_inner, theta1=0, theta2=180, color='#333333', lw=1.5))

    scale_points = [
        (0, "-10"), (20, "-6"), (40, "-2"),
        (50, "0"), (60, "+2"), (80, "+6"), (100, "+10")
    ]

    for p, value in scale_points:
        angle = np.deg2rad(percent_to_angle(p))
        ax.plot(
            [(r_outer + 0.01) * np.cos(angle), (r_outer + 0.06) * np.cos(angle)],
            [(r_outer + 0.01) * np.sin(angle), (r_outer + 0.06) * np.sin(angle)],
            color='#333333', lw=2
        )
        ax.text(
            (r_outer + 0.15) * np.cos(angle),
            (r_outer + 0.15) * np.sin(angle),
            value, ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1D3557' if value == "0" else '#333333'
        )

    # Indicator arrow
    result_angle = np.deg2rad(percent_to_angle(accuracy))
    cos_angle, sin_angle = np.cos(result_angle), np.sin(result_angle)
    ax.plot(
        [0.08 * cos_angle, 1.18 * cos_angle],
        [0.08 * sin_angle, 1.18 * sin_angle],
        color='#111111', linestyle=':', linewidth=2.5, zorder=9
    )
    ax.annotate(
        '', xy=(0.58 * cos_angle, 0.58 * sin_angle),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle='->,head_length=0.7,head_width=0.45',
            color='#111111', lw=5.5
        ),
        zorder=10
    )
    ax.add_patch(Circle((0, 0), 0.05, facecolor='#111111', zorder=11))
    ax.text(
        0, 1.20, "DISCRIMINATION CAPACITY SCALE",
        ha='center', va='center', fontsize=13,
        fontweight='bold', color='#1D3557'
    )

    sign = "+" if scale_value > 0 else ""
    scale_value_str = f"{sign}{scale_value:.1f}".replace('.', ',')
    ax.text(
        0, -0.15,
        f"Scale Score: {scale_value_str} / 10",
        ha='center', va='center', fontsize=13, color='#111111', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EAEAEA', edgecolor='#CCCCCC', lw=1.2)
    )
    plt.tight_layout()
    return fig

# 2. Test image bank
IMAGE_BANK = load_image_bank()

def save_data(user, responses):
    """
    Appends all responses to the results CSV.
    Creates the file if it does not exist.
    """
    rows = []
    for r in responses:
        rows.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Participant": user["name"],
            "Age": user["age"],
            "IA_Exposure": user["ia_exposure"],   # internal key: never / occasionally / frequently / daily
            "Gender": user["gender"],             # internal key: male / female / prefer_not_to_say
            "Image": r["image"],
            "Response": r["response"],
            "Solution": r["solution"],
            "Correct": 1 if r["response"] == r["solution"] else 0
        })
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_FILE):
        df_old = pd.read_csv(RESULTS_FILE)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(RESULTS_FILE, index=False)

# Language selection for participants - set English as default
if "language" not in st.session_state:
    st.session_state.language = "en"

# 3. Initialize session variables
if "screen" not in st.session_state:
    st.session_state.screen = "registration"
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "index" not in st.session_state:
    st.session_state.index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []

# Language dictionary
author = "Elna Grasa"
texts = {
    "ca": {
        "title_research": "TREBALL DE RECERCA",
        "title_project": "TEST DE CAPACITAT D'IDENTIFICACIÓ AUDIOVISUAL",
        "name_surname": "Nom i Cognom:",
        "age": "Edat:",
        "ia_exposure": "Ús de xarxes socials / exposició a contingut audiovisual:",
        "gender": "Sexe:",
        "next_button": "Següent",
        "instructions_title": "INSTRUCCIONS IMPORTANTS",
        "instructions_text": "* Per a cada imatge o vídeo, tria si creus que ha estat generat per intel·ligència artificial o si és real.\n* **NO RESPONGUIS A L'ATZAR O INTENTIS ENDEVINAR:** Si creus que és REAL respon REAL, s'avalua a la IA no a tu.\n* Les teves respostes s'utilitzaran de manera anònima.",
        "warning_text": "Recorda: Un cop comencis, veuràs els continguts d'un en un sense la possibilitat de tornar enrere ni comparar-los.",
        "start_button": "Començar el test",
        "ia_button": "IA",
        "real_button": "REAL",
        "progress_text": "Imatge {index} de {total}",
        "thank_you": "Moltes gràcies!",
        "result_title": "El teu resultat global:",
        "absolute_correct": "Encert absolut",
        "absolute_percentage": "Percentatge absolut",
        "adjusted_random": "Resultat ajustat a l'atzar",
        "icia": "ICIA",
        "content_type": "Per tipus de contingut:",
        "ai_content": "Contingut IA",
        "real_content": "Contingut REAL",
        "comments": "Comentaris:",
        "high_ability": "Capacitat alta: Ets capaç de detectar majorment la IA i la realitat, en un grau molt elevat.",
        "moderate_ability": "Capacitat moderada: Ets capaç de detectar en un grau significatiu la IA i la realitat.",
        "slightly_better": "Lleugerament millor que l'atzar: Però no ets capaç de diferenciar la IA de la realitat significativament.",
        "slightly_worse": "Lleugerament pitjor que l'atzar: No ets capaç de diferenciar la IA de la realitat.",
        "moderate_confusion": "Confusió moderada: Confons IA per realitat o viceversa en un cert grau significatiu.",
        "high_confusion": "Confusió alta: Confons majorment IA per realitat o viceversa, en un grau molt elevat.",
        "new_participant": "Nou Participant 🔄",
        "never": "Mai",
        "occasionally": "Esporàdicament (1-2 dies per setmana)",
        "frequently": "Freqüentment (3-5 dies per setmana)",
        "daily": "Diàriament (6-7 dies per setmana)",
        "male": "Home",
        "female": "Dona",
        "prefer_not_to_say": "Prefereixo no dir-ho",
        "error_name": "Si us plau, escriu el teu nom.",
        "instructions_subtitle": "Si us plau, llegeix atentament abans de començar.",
        "error_file_not_found": "Fitxer no trobat: {path}",
        "error_summary": "Has pensat que el {ai_error}% de les imatges IA eren reals, i el {real_error}% de les imatges reals eren IA."
    },
    "en": {
        "title_research": "RESEARCH PROJECT",
        "title_project": "AUDIOVISUAL IDENTIFICATION CAPACITY TEST",
        "name_surname": "Full Name:",
        "age": "Age:",
        "ia_exposure": "Social media usage / exposure to audiovisual content:",
        "gender": "Gender:",
        "next_button": "Next",
        "instructions_title": "IMPORTANT INSTRUCTIONS",
        "instructions_text": "* For each image or video, choose whether you think it was generated by artificial intelligence or if it is real.\n* **DO NOT ANSWER RANDOMLY OR GUESS:** If you think it's REAL, answer REAL. The evaluation is about the AI, not about you.\n* Your responses will be used anonymously.",
        "warning_text": "Remember: Once you start, you'll see the items one by one without the possibility to go back or compare them.",
        "start_button": "Start the test",
        "ia_button": "AI",
        "real_button": "REAL",
        "progress_text": "Image {index} of {total}",
        "thank_you": "Thank you very much!",
        "result_title": "Your overall result:",
        "absolute_correct": "Absolute correctness",
        "absolute_percentage": "Absolute percentage",
        "adjusted_random": "Random-adjusted result",
        "icia": "AICI",
        "content_type": "By content type:",
        "ai_content": "AI Content",
        "real_content": "REAL Content",
        "comments": "Comments:",
        "high_ability": "High ability: You are able to distinguish between AI and reality to a very high degree.",
        "moderate_ability": "Moderate ability: You are able to distinguish between AI and reality to a significant degree.",
        "slightly_better": "Slightly better than random: You are not able to significantly distinguish between AI and reality.",
        "slightly_worse": "Slightly worse than random: You are not able to distinguish AI from reality.",
        "moderate_confusion": "Moderate confusion: You confuse AI with reality or vice versa to a significant extent.",
        "high_confusion": "High confusion: You mainly confuse AI with reality or vice versa, to a very high extent.",
        "new_participant": "New Participant 🔄",
        "never": "Never",
        "occasionally": "Occasionally (1-2 days per week)",
        "frequently": "Frequently (3-5 days per week)",
        "daily": "Daily (6-7 days per week)",
        "male": "Male",
        "female": "Female",
        "prefer_not_to_say": "Prefer not to say",
        "error_name": "Please enter your name.",
        "instructions_subtitle": "If you please, read carefully before starting.",
        "error_file_not_found": "File not found: {path}",
        "error_summary": "You thought that {ai_error}% of the AI images were real, and {real_error}% of the real images were AI."
    }
}

# Get current language text
current_texts = texts[st.session_state.language]

# --- SCREEN 1: REGISTRATION ---
if st.session_state.screen == "registration":
    st.markdown(f"<p style='text-align: center;'>{current_texts['title_research']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 24px;'>{current_texts['title_project']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{author}</p>", unsafe_allow_html=True)

    # Language selector at top
    lang_options = {"en": "English", "ca": "Català"}
    selected_lang = st.selectbox("Language:", list(lang_options.keys()), format_func=lambda x: lang_options[x])
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        current_texts = texts[st.session_state.language]
        st.rerun()

    name = st.text_input(current_texts["name_surname"])
    age = st.number_input(current_texts["age"], min_value=1, max_value=110, value=17)

    # IA exposure: store internal key ("never"/"occasionally"/...) but show translated label
    ia_exposure_options = ["never", "occasionally", "frequently", "daily"]
    ia_exposure = st.selectbox(
        current_texts["ia_exposure"],
        ia_exposure_options,
        format_func=lambda k: current_texts[k]
    )

    # Gender: store internal key ("male"/"female"/"prefer_not_to_say") but show translated label
    gender_options = ["male", "female", "prefer_not_to_say"]
    gender = st.selectbox(
        current_texts["gender"],
        gender_options,
        format_func=lambda k: current_texts[k]
    )

    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #239b56 !important;
            color: white !important;
            border: 1px solid #1e8449;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button(current_texts["next_button"], use_container_width=True):
        if name.strip() != "":
            st.session_state.user_data = {
                "name": name,
                "age": age,
                "ia_exposure": ia_exposure,   # internal key
                "gender": gender              # internal key
            }
            st.session_state.screen = "instructions"
            st.rerun()
        else:
            st.error(current_texts["error_name"])

# --- INTERMEDIATE SCREEN: INSTRUCTIONS ---
elif st.session_state.screen == "instructions":
    st.title(current_texts["instructions_title"])
    st.subheader(current_texts["instructions_subtitle"])

    st.markdown(current_texts["instructions_text"])

    st.warning(current_texts["warning_text"])

    st.markdown("---")

    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #239b56 !important;
            color: white !important;
            border: 1px solid #1e8449;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button(current_texts["start_button"], use_container_width=True, type="primary"):
        st.session_state.screen = "test"
        st.rerun()

# --- SCREEN 2: TEST ---
elif st.session_state.screen == "test":
    idx = st.session_state.index
    total = len(IMAGE_BANK)

    if total == 0:
        st.error("No images loaded. Check images/image_list.csv")
    else:
        image_info = IMAGE_BANK[idx]
        image_path = os.path.join(IMAGES_DIR, image_info["file"])

        if os.path.exists(image_path):
            extension = image_info["file"].lower()
            if extension.endswith((".mp4", ".mov")):
                st.video(image_path, loop=False, autoplay=True, muted=False)
            else:
                st.image(image_path, use_container_width=True)
        else:
            st.error(current_texts["error_file_not_found"].format(path=image_path))

        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                gap: 8px !important;
            }
            div[data-testid="stHorizontalBlock"] > div {
                width: 50% !important;
                min-width: 50% !important;
            }
            div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                background-color: #E57373 !important;
                color: white !important;
                border: 1px solid #EF5350 !important;
            }
            div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
                background-color: #58D68D !important;
                color: white !important;
                border: 1px solid #2ECC71 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        vote = None
        with col1:
            if st.button(current_texts["ia_button"], use_container_width=True, type="primary"):
                vote = "AI"

        with col2:
            if st.button(current_texts["real_button"], use_container_width=True, type="secondary"):
                vote = "REAL"

        st.markdown("---")
        st.progress((idx + 1) / total)
        st.write(current_texts["progress_text"].format(index=idx + 1, total=total))

        if vote:
            st.session_state.responses.append({
                "image": image_info["file"],
                "response": vote,
                "solution": image_info["solution"]
            })
            if idx + 1 < total:
                st.session_state.index += 1
            else:
                save_data(st.session_state.user_data, st.session_state.responses)
                st.session_state.screen = "final"
            st.rerun()

# --- SCREEN 3: FINAL ---
elif st.session_state.screen == "final":
    st.title(current_texts["thank_you"])
    st.subheader(current_texts["result_title"])

    total_questions = len(st.session_state.responses)
    correct_count = sum(1 for r in st.session_state.responses if r["response"] == r["solution"])

    if total_questions > 0:
        correct_percentage = (correct_count / total_questions) * 100
    else:
        correct_percentage = 0.0

    # Calculations of derived metrics
    random_performance = (correct_percentage - 50) * 2
    scale_score = random_performance / 10

    sign_random = "+" if random_performance > 0 else ""
    sign_icia = "+" if scale_score > 0 else ""
    random_value_str = f"{sign_random}{random_performance:.0f}%"
    icia_value_str = f"{sign_icia}{scale_score:.1f}".replace('.', ',')

    # CSS for forcing 2x2 layout also on mobile
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 0.5rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            width: 50% !important;
            min-width: 50% !important;
            flex: 1 1 50% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Row 1: Absolute correct and Absolute percentage
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.metric(label=current_texts["absolute_correct"], value=f"{correct_count} / {total_questions}")
    with row1_col2:
        st.metric(label=current_texts["absolute_percentage"], value=f"{int(round(correct_percentage))}%")

    # Row 2: Result adjusted to random and ICIA
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.metric(label=current_texts["adjusted_random"], value=random_value_str)
    with row2_col2:
        st.metric(label=current_texts["icia"], value=icia_value_str)

    st.markdown("---")

    # Render gauge
    st.pyplot(draw_aesthetic_gauge(correct_percentage))

    st.markdown("---")

    # Breakdown by content type
    st.subheader(current_texts["content_type"])

    ai_responses = [r for r in st.session_state.responses if r["solution"] == "AI"]
    real_responses = [r for r in st.session_state.responses if r["solution"] == "REAL"]

    ai_correct = sum(1 for r in ai_responses if r["response"] == r["solution"])
    ai_total = len(ai_responses)
    ai_percentage = (ai_correct / ai_total * 100) if ai_total > 0 else 0

    real_correct = sum(1 for r in real_responses if r["response"] == r["solution"])
    real_total = len(real_responses)
    real_percentage = (real_correct / real_total * 100) if real_total > 0 else 0

    # Error percentages:
    # ai_error = % of AI images that were answered as REAL
    # real_error = % of REAL images that were answered as AI
    ai_error = 100 - ai_percentage if ai_total > 0 else 0
    real_error = 100 - real_percentage if real_total > 0 else 0

    col_ai, col_real = st.columns(2)
    with col_ai:
        st.markdown(f"**{current_texts['ai_content']}**")
        st.write(f"{current_texts['absolute_correct']}: {ai_correct} of {ai_total} ({int(round(ai_percentage))}%)")
        st.progress(ai_percentage / 100)
    with col_real:
        st.markdown(f"**{current_texts['real_content']}**")
        st.write(f"{current_texts['absolute_correct']}: {real_correct} of {real_total} ({int(round(real_percentage))}%)")
        st.progress(real_percentage / 100)

    st.markdown("---")

    st.subheader(current_texts["comments"])

    if correct_percentage >= 80:
        st.success(current_texts["high_ability"])
    elif correct_percentage >= 60:
        st.success(current_texts["moderate_ability"])
    elif correct_percentage >= 50:
        st.success(current_texts["slightly_better"])
    elif correct_percentage >= 40:
        st.success(current_texts["slightly_worse"])
    elif correct_percentage >= 20:
        st.info(current_texts["moderate_confusion"])
    else:
        st.warning(current_texts["high_confusion"])

    st.info(
        current_texts["error_summary"].format(
            ai_error=int(round(ai_error)),
            real_error=int(round(real_error))
        )
    )

    st.markdown("---")
    if st.button(current_texts["new_participant"], use_container_width=True):
        st.session_state.screen = "registration"
        st.session_state.index = 0
        st.session_state.responses = []
        st.rerun()