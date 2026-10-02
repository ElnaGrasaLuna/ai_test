"""
AUDIOVISUAL IDENTIFICATION CAPACITY TEST — CONTROL DASHBOARD
Author: Elna Grasa, 2026, Barcelona
Description: Control dashboard for analysing results of the AI perception test.

Access model:
  - Default state: only the "Administrator" option is available. A password
    is required to see anything.
  - When an administrator enables "public mode", the user selector shows two
    options: "Administrator" (password) and "Public" (aggregated data only).
  - The administrator can turn public mode off at any time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Arc
import os
import re
import hashlib
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Control Dashboard - AI Research", layout="wide")

# --- PATHS (absolute, based on file location to avoid CWD issues in Docker) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "csv")
IMAGE_LIST_FILE = os.path.join(DATA_DIR, "image_list.csv")
RESULTS_FILE = os.path.join(DATA_DIR, "results.csv")

# Secret salt for persistent pseudonymisation (used in public mode)
SALT_SECRET = "RecercaIA_admin_2026_Secret"

# Admin password
PASSWORD_ADMIN = "admin2026"

# Default end date for the test (editable by administrator)
DEFAULT_END_DATE = datetime(2028, 1, 1, 0, 0, 0)

# --- IMAGE THEME MAPPING (by number / id, extension-independent) ---
THEMES_BY_ID = {
    "Famous people": ["1", "23", "30", "31", "32", "34"],
    "Wars, conflicts and vulnerable situations": ["2b", "2", "3", "5", "8", "9b", "9", "10", "11", "12f", "12"],
    "People, influencers and content likely to spread on social media": [
        "4b", "4", "15", "18", "19", "20", "17b", "17", "25", "26", "29", "36b", "36", "37", "38b", "38"
    ],
    "Nature and natural disasters": ["35", "33", "13b", "13", "7", "6"],
    "Vintage photographs and content from the past": ["21", "22", "27", "28c", "28"],
    "Animals and nature": ["14", "16", "24"],
}


# ============================================================
# --- LANGUAGE DICTIONARY ---
# ============================================================
TEXTS = {
    "en": {
        # Sidebar
        "config_header": "CONFIGURATION",
        "active_user": "ACTIVE USER:",
        "user_admin": "Administrator",
        "user_public": "Public",
        "password_label": "Access password for Administrator:",
        "password_ok": "Access granted as Administrator.",
        "password_ko": "Incorrect password.",
        "role_admin": "🔒 Role: **Administrator** (full access)",
        "role_public": "👁 Role: **Public** (aggregated data only)",
        "public_mode_toggle": "🌐 Enable public mode",
        "public_mode_toggle_help": "If enabled, anonymous visitors can select the 'Public' user and view aggregated statistics without a password. Disable it to restrict access to administrators only.",
        "public_mode_active": "🌐 **Public mode is active.** Anyone can access the aggregated view without a password.",
        "logout_button": "Log out",
        "no_access_message": "🔒 This dashboard is restricted. Only administrators can access it. Please contact the researcher if you need access.",

        # Test control
        "test_control_header": "TEST CONTROL",
        "test_start": "**TEST START:**",
        "test_end": "**TEST END:**",
        "test_start_date": "Test start date",
        "test_start_time": "Test start time",
        "test_end_date": "Test end date",
        "test_end_time": "Test end time",
        "test_start_caption": "🔒 *The start date is automatically taken from the first record in the CSV and cannot be modified.*",
        "test_end_caption_admin": "🔓 *You can edit the end date of the test. Results will be filtered up to this date.*",
        "test_end_caption_public": "🔒 *The end date is fixed and cannot be modified to preserve the validity of results.*",
        "date_range_invalid": "⚠️ The start date must be earlier than the end date.",
        "refresh_button": "REFRESH DATA",
        "shutdown_button": "SHUT DOWN",
        "shutdown_warning": "Shutting down the Streamlit server...",
        "language_label": "Language:",

        # Main title
        "main_title": "CONTROL PANEL | AI RESEARCH PROJECT",
        "main_title_public": "AI RESEARCH PROJECT | RESULTS",
        "public_mode_notice": "👁 **Public mode.** You are viewing aggregated statistics only.",

        # Tabs
        "tab_global": "Global Metrics",
        "tab_charts": "Global Charts",
        "tab_participant": "Participant Inspector",
        "tab_content": "Content Analysis",
        "tab_themes": "Theme Analysis",
        "tab_images": "Analysed Images",
        "tab_data": "Raw Data",

        # Global tab
        "global_subtitle": "Overall Research Summary",
        "metric_total_participants": "Total Participants",
        "metric_mean_accuracy": "Mean Global Accuracy",
        "metric_random_adjusted": "Random-adjusted",
        "metric_icia_global": "Mean Global AICI",
        "scale_title": "Global Mean Capacity Scale",
        "age_group_perf_title": "Performance by Age Group",
        "col_age_group": "Age Group",
        "col_mean_accuracy": "Mean Accuracy (%)",
        "col_participants": "Participants",
        "extra_demographics": "Additional Demographic Analysis",
        "gender_perf_title": "Performance by Gender",
        "col_gender": "Gender",
        "exposure_perf_title": "Performance by AI Exposure",
        "col_exposure": "Exposure Level",

        # Content breakdown
        "by_content_type": "By content type:",
        "ai_content": "AI Content",
        "real_content": "REAL Content",
        "correct_label": "Correct",
        "diagnostic_comments": "Diagnostic comments:",
        "ability_high_plural": "High ability: They are able to distinguish between AI and reality to a very high degree.",
        "ability_moderate_plural": "Moderate ability: They are able to distinguish between AI and reality to a significant degree.",
        "better_random_plural": "Slightly better than random: However, they are not able to significantly distinguish AI from reality.",
        "worse_random_plural": "Slightly worse than random: They are not able to distinguish AI from reality.",
        "confusion_moderate_plural": "Moderate confusion: They confuse AI with reality or vice versa to a significant extent.",
        "confusion_high_plural": "High confusion: They mainly confuse AI with reality or vice versa, to a very high extent.",
        "ability_high_single": "High ability: You are able to distinguish between AI and reality to a very high degree.",
        "ability_moderate_single": "Moderate ability: You are able to distinguish between AI and reality to a significant degree.",
        "better_random_single": "Slightly better than random: However, you are not able to significantly distinguish AI from reality.",
        "worse_random_single": "Slightly worse than random: You are not able to distinguish AI from reality.",
        "confusion_moderate_single": "Moderate confusion: You confuse AI with reality or vice versa to a significant extent.",
        "confusion_high_single": "High confusion: You mainly confuse AI with reality or vice versa, to a very high extent.",
        "error_summary_plural": "They thought that {ai_error}% of the AI images were real, and {real_error}% of the real images were AI.",
        "error_summary_single": "You thought that {ai_error}% of the AI images were real, and {real_error}% of the real images were AI.",

        # Participant tab
        "participant_subtitle": "Individual Data Lookup",
        "select_participant": "Select a participant:",
        "p_age": "**Age:**",
        "p_gender": "**Gender:**",
        "p_exposure": "**AI Exposure:**",
        "p_test_date": "**Test date:**",
        "p_accuracy": "Accuracy Percentage",
        "p_icia": "Individual AICI",
        "response_detail": "Response details:",
        "col_image": "Image",
        "col_solution": "Solution",
        "col_response": "Response",
        "col_correct": "Correct",
        "correct_yes": "✅ Correct",
        "correct_no": "❌ Wrong",
        "ranking_title": "General Participant Ranking by Accuracy",
        "ranking_desc": "Below is the full list of participants sorted from **highest to lowest accuracy**:",
        "col_rank": "Rank",
        "col_accuracy_pct": "Accuracy (%)",
        "col_test_date_short": "Test date",
        "top_bottom_title": "Top 25 vs Bottom 25 Participants",
        "top25": "Top 25",
        "bottom25": "Bottom 25",
        "col_name": "Name",
        "col_accuracy_short": "% Accuracy",
        "col_exposure_short": "Exposure",
        "evolution_title": "Participation Evolution over Time",
        "evolution_desc": "Cumulative growth of participants according to the exact date and time they took the test:",

        # Content tab
        "content_subtitle": "Error and Difficulty Analysis per Content",
        "img_acc_title": "Individual Performance per Image / File (1 to 38)",
        "error_rate_title": "Error Rate: AI vs REAL Images (with Mean Error Dotted Line)",
        "detailed_table_title": "Detailed Difficulty Table per Audiovisual File",
        "ordered_by_confusion": "Ordered by most confusing content:",

        # Themes tab
        "themes_subtitle": "Error analysis by image theme",
        "themes_desc": "This tab shows **which image theme generated the most confusion** (most detection errors) and which the least.",
        "col_theme": "Theme",
        "col_pct_error": "% Error",
        "col_pct_correct": "% Correct",
        "theme_most_error": "**Theme with MOST errors (most manipulated/confusing):** {theme} → {pct:.1f}% error",
        "theme_least_error": "**Theme with LEAST errors (best detected):** {theme} → {pct:.1f}% error",

        # Images tab
        "images_subtitle": "Detailed File of Analysed Image / Video",
        "select_image": "Select an image or video to analyse:",
        "solution_real": "True Solution",
        "total_responses": "Total Responses",
        "accuracy_pct": "Accuracy Percentage",
        "confusion_rate": "Confusion Rate",
        "what_people_thought": "### What did people think?",
        "thought_ai": "**Thought it was AI:** `{n}` people (**{pct:.1f}%**)",
        "thought_real": "**Thought it was REAL:** `{n}` people (**{pct:.1f}%**)",
        "video_file": "Video file:",
        "image_file": "Image file:",
        "file_label": "File:",
        "filename_hint": "**File name:** `{name}`\n\n*(Save the file to the `./images/` folder to see the preview on screen)*",
        "demographic_breakdown": "### Demographic breakdown for this file",
        "accuracy_by_age": "#### Accuracy by Age Group:",
        "accuracy_by_gender": "#### Accuracy by Gender:",
        "accuracy_by_exposure": "#### Accuracy by Exposure:",
        "col_responses": "Responses",

        # Raw data tab
        "raw_subtitle": "Full Data Table",
        "download_csv": "Download CSV",
        "export_filename": "full_results_exported.csv",

        # Warnings
        "no_data": "No data could be loaded or the CSV file is empty.",
        "file_not_found": "Local file not found: {path}",
        "file_read_error": "Error reading local file {path}: {error}",

        # Chart labels
        "chart_date_time": "Participation Date and Time",
        "chart_cumulative_participants": "Number of Cumulative Participants",
        "chart_evolution_title": "Participant Evolution over Time",
        "chart_mean_accuracy_sd": "Mean Accuracy Percentage (± SD)",
        "chart_trend_line": "Trend Line",
        "chart_random_50": "Random (50%)",
        "chart_age_groups": "Age Groups",
        "chart_perf_age_groups": "Performance by Age Group (with SD and Trend)",
        "chart_age_intervals_5": "Age Intervals (5 by 5 years)",
        "chart_perf_age_5": "Mean Performance by 5-Year Age Intervals",
        "chart_top25_age_dist": "Top 25 Participants: Age Distribution",
        "chart_bottom25_age_dist": "Bottom 25 Participants: Age Distribution",
        "chart_age_group_label": "Age Group",
        "chart_n_participants": "Number of Participants",
        "chart_exposure_level": "Exposure Level",
        "chart_top25_exposure": "Top 25 Participants: Exposure Distribution",
        "chart_bottom25_exposure": "Bottom 25 Participants: Exposure Distribution",
        "chart_gender": "Gender",
        "chart_top25_gender": "Top 25 Participants: Gender Distribution",
        "chart_bottom25_gender": "Bottom 25 Participants: Gender Distribution",
        "chart_perf_gender": "Performance by Gender",
        "chart_exposure_more_less": "Exposure Level (More to Less)",
        "chart_perf_exposure": "Performance by AI Exposure (± SD)",
        "chart_participants_per_exposure": "Participants per AI Exposure Level",
        "chart_exposure_by_gender": "AI Exposure Normalised by Gender (%)",
        "chart_exposure_by_age": "AI Exposure Normalised by Age Group (%)",
        "chart_pct_of_gender": "Percentage of each gender's 100% (%)",
        "chart_pct_of_age": "Percentage of each age group's 100% (%)",
        "chart_accuracy_per_image": "Accuracy Percentage per Image / File (1 to 38)",
        "chart_error_ai": "Error Rate: AI Images (Most to Least Confusion)",
        "chart_error_real": "Error Rate: REAL Images (Most to Least Confusion)",
        "chart_mean_error_ai": "Mean AI error ({v:.1f}%)",
        "chart_mean_error_real": "Mean REAL error ({v:.1f}%)",
        "chart_mean_error_global": "Mean global error ({v:.1f}%)",
        "chart_continuous_age": "Continuous Age Distribution (6-80 years) with 10-Year Moving Average",
        "chart_age_years": "Age (years)",
        "chart_moving_avg_10": "Moving Average (10-year window)",
        "chart_age_mean_sd": "Mean accuracy per age (± SD)",
        "chart_participants_scatter": "Participants",
        "chart_age_bins_5": "Performance by Age Intervals of 5 Years",
        "chart_not_enough_ages": "Not enough data for ages between 6 and 80",
        "chart_not_enough_ages_5_90": "Not enough data for ages between 5 and 90",
        "chart_scale_score": "Scale Score:",
    },
    "ca": {
        # Sidebar
        "config_header": "CONFIGURACIÓ",
        "active_user": "USUARI ACTIU:",
        "user_admin": "Administrador",
        "user_public": "Públic",
        "password_label": "Contrasenya d'accés per a Administrador:",
        "password_ok": "Accés concedit com a Administrador.",
        "password_ko": "Contrasenya incorrecta.",
        "role_admin": "🔒 Rol: **Administrador** (accés complet)",
        "role_public": "👁 Rol: **Públic** (només dades agregades)",
        "public_mode_toggle": "🌐 Activar mode públic",
        "public_mode_toggle_help": "Si s'activa, els visitants anònims podran seleccionar l'usuari 'Públic' i veure estadístiques agregades sense contrasenya. Desactiva'l per restringir l'accés només a administradors.",
        "public_mode_active": "🌐 **El mode públic està actiu.** Qualsevol pot accedir a la vista agregada sense contrasenya.",
        "logout_button": "Tancar sessió",
        "no_access_message": "🔒 Aquest panell està restringit. Només els administradors hi poden accedir. Contacta amb la investigadora si necessites accés.",

        # Test control
        "test_control_header": "CONTROL DEL TEST",
        "test_start": "**INICI DEL TEST:**",
        "test_end": "**FI DEL TEST:**",
        "test_start_date": "Data d'inici del test",
        "test_start_time": "Hora d'inici del test",
        "test_end_date": "Data final del test",
        "test_end_time": "Hora final del test",
        "test_start_caption": "🔒 *La data d'inici s'obté automàticament del primer registre del CSV i no es pot modificar.*",
        "test_end_caption_admin": "🔓 *Pots editar la data final del test. Els resultats es filtraran fins aquesta data.*",
        "test_end_caption_public": "🔒 *La data final està fixada i no permet modificacions per mantenir la validesa dels resultats.*",
        "date_range_invalid": "⚠️ La data d'inici ha de ser anterior a la data final.",
        "refresh_button": "ACTUALITZA DADES",
        "shutdown_button": "APAGAR",
        "shutdown_warning": "Tancant el servidor de Streamlit...",
        "language_label": "Idioma:",

        # Main title
        "main_title": "PANELL DE CONTROL | TREBALL DE RECERCA IA",
        "main_title_public": "TREBALL DE RECERCA IA | RESULTATS",
        "public_mode_notice": "👁 **Mode públic.** Estàs veient només estadístiques agregades.",

        # Tabs
        "tab_global": "Mètriques Globals",
        "tab_charts": "Gràfiques Globals",
        "tab_participant": "Inspector de Participant",
        "tab_content": "Anàlisi per Contingut",
        "tab_themes": "Anàlisi per Temes",
        "tab_images": "Imatges Analitzades",
        "tab_data": "Dades Brutes",

        # Global tab
        "global_subtitle": "Resum General de la Recerca",
        "metric_total_participants": "Participants Totals",
        "metric_mean_accuracy": "Encert Mitjà Global",
        "metric_random_adjusted": "Ajustat a l'Atzar",
        "metric_icia_global": "ICIA Mitjà Global",
        "scale_title": "Escala de Capacitat Mitjana Global",
        "age_group_perf_title": "Rendiment per Grups d'Edat",
        "col_age_group": "Grup d'Edat",
        "col_mean_accuracy": "Encert Mitjà (%)",
        "col_participants": "Participants",
        "extra_demographics": "Anàlisi Demogràfica Addicional",
        "gender_perf_title": "Rendiment per Gènere / Sexe",
        "col_gender": "Gènere / Sexe",
        "exposure_perf_title": "Rendiment segons Exposició a la IA",
        "col_exposure": "Nivell d'Exposició",

        # Content breakdown
        "by_content_type": "Per tipus de contingut:",
        "ai_content": "Contingut IA",
        "real_content": "Contingut REAL",
        "correct_label": "Encerts",
        "diagnostic_comments": "Comentaris diagnòstics:",
        "ability_high_plural": "Capacitat alta: Són capaços de detectar majorment la IA i la realitat, en un grau molt elevat.",
        "ability_moderate_plural": "Capacitat moderada: Són capaços de detectar en un grau significatiu la IA i la realitat.",
        "better_random_plural": "Lleugerament millor que l'atzar: Però no són capaços de diferenciar la IA de la realitat significativament.",
        "worse_random_plural": "Lleugerament pitjor que l'atzar: No són capaços de diferenciar la IA de la realitat.",
        "confusion_moderate_plural": "Confusió moderada: Confonen IA per realitat o viceversa en un cert grau significatiu.",
        "confusion_high_plural": "Confusió alta: Confonen majorment IA per realitat o viceversa, en un grau molt elevat.",
        "ability_high_single": "Capacitat alta: És capaç de detectar majorment la IA i la realitat, en un grau molt elevat.",
        "ability_moderate_single": "Capacitat moderada: És capaç de detectar en un grau significatiu la IA i la realitat.",
        "better_random_single": "Lleugerament millor que l'atzar: Però no és capaç de diferenciar la IA de la realitat significativament.",
        "worse_random_single": "Lleugerament pitjor que l'atzar: No és capaç de diferenciar la IA de la realitat.",
        "confusion_moderate_single": "Confusió moderada: Confon IA per realitat o viceversa en un cert grau significatiu.",
        "confusion_high_single": "Confusió alta: Confon majorment IA per realitat o viceversa, en un grau molt elevat.",
        "error_summary_plural": "Han cregut que el {ai_error}% de les imatges IA eren reals, i que el {real_error}% de les imatges reals eren IA.",
        "error_summary_single": "Ha cregut que el {ai_error}% de les imatges IA eren reals, i que el {real_error}% de les imatges reals eren IA.",

        # Participant tab
        "participant_subtitle": "Consulta de Dades Individuals",
        "select_participant": "Selecciona un participant:",
        "p_age": "**Edat:**",
        "p_gender": "**Sexe:**",
        "p_exposure": "**Exposició IA:**",
        "p_test_date": "**Data test:**",
        "p_accuracy": "Percentatge Encert",
        "p_icia": "ICIA Individual",
        "response_detail": "Detall de les respostes:",
        "col_image": "Imatge",
        "col_solution": "Solució",
        "col_response": "Resposta",
        "col_correct": "Encert",
        "correct_yes": "✅ Encert",
        "correct_no": "❌ Error",
        "ranking_title": "Rànquing General de Participants per Encert",
        "ranking_desc": "A continuació es mostra el llistat complet de tots els participants ordenats de **major a menor encert**:",
        "col_rank": "Rànquing",
        "col_accuracy_pct": "Encert (%)",
        "col_test_date_short": "Data de Participació",
        "top_bottom_title": "Millors 25 vs Pitjors 25 Participants",
        "top25": "Millors 25",
        "bottom25": "Pitjors 25",
        "col_name": "Nom",
        "col_accuracy_short": "% Encert",
        "col_exposure_short": "Exposició",
        "evolution_title": "Evolució de Participació en el Temps",
        "evolution_desc": "Gràfica de creixement acumulat de participants segons la data i hora exacta en què van realitzar el test:",

        # Content tab
        "content_subtitle": "Anàlisi d'Error i Dificultat per Contingut",
        "img_acc_title": "Rendiment Individual per Imatges / Fitxers (1 a 38)",
        "error_rate_title": "Taxa d'Error: Imatges IA vs REALS (amb Línia de Punts d'Error Mitjà)",
        "detailed_table_title": "Taula Detallada de Dificultat per Fitxer Audiovisual",
        "ordered_by_confusion": "Ordenat per contingut que MÉS confongui als participants:",

        # Themes tab
        "themes_subtitle": "Anàlisi d'errors per tema d'imatge",
        "themes_desc": "Aquesta pestanya mostra **quin tema d'imatges ha generat més confusió** (més errors de detecció) i quin menys.",
        "col_theme": "Tema",
        "col_pct_error": "% Error",
        "col_pct_correct": "% Encert",
        "theme_most_error": "**Tema amb MÉS error (més manipulat/confós):** {theme} → {pct:.1f}% d'error",
        "theme_least_error": "**Tema amb MENYS error (millor detectat):** {theme} → {pct:.1f}% d'error",

        # Images tab
        "images_subtitle": "Fitxa Detallada d'Imatge / Vídeo Analitzat",
        "select_image": "Selecciona una imatge o vídeo a analitzar:",
        "solution_real": "Solució Real",
        "total_responses": "Total Respostes",
        "accuracy_pct": "Percentatge d'Encert",
        "confusion_rate": "Taxa de Confusió",
        "what_people_thought": "### Què ha pensat la gent?",
        "thought_ai": "**Han pensat que era IA:** `{n}` persones (**{pct:.1f}%**)",
        "thought_real": "**Han pensat que era REAL:** `{n}` persones (**{pct:.1f}%**)",
        "video_file": "Fitxer de vídeo:",
        "image_file": "Fitxer d'imatge:",
        "file_label": "Fitxer:",
        "filename_hint": "**Nom del fitxer:** `{name}`\n\n*(Desa el fitxer a la carpeta `./images/` si vols veure la vista prèvia en pantalla)*",
        "demographic_breakdown": "### Desglossament Demogràfic per aquest fitxer",
        "accuracy_by_age": "#### Encert per Grup d'Edat:",
        "accuracy_by_gender": "#### Encert per Sexe:",
        "accuracy_by_exposure": "#### Encert per Exposició:",
        "col_responses": "Respostes",

        # Raw data tab
        "raw_subtitle": "Taula de Dades Completa",
        "download_csv": "Descarregar CSV",
        "export_filename": "resultats_complets_exportats.csv",

        # Warnings
        "no_data": "No s'han pogut carregar les dades o el fitxer CSV està buit.",
        "file_not_found": "No s'ha trobat el fitxer local: {path}",
        "file_read_error": "Error en llegir el fitxer local {path}: {error}",

        # Chart labels
        "chart_date_time": "Data i Hora de Participació",
        "chart_cumulative_participants": "Nombre de Participants Acumulats",
        "chart_evolution_title": "Evolució del Nombre de Participants en el Temps",
        "chart_mean_accuracy_sd": "Percentatge d'Encert Mitjà (± SD)",
        "chart_trend_line": "Línia de Tendència",
        "chart_random_50": "Atzar (50%)",
        "chart_age_groups": "Grups d'Edat",
        "chart_perf_age_groups": "Rendiment per Grups d'Edat (amb SD i Tendència)",
        "chart_age_intervals_5": "Intervals d'Edat (5 en 5 anys)",
        "chart_perf_age_5": "Rendiment Mitjà per Intervals d'Edat de 5 en 5 Anys",
        "chart_top25_age_dist": "Millors 25 Participants: Distribució per Edats",
        "chart_bottom25_age_dist": "Pitjors 25 Participants: Distribució per Edats",
        "chart_age_group_label": "Grup d'Edat",
        "chart_n_participants": "Nombre de Participants",
        "chart_exposure_level": "Nivell d'Exposició",
        "chart_top25_exposure": "Millors 25 Participants: Distribució per Exposició",
        "chart_bottom25_exposure": "Pitjors 25 Participants: Distribució per Exposició",
        "chart_gender": "Sexe",
        "chart_top25_gender": "Millors 25 Participants: Distribució per Sexe",
        "chart_bottom25_gender": "Pitjors 25 Participants: Distribució per Sexe",
        "chart_perf_gender": "Rendiment per sexe",
        "chart_exposure_more_less": "Nivell d'Exposició (Més a Menys)",
        "chart_perf_exposure": "Rendiment per Exposició a la IA (± SD)",
        "chart_participants_per_exposure": "Participants per Nivell d'Exposició a la IA",
        "chart_exposure_by_gender": "Exposició a la IA Normalitzada per Sexe (%)",
        "chart_exposure_by_age": "Exposició a la IA Normalitzada per Grups d'Edat (%)",
        "chart_pct_of_gender": "Percentatge respecte al 100% de cada sexe (%)",
        "chart_pct_of_age": "Percentatge respecte al 100% de cada grup d'edat (%)",
        "chart_accuracy_per_image": "Percentatge d'Encert per Imatge / Fitxer (1 a 38)",
        "chart_error_ai": "Taxa d'Error: Imatges IA (Més a Menys Confusió)",
        "chart_error_real": "Taxa d'Error: Imatges REALS (Més a Menys Confusió)",
        "chart_mean_error_ai": "Error Mitjà IA ({v:.1f}%)",
        "chart_mean_error_real": "Error Mitjà REAL ({v:.1f}%)",
        "chart_mean_error_global": "Error Mitjà Global ({v:.1f}%)",
        "chart_continuous_age": "Distribució Contínua per Edat (6-80 anys) amb Mitjana Mòbil de 10 Anys",
        "chart_age_years": "Edat (anys)",
        "chart_moving_avg_10": "Mitjana Mòbil (Finestra 10 Anys)",
        "chart_age_mean_sd": "Encert mitjà per edat (± SD)",
        "chart_participants_scatter": "Participants",
        "chart_age_bins_5": "Rendiment per Intervals d'Edat de 5 en 5 Anys",
        "chart_not_enough_ages": "No hi ha dades suficients per a edats entre 6 i 80 anys",
        "chart_not_enough_ages_5_90": "No hi ha dades suficients per a edats entre 5 i 90 anys",
        "chart_scale_score": "Puntuació Escala:",
    }
}


# ============================================================
# --- SESSION STATE ---
# ============================================================
if "language" not in st.session_state:
    st.session_state.language = "en"

# Current language dictionary
T = TEXTS[st.session_state.language]

# Whether the admin has enabled public access to the aggregated view
if "public_mode" not in st.session_state:
    st.session_state.public_mode = False

# Current user role: "admin", "public", or None
if "active_role" not in st.session_state:
    st.session_state.active_role = None

# Whether the admin is currently authenticated
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# End date of the test (editable by admin)
if "test_end_dt" not in st.session_state:
    st.session_state.test_end_dt = DEFAULT_END_DATE


# ============================================================
# --- HELPER FUNCTIONS ---
# ============================================================
def _extract_content_id(name):
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip().lower()
    s = re.sub(r'\.(jpg|jpeg|png|webp|avif|mp4|mov|avi|gif|bmp|tiff)$', '', s)
    s = s.strip()
    s = re.sub(r'-(jpg|jpeg|png|webp|mov)$', '', s)
    m = re.search(r'(\d+[a-z]?)', s)
    if m:
        return m.group(1)
    return s if s else None


def assign_theme(content):
    if pd.isna(content):
        return "Other / Unknown"
    cid = _extract_content_id(content)
    if not cid:
        return "Other / Unknown"
    for theme, ids in THEMES_BY_ID.items():
        if cid in ids:
            return theme
    only_num = re.sub(r'[a-z]+$', '', cid)
    if only_num and only_num != cid:
        for theme, ids in THEMES_BY_ID.items():
            if only_num in ids:
                return theme
    return "Other / Unknown"


def generate_pseudonym(original_name):
    hash_input = str(original_name).strip().lower() + SALT_SECRET
    hash_hex = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    return f"Participant_{hash_hex[:4].upper()}"


def apply_anonymisation(df_in, is_admin_user):
    df_mod = df_in.copy()
    if not is_admin_user:
        df_mod['Participant'] = df_mod['Participant'].apply(generate_pseudonym)
    return df_mod


def load_live_data():
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_csv(RESULTS_FILE)
            return df
        except Exception as e:
            st.error(T["file_read_error"].format(path=RESULTS_FILE, error=e))
            return None
    else:
        st.error(T["file_not_found"].format(path=RESULTS_FILE))
        return None


def normalise_exposure(val):
    val_str = str(val).lower().strip()
    if val_str in ("daily", "6-7days", "6-7dies") or any(k in val_str for k in ['6', '7', 'alta', 'daily', 'diari']):
        return "6-7days"
    elif val_str in ("frequently", "3-5days", "3-5dies") or any(k in val_str for k in ['3', '4', '5', 'medium', 'mitjan', 'frequently']):
        return "3-5days"
    elif val_str in ("occasionally", "1-2days", "1-2dies") or any(k in val_str for k in ['1', '2', 'low', 'ocasional', 'occasionally']):
        return "1-2days"
    elif val_str in ("never", "mai", "nunca", "0"):
        return "never"
    else:
        return val_str


def normalise_gender(val):
    val_str = str(val).lower().strip()
    if val_str in ("male", "home"):
        return "Male"
    elif val_str in ("female", "dona"):
        return "Female"
    elif val_str in ("prefer_not_to_say", "prefer not to say", "prefereixo no dir-ho"):
        return "Prefer not to say"
    else:
        return str(val)


# --- GAUGE DRAWING ---
def draw_aesthetic_gauge(accuracy):
    random_performance = (accuracy - 50) * 2
    scale_value = random_performance / 10

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#FAFAFA')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.30, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    def percent_to_angle(p):
        return 180 - (p / 100) * 180

    if st.session_state.language == "en":
        zones = [
            (0, 20,  '#C62828', "High\nconfusion"),
            (20, 40, '#E53935', "Moderate\nconfusion"),
            (40, 50, '#FB8C00', "Slightly\nworse\nthan random"),
            (50, 60, '#C0CA33', "Slightly\nbetter\nthan random"),
            (60, 80, '#66BB6A', "Moderate\nability"),
            (80, 100,'#2E7D32', "High\nability")
        ]
    else:
        zones = [
            (0, 20,  '#C62828', "Confusió\nalta"),
            (20, 40, '#E53935', "Confusió\nmoderada"),
            (40, 50, '#FB8C00', "Lleugerament\npitjor\nque l'atzar"),
            (50, 60, '#C0CA33', "Lleugerament\nmillor\nque l'atzar"),
            (60, 80, '#66BB6A', "Capacitat\nmoderada"),
            (80, 100,'#2E7D32', "Capacitat\nalta")
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
        mid_p = (start + end) / 2
        ang_mid = np.deg2rad(percent_to_angle(mid_p))
        r_text = 0.92 if 40 <= start < 60 else 0.81
        text_color = 'white' if start < 40 or start >= 80 else '#1A365D'
        ax.text(
            r_text * np.cos(ang_mid), r_text * np.sin(ang_mid), label,
            ha='center', va='center', fontsize=8.5,
            color=text_color, fontweight='bold', linespacing=1.1
        )

    ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=180, color='#333333', lw=2.5))
    ax.add_patch(Arc((0, 0), 2 * r_inner, 2 * r_inner, theta1=0, theta2=180, color='#333333', lw=1.5))

    scale_points = [
        (0, "-10"), (20, "-6"), (40, "-2"),
        (50, "0"), (60, "+2"), (80, "+6"), (100, "+10")
    ]

    for p, val in scale_points:
        ang = np.deg2rad(percent_to_angle(p))
        ax.plot(
            [(r_outer + 0.01) * np.cos(ang), (r_outer + 0.06) * np.cos(ang)],
            [(r_outer + 0.01) * np.sin(ang), (r_outer + 0.06) * np.sin(ang)],
            color='#333333', lw=2
        )
        ax.text(
            (r_outer + 0.15) * np.cos(ang),
            (r_outer + 0.15) * np.sin(ang),
            val, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1D3557' if val == "0" else '#333333'
        )

    result_angle = np.deg2rad(percent_to_angle(accuracy))
    cos_a, sin_a = np.cos(result_angle), np.sin(result_angle)
    ax.plot([0.08 * cos_a, 1.18 * cos_a], [0.08 * sin_a, 1.18 * sin_a], color='#111111', linestyle=':', linewidth=2.5, zorder=9)
    ax.annotate('', xy=(0.58 * cos_a, 0.58 * sin_a), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->,head_length=0.7,head_width=0.45', color='#111111', lw=5.5), zorder=10)
    ax.add_patch(Circle((0, 0), 0.05, facecolor='#111111', zorder=11))

    sign = "+" if scale_value > 0 else ""
    scale_value_str = f"{sign}{scale_value:.1f}".replace('.', ',')
    ax.text(
        0, -0.15, f"{T['chart_scale_score']} {scale_value_str} / 10",
        ha='center', va='center', fontsize=11, color='#111111', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EAEAEA', edgecolor='#CCCCCC', lw=1.2)
    )

    plt.tight_layout()
    return fig


def show_breakdown_and_comments(df_sub, is_plural=False):
    ai_responses = df_sub[df_sub["Solution"] == "AI"]
    real_responses = df_sub[df_sub["Solution"] == "REAL"]

    total_ai = len(ai_responses)
    correct_ai = ai_responses["Correct"].sum() if total_ai > 0 else 0
    pct_ai = (correct_ai / total_ai * 100) if total_ai > 0 else 0
    error_ai = 100 - pct_ai if total_ai > 0 else 0

    total_real = len(real_responses)
    correct_real = real_responses["Correct"].sum() if total_real > 0 else 0
    pct_real = (correct_real / total_real * 100) if total_real > 0 else 0
    error_real = 100 - pct_real if total_real > 0 else 0

    total_questions = len(df_sub)
    total_correct = df_sub["Correct"].sum() if total_questions > 0 else 0
    correct_percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0

    st.markdown(f"#### {T['by_content_type']}")

    col_ai, col_real = st.columns(2)
    with col_ai:
        st.markdown(f"**{T['ai_content']}**")
        st.write(f"{T['correct_label']}: {correct_ai} / {total_ai} ({int(round(pct_ai))}%)")
        st.progress(pct_ai / 100)
    with col_real:
        st.markdown(f"**{T['real_content']}**")
        st.write(f"{T['correct_label']}: {correct_real} / {total_real} ({int(round(pct_real))}%)")
        st.progress(pct_real / 100)

    st.markdown(f"#### {T['diagnostic_comments']}")

    suffix = "_plural" if is_plural else "_single"
    if correct_percentage >= 80:
        st.success(T[f"ability_high{suffix}"])
    elif correct_percentage >= 60:
        st.success(T[f"ability_moderate{suffix}"])
    elif correct_percentage >= 50:
        st.success(T[f"better_random{suffix}"])
    elif correct_percentage >= 40:
        st.success(T[f"worse_random{suffix}"])
    elif correct_percentage >= 20:
        st.info(T[f"confusion_moderate{suffix}"])
    else:
        st.warning(T[f"confusion_high{suffix}"])

    st.info(
        T[f"error_summary{suffix}"].format(
            ai_error=int(round(error_ai)),
            real_error=int(round(error_real))
        )
    )


# ============================================================
# --- SCIENTIFIC CHARTS ---
# ============================================================
def plot_participant_evolution(df):
    df_p_time = df.groupby('Participant')['Date'].min().reset_index().sort_values('Date')
    df_p_time['Count'] = 1
    df_p_time['Cumulative'] = df_p_time['Count'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    ax.plot(df_p_time['Date'], df_p_time['Cumulative'], color='#1D3557', linewidth=2.5,
            marker='o', markersize=4, markerfacecolor='#E63946', label='Total')
    ax.set_xlabel(T["chart_date_time"], fontsize=10, fontweight='bold')
    ax.set_ylabel(T["chart_cumulative_participants"], fontsize=10, fontweight='bold')
    ax.set_title(T["chart_evolution_title"], fontsize=11, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_scientific_age_4_groups(participants_summary):
    stats_age = participants_summary.groupby('Age_Group', observed=False)['Accuracy_Percentage'].agg(
        mean='mean', std='std', count='count'
    ).reset_index()
    stats_age['std'] = stats_age['std'].fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')

    categories = stats_age['Age_Group'].astype(str)
    x = np.arange(len(categories))
    y = stats_age['mean']
    yerr = stats_age['std']
    colors_age = ['#1D3557', '#457B9D', '#2A9D8F', '#E76F51']

    bars = ax.bar(
        x, y, yerr=yerr, capsize=6, color=colors_age[:len(categories)], edgecolor='#111111',
        alpha=0.88, error_kw=dict(ecolor='#D32F2F', lw=2, capsize=6, capthick=2),
        label=T['chart_mean_accuracy_sd']
    )

    if len(x) >= 2 and not y.isna().all():
        valid_mask = ~y.isna()
        x_val = x[valid_mask]
        y_val = y[valid_mask]
        if len(x_val) >= 2:
            deg = 2 if len(x_val) >= 3 else 1
            z = np.polyfit(x_val, y_val, deg)
            p = np.poly1d(z)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_smooth, p(x_smooth), color='#E65100', linestyle='--', linewidth=2.5,
                    label=T['chart_trend_line'])

    ax.axhline(50, color='#757575', linestyle=':', label=T['chart_random_50'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_ylabel(T['col_mean_accuracy'], fontsize=10, fontweight='bold')
    ax.set_title(T['chart_perf_age_groups'], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=8.5)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            sd_val = yerr.iloc[i]
            ax.text(bar.get_x() + bar.get_width()/2., height + sd_val + 2,
                    f"{height:.1f}%\n(±{sd_val:.1f})", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_age_bins_5(participants_summary):
    df_age = participants_summary.dropna(subset=['Age_num']).copy()
    df_age = df_age[(df_age['Age_num'] >= 5) & (df_age['Age_num'] <= 90)]

    if df_age.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, T["chart_not_enough_ages_5_90"], ha='center', va='center')
        return fig

    bins = list(range(5, 95, 5))
    labels = [f"{b}-{b+4}" for b in bins[:-1]]
    df_age['Age_Group_5'] = pd.cut(df_age['Age_num'], bins=bins, labels=labels, right=False)

    stats_5 = df_age.groupby('Age_Group_5', observed=False)['Accuracy_Percentage'].agg(
        mean='mean', std='std', count='count'
    ).reset_index()
    stats_5 = stats_5[stats_5['count'] > 0].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')

    categories = stats_5['Age_Group_5'].astype(str)
    x = np.arange(len(categories))
    y = stats_5['mean']
    yerr = stats_5['std'].fillna(0)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(categories)))

    bars = ax.bar(x, y, yerr=yerr, capsize=4, color=colors, edgecolor='#111111', alpha=0.88,
                  error_kw=dict(ecolor='#C2185B', lw=1.5, capsize=4))

    ax.axhline(50, color='#757575', linestyle=':', label=T['chart_random_50'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, fontsize=8.5, fontweight='bold')
    ax.set_xlabel(T["chart_age_intervals_5"], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['col_mean_accuracy'], fontsize=10, fontweight='bold')
    ax.set_title(T["chart_perf_age_5"], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = stats_5['count'].iloc[i]
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + yerr.iloc[i] + 1.5,
                    f"{height:.1f}%\n(n={count})", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_top25_age_distribution(participants_summary):
    top_25 = participants_summary.nlargest(25, 'Accuracy_Percentage')
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    colors_age = ['#1D3557', '#457B9D', '#2A9D8F', '#E76F51']
    counts_top = top_25.groupby('Age_Group', observed=False)['Participant'].count().reset_index()
    bars = ax.bar(counts_top['Age_Group'].astype(str), counts_top['Participant'],
                  color=colors_age[:len(counts_top)], edgecolor='#111111', alpha=0.85)
    ax.set_title(T['chart_top25_age_dist'], fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(T['chart_age_group_label'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_bottom25_age_distribution(participants_summary):
    bottom_25 = participants_summary.nsmallest(25, 'Accuracy_Percentage')
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    colors_age = ['#1D3557', '#457B9D', '#2A9D8F', '#E76F51']
    counts_bot = bottom_25.groupby('Age_Group', observed=False)['Participant'].count().reset_index()
    bars = ax.bar(counts_bot['Age_Group'].astype(str), counts_bot['Participant'],
                  color=colors_age[:len(counts_bot)], edgecolor='#111111', alpha=0.85)
    ax.set_title(T['chart_bottom25_age_dist'], fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(T['chart_age_group_label'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_exposure_top_bottom(participants_summary):
    df_exp = participants_summary.copy()
    df_exp['Exp_Norm'] = df_exp['IA_Exposure'].apply(normalise_exposure)
    order_exp = ['6-7days', '3-5days', '1-2days', 'never']
    df_exp['Exp_Norm'] = pd.Categorical(df_exp['Exp_Norm'], categories=order_exp, ordered=True)
    top_25 = df_exp.nlargest(25, 'Accuracy_Percentage')
    bottom_25 = df_exp.nsmallest(25, 'Accuracy_Percentage')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='#FAFAFA')
    ax1.set_facecolor('#FFFFFF')
    ax2.set_facecolor('#FFFFFF')
    colors_exp = ['#1B4332', '#2D6A4F', '#52B788', '#D8F3DC']

    counts_top = top_25.groupby('Exp_Norm', observed=False)['Participant'].count().reset_index()
    bars1 = ax1.bar(counts_top['Exp_Norm'].astype(str), counts_top['Participant'],
                    color=colors_exp[:len(counts_top)], edgecolor='#111111', alpha=0.85)
    ax1.set_title(T['chart_top25_exposure'], fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel(T['chart_exposure_level'], fontsize=10, fontweight='bold')
    ax1.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    counts_bot = bottom_25.groupby('Exp_Norm', observed=False)['Participant'].count().reset_index()
    bars2 = ax2.bar(counts_bot['Exp_Norm'].astype(str), counts_bot['Participant'],
                    color=colors_exp[:len(counts_bot)], edgecolor='#111111', alpha=0.85)
    ax2.set_title(T['chart_bottom25_exposure'], fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel(T['chart_exposure_level'], fontsize=10, fontweight='bold')
    ax2.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_gender_top_bottom(participants_summary):
    df_gen = participants_summary.copy()
    df_gen = df_gen[~df_gen['Gender'].astype(str).str.lower().str.contains('prefer|no dir|sense indicar', na=False)]
    top_25 = df_gen.nlargest(25, 'Accuracy_Percentage')
    bottom_25 = df_gen.nsmallest(25, 'Accuracy_Percentage')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='#FAFAFA')
    ax1.set_facecolor('#FFFFFF')
    ax2.set_facecolor('#FFFFFF')
    colors_gen = ['#1B5E20', '#0D47A1', '#E65100']

    counts_top = top_25.groupby('Gender', observed=False)['Participant'].count().reset_index()
    bars1 = ax1.bar(counts_top['Gender'].astype(str), counts_top['Participant'],
                    color=colors_gen[:len(counts_top)], edgecolor='#111111', alpha=0.85)
    ax1.set_title(T['chart_top25_gender'], fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel(T['chart_gender'], fontsize=10, fontweight='bold')
    ax1.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    counts_bot = bottom_25.groupby('Gender', observed=False)['Participant'].count().reset_index()
    bars2 = ax2.bar(counts_bot['Gender'].astype(str), counts_bot['Participant'],
                    color=colors_gen[:len(counts_bot)], edgecolor='#111111', alpha=0.85)
    ax2.set_title(T['chart_bottom25_gender'], fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel(T['chart_gender'], fontsize=10, fontweight='bold')
    ax2.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_scientific_age_continuous(participants_summary):
    df_age = participants_summary.dropna(subset=['Age_num']).copy()
    df_age = df_age[(df_age['Age_num'] >= 6) & (df_age['Age_num'] <= 80)]

    if df_age.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, T["chart_not_enough_ages"], ha='center', va='center')
        return fig

    stats_cont = df_age.groupby('Age_num')['Accuracy_Percentage'].agg(
        mean='mean', std='std', count='count'
    ).reset_index()
    stats_cont['std'] = stats_cont['std'].fillna(0)

    full_ages = pd.DataFrame({'Age_num': np.arange(6, 81)})
    merged_ages = pd.merge(full_ages, stats_cont, on='Age_num', how='left')
    merged_ages['mean_interp'] = merged_ages['mean'].interpolate(method='linear')
    merged_ages['mean_rolling_10'] = merged_ages['mean_interp'].rolling(window=10, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    x = stats_cont['Age_num']
    y = stats_cont['mean']
    yerr = stats_cont['std']

    ax.bar(x, y, yerr=yerr, capsize=3, color='#4A148C', alpha=0.35, edgecolor='#311B92',
           error_kw=dict(ecolor='#C2185B', lw=1.2, capsize=3), label=T['chart_age_mean_sd'])
    ax.scatter(df_age['Age_num'], df_age['Accuracy_Percentage'], color='#1565C0', alpha=0.5, s=20,
               label=T['chart_participants_scatter'])
    ax.plot(merged_ages['Age_num'], merged_ages['mean_rolling_10'], color='#D32F2F', linewidth=2.8,
            label=T['chart_moving_avg_10'])
    ax.axhline(50, color='#757575', linestyle=':', label=T['chart_random_50'])
    ax.set_xlabel(T['chart_age_years'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['col_mean_accuracy'], fontsize=10, fontweight='bold')
    ax.set_title(T['chart_continuous_age'], fontsize=11, fontweight='bold', pad=12)
    ax.set_xlim(5, 81)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    return fig


def plot_scientific_exposure(participants_summary):
    df_exp = participants_summary.copy()
    df_exp['Exp_Norm'] = df_exp['IA_Exposure'].apply(normalise_exposure)
    order_exp = ['6-7days', '3-5days', '1-2days', 'never']
    df_exp['Exp_Norm'] = pd.Categorical(df_exp['Exp_Norm'], categories=order_exp, ordered=True)

    stats_exp = df_exp.groupby('Exp_Norm', observed=False)['Accuracy_Percentage'].agg(
        mean='mean', std='std', count='count'
    ).reset_index()
    stats_exp['std'] = stats_exp['std'].fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    categories = stats_exp['Exp_Norm'].astype(str)
    x = np.arange(len(categories))
    y = stats_exp['mean']
    yerr = stats_exp['std']
    colors_exp = ['#1B4332', '#2D6A4F', '#52B788', '#D8F3DC']

    bars = ax.bar(x, y, yerr=yerr, capsize=6, color=colors_exp[:len(categories)],
                  alpha=0.9, edgecolor='#111111', lw=1.2,
                  error_kw=dict(ecolor='#C2185B', lw=2, capsize=6, capthick=2),
                  label=T['chart_mean_accuracy_sd'])

    ax.axhline(50, color='#757575', linestyle=':', label=T['chart_random_50'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_xlabel(T['chart_exposure_more_less'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['col_mean_accuracy'], fontsize=10, fontweight='bold')
    ax.set_title(T['chart_perf_exposure'], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            sd_val = yerr.iloc[i]
            ax.text(bar.get_x() + bar.get_width()/2., height + sd_val + 2,
                    f"{height:.1f}%\n(±{sd_val:.1f})", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_participants_per_exposure(participants_summary):
    df_exp = participants_summary.copy()
    df_exp['Exp_Norm'] = df_exp['IA_Exposure'].apply(normalise_exposure)
    order_exp = ['6-7days', '3-5days', '1-2days', 'never']
    df_exp['Exp_Norm'] = pd.Categorical(df_exp['Exp_Norm'], categories=order_exp, ordered=True)
    counts_exp = df_exp.groupby('Exp_Norm', observed=False)['Participant'].count().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    categories = counts_exp['Exp_Norm'].astype(str)
    x = np.arange(len(categories))
    y = counts_exp['Participant']
    colors_exp = ['#1F4E5B', '#2E7D8E', '#6FB3B8', '#BAD7E9']
    bars = ax.bar(x, y, color=colors_exp[:len(categories)], edgecolor='#111111', lw=1.2, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_xlabel(T['chart_exposure_more_less'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['chart_n_participants'], fontsize=10, fontweight='bold')
    ax.set_title(T['chart_participants_per_exposure'], fontsize=11, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{int(height)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_exposure_cross_gender(participants_summary):
    df_exp = participants_summary.copy()
    df_exp = df_exp[~df_exp['Gender'].astype(str).str.lower().str.contains('prefer|no dir|sense indicar', na=False)]
    df_exp['Exp_Norm'] = df_exp['IA_Exposure'].apply(normalise_exposure)
    order_exp = ['6-7days', '3-5days', '1-2days', 'never']
    df_exp['Exp_Norm'] = pd.Categorical(df_exp['Exp_Norm'], categories=order_exp, ordered=True)
    ct_pct = pd.crosstab(df_exp['Exp_Norm'], df_exp['Gender'], normalize='columns') * 100

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    genders = ct_pct.columns.tolist()
    categories = [str(c) for c in ct_pct.index]
    x = np.arange(len(categories))
    width = 0.75 / max(len(genders), 1)
    colors_gen = ['#2B5C8F', '#E63946', '#457B9D', '#2A9D8F']

    for i, g in enumerate(genders):
        y_vals = ct_pct[g].values
        bars = ax.bar(x + i * width - (len(genders) - 1) * width / 2, y_vals, width=width,
                      label=str(g), color=colors_gen[i % len(colors_gen)],
                      edgecolor='#111111', linewidth=0.8, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                        f"{h:.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_xlabel(T['chart_exposure_level'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['chart_pct_of_gender'], fontsize=9.5, fontweight='bold')
    ax.set_title(T['chart_exposure_by_gender'], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    ax.legend(title=T['chart_gender'], fontsize=8.5, title_fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_exposure_cross_age(participants_summary):
    df_exp = participants_summary.dropna(subset=['Age_Group']).copy()
    df_exp['Exp_Norm'] = df_exp['IA_Exposure'].apply(normalise_exposure)
    order_exp = ['6-7days', '3-5days', '1-2days', 'never']
    df_exp['Exp_Norm'] = pd.Categorical(df_exp['Exp_Norm'], categories=order_exp, ordered=True)
    ct_pct = pd.crosstab(df_exp['Exp_Norm'], df_exp['Age_Group'], normalize='columns') * 100

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    age_groups = ct_pct.columns.tolist()
    categories = [str(c) for c in ct_pct.index]
    x = np.arange(len(categories))
    width = 0.75 / max(len(age_groups), 1)
    colors_age = ['#1D3557', '#457B9D', '#2A9D8F', '#E76F51']

    for i, g in enumerate(age_groups):
        y_vals = ct_pct[g].values
        bars = ax.bar(x + i * width - (len(age_groups) - 1) * width / 2, y_vals, width=width,
                      label=str(g), color=colors_age[i % len(colors_age)],
                      edgecolor='#111111', linewidth=0.8, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                        f"{h:.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_xlabel(T['chart_exposure_level'], fontsize=10, fontweight='bold')
    ax.set_ylabel(T['chart_pct_of_age'], fontsize=9.5, fontweight='bold')
    ax.set_title(T['chart_exposure_by_age'], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    ax.legend(title=T['chart_age_group_label'], fontsize=8.5, title_fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_scientific_gender(participants_summary):
    df_gen = participants_summary.copy()
    df_gen = df_gen[~df_gen['Gender'].astype(str).str.lower().str.contains('prefer|no dir|sense indicar', na=False)]
    stats_gen = df_gen.groupby('Gender')['Accuracy_Percentage'].agg(
        mean='mean', std='std', count='count'
    ).reset_index()
    stats_gen['std'] = stats_gen['std'].fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    categories = stats_gen['Gender'].astype(str)
    x = np.arange(len(categories))
    y = stats_gen['mean']
    yerr = stats_gen['std']
    colors = ['#1B5E20', '#0D47A1', '#E65100', '#4A148C'][:len(categories)]

    bars = ax.bar(x, y, yerr=yerr, capsize=6, color=colors, alpha=0.8, edgecolor='#111111',
                  error_kw=dict(ecolor='#B71C1C', lw=2, capsize=6, capthick=2),
                  label=T['chart_mean_accuracy_sd'])

    ax.axhline(50, color='#757575', linestyle=':', label=T['chart_random_50'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_ylabel(T['col_mean_accuracy'], fontsize=10, fontweight='bold')
    ax.set_title(T['chart_perf_gender'], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            sd_val = yerr.iloc[i]
            ax.text(bar.get_x() + bar.get_width()/2., height + sd_val + 2,
                    f"{height:.1f}%\n(±{sd_val:.1f})", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_accuracy_per_image(df):
    img_stats = df.groupby('Image')['Correct'].agg(['mean', 'count']).reset_index()
    img_stats['Accuracy_Percentage'] = img_stats['mean'] * 100

    def extract_number(name):
        nums = re.findall(r'\d+', str(name))
        return int(nums[0]) if nums else 999

    img_stats['num_sort'] = img_stats['Image'].apply(extract_number)
    img_stats = img_stats.sort_values(by='num_sort').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor='#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    x = np.arange(len(img_stats))
    y = img_stats['Accuracy_Percentage']
    labels = [str(name) for name in img_stats['Image']]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(img_stats)))

    bars = ax.bar(x, y, color=colors, edgecolor='#222222', linewidth=0.7, alpha=0.9)
    ax.axhline(50, color='#D32F2F', linestyle='--', linewidth=1.5, label=T['chart_random_50'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8, fontweight='bold')
    ax.set_ylabel(T['col_mean_accuracy'], fontsize=10, fontweight='bold')
    ax.set_xlabel(T['col_image'], fontsize=10, fontweight='bold')
    ax.set_title(T['chart_accuracy_per_image'], fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f"{height:.0f}%", ha='center', va='bottom', fontsize=6.5, rotation=90, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_error_ai_vs_real(df):
    img_stats = df.groupby(['Image', 'Solution'])['Correct'].agg(['mean', 'count']).reset_index()
    img_stats['Error_Percentage'] = (1 - img_stats['mean']) * 100

    df_ai = img_stats[img_stats['Solution'] == 'AI'].sort_values(by='Error_Percentage', ascending=False)
    df_real = img_stats[img_stats['Solution'] == 'REAL'].sort_values(by='Error_Percentage', ascending=False)

    mean_error_global = img_stats['Error_Percentage'].mean()
    mean_error_ai = df_ai['Error_Percentage'].mean() if not df_ai.empty else 0
    mean_error_real = df_real['Error_Percentage'].mean() if not df_real.empty else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor='#FAFAFA', sharey=True)
    ax1.set_facecolor('#FFFFFF')
    ax2.set_facecolor('#FFFFFF')

    bars1 = ax1.bar(df_ai['Image'], df_ai['Error_Percentage'], color='#D90429', alpha=0.85, edgecolor='#111111', lw=0.8)
    ax1.axhline(mean_error_ai, color='#8D0801', linestyle=':', linewidth=2.5,
                label=T['chart_mean_error_ai'].format(v=mean_error_ai))
    ax1.axhline(mean_error_global, color='#2B2D42', linestyle='--', linewidth=1.5, alpha=0.7,
                label=T['chart_mean_error_global'].format(v=mean_error_global))
    ax1.set_title(T['chart_error_ai'], fontsize=11, fontweight='bold', pad=12)
    ax1.set_ylabel("Error (%)", fontsize=10, fontweight='bold')
    ax1.set_xticklabels(df_ai['Image'], rotation=90, fontsize=8, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=8.5)
    ax1.set_ylim(0, 115)

    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax1.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.0f}%", ha='center', va='bottom', fontsize=7, rotation=90, fontweight='bold')

    bars2 = ax2.bar(df_real['Image'], df_real['Error_Percentage'], color='#0077B6', alpha=0.85, edgecolor='#111111', lw=0.8)
    ax2.axhline(mean_error_real, color='#03045E', linestyle=':', linewidth=2.5,
                label=T['chart_mean_error_real'].format(v=mean_error_real))
    ax2.axhline(mean_error_global, color='#2B2D42', linestyle='--', linewidth=1.5, alpha=0.7,
                label=T['chart_mean_error_global'].format(v=mean_error_global))
    ax2.set_title(T['chart_error_real'], fontsize=11, fontweight='bold', pad=12)
    ax2.set_xticklabels(df_real['Image'], rotation=90, fontsize=8, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=8.5)

    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax2.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.0f}%", ha='center', va='bottom', fontsize=7, rotation=90, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# --- SIDEBAR (ALWAYS RENDERED FIRST) ---
# ============================================================
with st.sidebar:
    # Language selector
    lang_options = {"en": "English", "ca": "Català"}
    selected_lang = st.selectbox(
        "Language / Idioma:",
        list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.language),
        key="lang_selector"
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    st.markdown("---")
    st.header(T["config_header"])

    # --- Build the user selector based on public_mode ---
    if st.session_state.public_mode:
        user_options = [T["user_admin"], T["user_public"]]
    else:
        user_options = [T["user_admin"]]

    # Default selection depends on current role
    if st.session_state.active_role == "public":
        default_idx = user_options.index(T["user_public"])
    else:
        default_idx = 0

    selected_user = st.selectbox(T["active_user"], user_options, index=default_idx)

    # --- Handle selection ---
    if selected_user == T["user_admin"]:
        if not st.session_state.admin_authenticated:
            pw_input = st.text_input(T["password_label"], type="password", key="pwd_admin_input")
            if pw_input:
                if pw_input == PASSWORD_ADMIN:
                    st.session_state.admin_authenticated = True
                    st.session_state.active_role = "admin"
                    st.success(T["password_ok"])
                    st.rerun()
                else:
                    st.error(T["password_ko"])
            st.session_state.active_role = None
        else:
            st.session_state.active_role = "admin"
            st.success(T["role_admin"])
    else:
        st.session_state.active_role = "public"
        st.session_state.admin_authenticated = False
        st.info(T["role_public"])

    # Convenience flags
    is_admin = st.session_state.active_role == "admin"
    is_public = st.session_state.active_role == "public"

    # --- Admin-only controls ---
    if is_admin:
        st.markdown("---")
        st.subheader("🔧 Admin controls")

        public_toggle = st.checkbox(
            T["public_mode_toggle"],
            value=st.session_state.public_mode,
            help=T["public_mode_toggle_help"],
            key="public_mode_checkbox"
        )
        if public_toggle != st.session_state.public_mode:
            st.session_state.public_mode = public_toggle
            st.rerun()

        if st.session_state.public_mode:
            st.caption(T["public_mode_active"])

        if st.button(T["logout_button"], use_container_width=True):
            st.session_state.admin_authenticated = False
            st.session_state.active_role = None
            st.rerun()

    st.markdown("---")

    # Test control
    st.subheader(T["test_control_header"])

    # Start date (auto, read-only, from CSV)
    df_sidebar = load_live_data()
    if df_sidebar is not None and not df_sidebar.empty and 'Date' in df_sidebar.columns:
        start_dt = pd.to_datetime(df_sidebar['Date']).min()
    else:
        start_dt = None

    st.markdown(T["test_start"])
    if start_dt is not None:
        st.markdown(f"`{start_dt.strftime('%Y-%m-%d %H:%M:%S')}`")
    else:
        st.markdown("`N/A`")
    st.caption(T["test_start_caption"])

    # End date (editable by admin, read-only for public)
    st.markdown(T["test_end"])
    end_col1, end_col2 = st.columns(2)
    with end_col1:
        new_end_date = st.date_input(
            T["test_end_date"],
            value=st.session_state.test_end_dt.date(),
            disabled=not is_admin,
            key="input_end_date"
        )
    with end_col2:
        new_end_time = st.time_input(
            T["test_end_time"],
            value=st.session_state.test_end_dt.time(),
            disabled=not is_admin,
            key="input_end_time"
        )

    new_end_dt = datetime.combine(new_end_date, new_end_time)

    if start_dt is not None and new_end_dt <= start_dt:
        st.error(T["date_range_invalid"])
    else:
        st.session_state.test_end_dt = new_end_dt
        if is_admin:
            st.caption(T["test_end_caption_admin"])
        else:
            st.caption(T["test_end_caption_public"])

    st.markdown("---")

    if st.button(T["refresh_button"], use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if is_admin and st.button(T["shutdown_button"], use_container_width=True):
        st.warning(T["shutdown_warning"])
        os._exit(0)


# ============================================================
# --- ACCESS GATE (AFTER SIDEBAR, BEFORE MAIN CONTENT) ---
# ============================================================
if st.session_state.active_role is None:
    st.title(T["main_title"])
    st.info(T["no_access_message"])
    st.stop()


# ============================================================
# --- MAIN UI ---
# ============================================================
if st.session_state.active_role == "admin":
    st.title(T["main_title"])
elif st.session_state.active_role == "public":
    st.title(T["main_title_public"])
    st.info(T["public_mode_notice"])

# Load data from local CSV
df = load_live_data()


# ============================================================
# --- DATA PROCESSING AND TABS ---
# ============================================================
if df is not None and not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])
    effective_start = df['Date'].min()
    effective_end = st.session_state.test_end_dt
    df = df[(df['Date'] >= effective_start) & (df['Date'] <= effective_end)]

    # Anonymisation based on role
    is_admin = st.session_state.active_role == "admin"
    df = apply_anonymisation(df, is_admin)

    df['Gender'] = df['Gender'].apply(normalise_gender)

    participants_summary = df.groupby(['Participant', 'Age', 'Gender', 'IA_Exposure']).agg(
        Total_Responses=('Correct', 'count'),
        Total_Correct=('Correct', 'sum'),
        Accuracy_Percentage=('Correct', lambda x: (x.sum() / x.count()) * 100),
        Participation_Date=('Date', lambda x: x.max().strftime('%Y-%m-%d %H:%M') if pd.notnull(x.max()) else 'N/A')
    ).reset_index()

    participants_summary['AICI'] = ((participants_summary['Accuracy_Percentage'] - 50) * 2) / 10

    participants_summary['Age_num'] = pd.to_numeric(participants_summary['Age'], errors='coerce')
    age_bins = [6, 18, 30, 50, 81]
    age_labels = ['6-17 years', '18-29 years', '30-49 years', '50-80 years'] if st.session_state.language == "en" else ['6-17 anys', '18-29 anys', '30-49 anys', '50-80 anys']
    participants_summary['Age_Group'] = pd.cut(participants_summary['Age_num'], bins=age_bins, labels=age_labels, right=False)

    # Build tabs based on role
    if is_admin:
        tab_global, tab_charts, tab_participant, tab_content, tab_themes, tab_images, tab_data = st.tabs([
            T["tab_global"], T["tab_charts"], T["tab_participant"],
            T["tab_content"], T["tab_themes"], T["tab_images"], T["tab_data"]
        ])
    else:
        tab_global, tab_charts, tab_content, tab_themes, tab_images = st.tabs([
            T["tab_global"], T["tab_charts"],
            T["tab_content"], T["tab_themes"], T["tab_images"]
        ])
        tab_participant = None
        tab_data = None

    # ----------------------------------------------------
    # TAB 1: GLOBAL METRICS
    # ----------------------------------------------------
    with tab_global:
        st.subheader(T["global_subtitle"])

        total_participants = len(participants_summary)
        mean_accuracy_global = df['Correct'].mean() * 100
        random_performance_global = (mean_accuracy_global - 50) * 2
        aici_global = random_performance_global / 10

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(T["metric_total_participants"], total_participants)
        col2.metric(T["metric_mean_accuracy"], f"{mean_accuracy_global:.1f}%")
        col3.metric(T["metric_random_adjusted"], f"{random_performance_global:+.1f}%")
        col4.metric(T["metric_icia_global"], f"{aici_global:+.1f}")

        st.markdown("---")
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown(f"### {T['scale_title']}")
            st.pyplot(draw_aesthetic_gauge(mean_accuracy_global))
        with c_right:
            st.markdown(f"### {T['age_group_perf_title']}")
            age_df = participants_summary.groupby('Age_Group', observed=False)['Accuracy_Percentage'].agg(['mean', 'count']).reset_index()
            age_df['mean_fmt'] = age_df['mean'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
            st.dataframe(
                age_df[['Age_Group', 'mean_fmt', 'count']].rename(
                    columns={'Age_Group': T['col_age_group'], 'mean_fmt': T['col_mean_accuracy'], 'count': T['col_participants']}
                ),
                use_container_width=True, hide_index=True
            )

        st.markdown("---")
        st.markdown(f"### {T['extra_demographics']}")
        col_gen, col_exp = st.columns(2)
        with col_gen:
            st.markdown(f"#### {T['gender_perf_title']}")
            gen_df = participants_summary.groupby('Gender')['Accuracy_Percentage'].agg(['mean', 'count']).reset_index()
            gen_df['mean_fmt'] = gen_df['mean'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
            st.dataframe(
                gen_df[['Gender', 'mean_fmt', 'count']].rename(
                    columns={'Gender': T['col_gender'], 'mean_fmt': T['col_mean_accuracy'], 'count': T['col_participants']}
                ),
                use_container_width=True, hide_index=True
            )
        with col_exp:
            st.markdown(f"#### {T['exposure_perf_title']}")
            exp_df = participants_summary.groupby('IA_Exposure')['Accuracy_Percentage'].agg(['mean', 'count']).reset_index()
            exp_df['mean_fmt'] = exp_df['mean'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
            st.dataframe(
                exp_df[['IA_Exposure', 'mean_fmt', 'count']].rename(
                    columns={'IA_Exposure': T['col_exposure'], 'mean_fmt': T['col_mean_accuracy'], 'count': T['col_participants']}
                ),
                use_container_width=True, hide_index=True
            )
        st.markdown("---")
        show_breakdown_and_comments(df, is_plural=True)

    # ----------------------------------------------------
    # TAB 2: GLOBAL CHARTS
    # ----------------------------------------------------
    with tab_charts:
        st.subheader(T["tab_charts"])
        st.markdown("### 1. Age analysis" if st.session_state.language == "en" else "### 1. Anàlisi per Edat")
        col_age1, col_age2 = st.columns([1, 1])
        with col_age1:
            st.markdown("#### A) 4 age groups histogram" if st.session_state.language == "en" else "#### A) Histogrames per als 4 Grups d'Edat")
            st.pyplot(plot_scientific_age_4_groups(participants_summary))
        with col_age2:
            st.markdown("#### B) Continuous distribution with 10-year moving average" if st.session_state.language == "en" else "#### B) Distribució Contínua amb Mitjana Mòbil (10 Anys)")
            st.pyplot(plot_scientific_age_continuous(participants_summary))
        st.markdown("---")
        st.markdown(f"#### C) {T['chart_perf_age_5']}")
        st.pyplot(plot_age_bins_5(participants_summary))
        st.markdown("---")
        col_de1, col_de2 = st.columns([1, 1])
        with col_de1:
            st.markdown(f"#### D) {T['chart_top25_age_dist']}")
            st.pyplot(plot_top25_age_distribution(participants_summary))
        with col_de2:
            st.markdown(f"#### E) {T['chart_bottom25_age_dist']}")
            st.pyplot(plot_bottom25_age_distribution(participants_summary))
        st.markdown("---")
        st.markdown("### 2. AI exposure analysis" if st.session_state.language == "en" else "### 2. Anàlisi per Exposició a la IA")
        col_exp1, col_exp2 = st.columns([1, 1])
        with col_exp1:
            st.markdown(f"#### A) {T['chart_perf_exposure']}")
            st.pyplot(plot_scientific_exposure(participants_summary))
        with col_exp2:
            st.markdown(f"#### B) {T['chart_participants_per_exposure']}")
            st.pyplot(plot_participants_per_exposure(participants_summary))
        st.markdown("---")
        st.markdown(f"#### C) {T['chart_top25_exposure']} vs {T['chart_bottom25_exposure']}")
        st.pyplot(plot_exposure_top_bottom(participants_summary))
        st.markdown("---")
        st.markdown("### 3. Performance by gender" if st.session_state.language == "en" else "### 3. Rendiment per Sexe")
        st.markdown(f"#### A) {T['chart_perf_gender']}")
        st.pyplot(plot_scientific_gender(participants_summary))
        st.markdown("---")
        st.markdown(f"#### B) {T['chart_top25_gender']} vs {T['chart_bottom25_gender']}")
        st.pyplot(plot_gender_top_bottom(participants_summary))
        st.markdown("---")
        st.markdown("### 4. Cross-data analysis" if st.session_state.language == "en" else "### 4. Anàlisi de dades creuat")
        col_cr1, col_cr2 = st.columns([1, 1])
        with col_cr1:
            st.markdown(f"#### A) {T['chart_exposure_by_gender']}")
            st.pyplot(plot_exposure_cross_gender(participants_summary))
        with col_cr2:
            st.markdown(f"#### B) {T['chart_exposure_by_age']}")
            st.pyplot(plot_exposure_cross_age(participants_summary))

    # ----------------------------------------------------
    # TAB 3: PARTICIPANT INSPECTOR (ADMIN ONLY)
    # ----------------------------------------------------
    if is_admin and tab_participant is not None:
        with tab_participant:
            st.subheader(T["participant_subtitle"])
            participant_list = participants_summary['Participant'].unique().tolist()
            selected_participant = st.selectbox(T["select_participant"], participant_list)
            if selected_participant:
                p_data = participants_summary[participants_summary['Participant'] == selected_participant].iloc[0]
                p_df = df[df['Participant'] == selected_participant]
                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.write(f"{T['p_age']} {p_data['Age']} ({p_data['Age_Group']})")
                col_p1.write(f"{T['p_gender']} {p_data['Gender']}")
                col_p2.write(f"{T['p_exposure']} {p_data['IA_Exposure']}")
                col_p2.write(f"{T['p_test_date']} {p_data['Participation_Date']}")
                col_p3.metric(T["p_accuracy"], f"{p_data['Accuracy_Percentage']:.1f}%")
                col_p3.metric(T["p_icia"], f"{p_data['AICI']:+.1f}")
                st.markdown("---")
                c_gauge, c_detail = st.columns([1, 1])
                with c_gauge:
                    st.pyplot(draw_aesthetic_gauge(p_data['Accuracy_Percentage']))
                with c_detail:
                    st.markdown(f"#### {T['response_detail']}")
                    p_df_display = p_df[['Image', 'Solution', 'Response', 'Correct']].copy()
                    p_df_display['Correct'] = p_df_display['Correct'].replace({1: T["correct_yes"], 0: T["correct_no"]})
                    st.dataframe(p_df_display, use_container_width=True, height=320, hide_index=True)
                st.markdown("---")
                show_breakdown_and_comments(p_df, is_plural=False)
            st.markdown("---")
            st.markdown(f"### {T['ranking_title']}")
            st.markdown(T["ranking_desc"])
            ranking_df = participants_summary[[
                'Participant', 'Accuracy_Percentage', 'Age', 'Gender', 'IA_Exposure', 'Participation_Date'
            ]].copy()
            ranking_df = ranking_df.sort_values(by='Accuracy_Percentage', ascending=False).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            ranking_df = ranking_df.reset_index().rename(columns={
                'index': T['col_rank'],
                'Accuracy_Percentage': T['col_accuracy_pct'],
                'Gender': T['col_gender'],
                'IA_Exposure': T['col_exposure'],
                'Participation_Date': T['col_test_date_short']
            })
            st.dataframe(
                ranking_df.style.background_gradient(subset=[T['col_accuracy_pct']], cmap='Greens')
                                .format({T['col_accuracy_pct']: '{:.1f}%'}),
                use_container_width=True, hide_index=True
            )
            st.markdown("---")
            st.markdown(f"### {T['top_bottom_title']}")
            col_m25, col_p25 = st.columns(2)
            top_25_df = participants_summary.nlargest(25, 'Accuracy_Percentage')[[
                'Participant', 'Accuracy_Percentage', 'Age', 'Gender', 'IA_Exposure'
            ]].reset_index(drop=True)
            top_25_df.index = top_25_df.index + 1
            top_25_df = top_25_df.rename(columns={
                'Participant': T['col_name'], 'Accuracy_Percentage': T['col_accuracy_short'],
                'Gender': T['col_gender'], 'IA_Exposure': T['col_exposure_short']
            })
            bottom_25_df = participants_summary.nsmallest(25, 'Accuracy_Percentage')[[
                'Participant', 'Accuracy_Percentage', 'Age', 'Gender', 'IA_Exposure'
            ]].reset_index(drop=True)
            bottom_25_df.index = bottom_25_df.index + 1
            bottom_25_df = bottom_25_df.rename(columns={
                'Participant': T['col_name'], 'Accuracy_Percentage': T['col_accuracy_short'],
                'Gender': T['col_gender'], 'IA_Exposure': T['col_exposure_short']
            })
            with col_m25:
                st.markdown(f"#### {T['top25']}")
                st.dataframe(
                    top_25_df.style.background_gradient(subset=[T['col_accuracy_short']], cmap='Greens')
                                   .format({T['col_accuracy_short']: '{:.1f}%'}),
                    use_container_width=True
                )
            with col_p25:
                st.markdown(f"#### {T['bottom25']}")
                st.dataframe(
                    bottom_25_df.style.background_gradient(subset=[T['col_accuracy_short']], cmap='Reds_r')
                                      .format({T['col_accuracy_short']: '{:.1f}%'}),
                    use_container_width=True
                )
            st.markdown("---")
            st.markdown(f"### {T['evolution_title']}")
            st.markdown(T["evolution_desc"])
            st.pyplot(plot_participant_evolution(df))

    # ----------------------------------------------------
    # TAB 4: CONTENT ANALYSIS
    # ----------------------------------------------------
    with tab_content:
        st.subheader(T["content_subtitle"])
        st.markdown(f"### {T['img_acc_title']}")
        st.pyplot(plot_accuracy_per_image(df))
        st.markdown("---")
        st.markdown(f"### {T['error_rate_title']}")
        st.pyplot(plot_error_ai_vs_real(df))
        st.markdown("---")
        st.markdown(f"### {T['detailed_table_title']}")
        item_stats = df.groupby(['Image', 'Solution']).agg(
            Total_Responses=('Correct', 'count'),
            Correct_Count=('Correct', 'sum'),
            Accuracy_Percentage=('Correct', lambda x: (x.sum() / x.count()) * 100)
        ).reset_index()
        item_stats['Error_Percentage'] = 100 - item_stats['Accuracy_Percentage']
        item_stats = item_stats.sort_values(by='Error_Percentage', ascending=False)
        st.markdown(f"#### {T['ordered_by_confusion']}")
        st.dataframe(
            item_stats.style.background_gradient(subset=['Error_Percentage'], cmap='Reds'),
            use_container_width=True, hide_index=True
        )

    # ----------------------------------------------------
    # TAB 5: THEME ANALYSIS
    # ----------------------------------------------------
    with tab_themes:
        st.subheader(T["themes_subtitle"])
        st.markdown(T["themes_desc"])
        df_theme = df.copy()
        df_theme["Theme"] = df_theme["Image"].apply(assign_theme)
        stats_theme = df_theme.groupby("Theme").agg(
            total=("Correct", "count"),
            correct=("Correct", "sum")
        ).reset_index()
        stats_theme["pct_correct"] = (stats_theme["correct"] / stats_theme["total"] * 100).round(1)
        stats_theme["pct_error"] = (100 - stats_theme["pct_correct"]).round(1)
        stats_theme = stats_theme.sort_values("pct_error", ascending=False).reset_index(drop=True)

        def error_to_color(pct):
            t = max(0.0, min(1.0, pct / 100.0))
            r = int(40 + t * 180)
            g = int(180 - t * 140)
            b = int(60 - t * 40)
            return f"rgb({r},{g},{b})"

        h1, h2, h3, h4 = st.columns([3.2, 4.5, 1.0, 1.0])
        h1.markdown(f"**{T['col_theme']}**")
        h2.markdown("")
        h3.markdown(f"**{T['col_pct_error']}**")
        h4.markdown(f"**{T['col_pct_correct']}**")
        st.markdown("<hr style='margin:4px 0 10px 0; border:none; border-top:1px solid #ddd;'>", unsafe_allow_html=True)

        for _, row in stats_theme.iterrows():
            c1, c2, c3, c4 = st.columns([3.2, 4.5, 1.0, 1.0])
            c1.write(row["Theme"])
            color = error_to_color(row["pct_error"])
            w = max(2.0, min(100.0, float(row["pct_error"])))
            bar_html = f'''
            <div style="background:#e8e8e8; border-radius:5px; height:18px; width:100%; overflow:hidden;">
              <div style="width:{w}%; height:100%; background:{color}; border-radius:5px;"></div>
            </div>
            '''
            c2.markdown(bar_html, unsafe_allow_html=True)
            c3.markdown(f"<div style='text-align:center; font-weight:600;'>{row['pct_error']:.1f}</div>", unsafe_allow_html=True)
            c4.markdown(f"<div style='text-align:center; font-weight:600;'>{row['pct_correct']:.1f}</div>", unsafe_allow_html=True)

        if not stats_theme.empty:
            most_error = stats_theme.iloc[0]
            least_error = stats_theme.iloc[-1]
            st.markdown("")
            st.success(T["theme_most_error"].format(theme=most_error['Theme'], pct=most_error['pct_error']))
            st.info(T["theme_least_error"].format(theme=least_error['Theme'], pct=least_error['pct_error']))

    # ----------------------------------------------------
    # TAB 6: ANALYSED IMAGES
    # ----------------------------------------------------
    with tab_images:
        st.subheader(T["images_subtitle"])
        image_list = sorted(df['Image'].unique().tolist())
        selected_image = st.selectbox(T["select_image"], image_list)
        if selected_image:
            df_img = df[df['Image'] == selected_image]
            true_solution = df_img['Solution'].iloc[0]
            total_votes = len(df_img)
            votes_ai = (df_img['Response'] == 'AI').sum()
            votes_real = (df_img['Response'] == 'REAL').sum()
            pct_ai = (votes_ai / total_votes * 100) if total_votes > 0 else 0
            pct_real = (votes_real / total_votes * 100) if total_votes > 0 else 0
            correct_count = df_img['Correct'].sum()
            accuracy_pct = (correct_count / total_votes * 100) if total_votes > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(T["solution_real"], true_solution)
            m2.metric(T["total_responses"], total_votes)
            m3.metric(T["accuracy_pct"], f"{accuracy_pct:.1f}%")
            m4.metric(T["confusion_rate"], f"{100 - accuracy_pct:.1f}%")

            st.markdown("---")
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown(T["what_people_thought"])
                st.markdown(T["thought_ai"].format(n=votes_ai, pct=pct_ai))
                st.progress(pct_ai / 100)
                st.markdown(T["thought_real"].format(n=votes_real, pct=pct_real))
                st.progress(pct_real / 100)
                st.markdown("---")
                local_path = os.path.join(IMAGES_DIR, selected_image)
                if os.path.exists(local_path):
                    ext = os.path.splitext(selected_image)[1].lower()
                    video_exts = ['.mp4', '.mov', '.avi', '.m4v', '.webm', '.mkv']
                    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff']
                    if ext in video_exts:
                        st.video(local_path)
                        st.caption(f"{T['video_file']} `{selected_image}`")
                    elif ext in image_exts:
                        st.image(local_path, caption=f"{T['image_file']} {selected_image}", use_container_width=True)
                    else:
                        try:
                            st.image(local_path, caption=f"{T['file_label']} {selected_image}", use_container_width=True)
                        except Exception:
                            st.video(local_path)
                            st.caption(f"{T['video_file']} `{selected_image}`")
                else:
                    st.info(T["filename_hint"].format(name=selected_image))

            with col_right:
                st.markdown(T["demographic_breakdown"])
                df_img_merged = df_img.merge(
                    participants_summary[['Participant', 'Age_Group']],
                    on='Participant', how='left'
                )
                st.markdown(T["accuracy_by_age"])
                age_img = df_img_merged.groupby('Age_Group', observed=False)['Correct'].agg(['mean', 'count']).reset_index()
                age_img['Accuracy (%)'] = age_img['mean'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
                st.dataframe(
                    age_img[['Age_Group', 'Accuracy (%)', 'count']].rename(
                        columns={'Age_Group': T['col_age_group'], 'count': T['col_responses']}
                    ),
                    use_container_width=True, hide_index=True
                )
                st.markdown(T["accuracy_by_gender"])
                gen_img = df_img_merged.groupby('Gender')['Correct'].agg(['mean', 'count']).reset_index()
                gen_img['Accuracy (%)'] = gen_img['mean'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
                st.dataframe(
                    gen_img[['Gender', 'Accuracy (%)', 'count']].rename(
                        columns={'Gender': T['col_gender'], 'count': T['col_responses']}
                    ),
                    use_container_width=True, hide_index=True
                )
                st.markdown(T["accuracy_by_exposure"])
                exp_img = df_img_merged.groupby('IA_Exposure')['Correct'].agg(['mean', 'count']).reset_index()
                exp_img['Accuracy (%)'] = exp_img['mean'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
                st.dataframe(
                    exp_img[['IA_Exposure', 'Accuracy (%)', 'count']].rename(
                        columns={'IA_Exposure': T['col_exposure'], 'count': T['col_responses']}
                    ),
                    use_container_width=True, hide_index=True
                )

    # ----------------------------------------------------
    # TAB 7: RAW DATA (ADMIN ONLY)
    # ----------------------------------------------------
    if is_admin and tab_data is not None:
        with tab_data:
            st.subheader(T["raw_subtitle"])
            st.dataframe(df, use_container_width=True)
            csv_buffer = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=T["download_csv"],
                data=csv_buffer,
                file_name=T["export_filename"],
                mime="text/csv"
            )

else:
    st.warning(T["no_data"])