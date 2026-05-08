"""
WELYNE — Application complète autonome
Le modèle ML est chargé directement — pas besoin d'API séparée
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Welyne — Bilan Morphologique",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODELE
# ─────────────────────────────────────────────────────────────────
# Modele Ridge integre en base64
MODEL_B64 = "gASVuQUAAAAAAACMEHNrbGVhcm4ucGlwZWxpbmWUjAhQaXBlbGluZZSTlCmBlH2UKIwFc3RlcHOUXZQojAZzY2FsZXKUjBtza2xlYXJuLnByZXByb2Nlc3NpbmcuX2RhdGGUjA5TdGFuZGFyZFNjYWxlcpSTlCmBlH2UKIwJd2l0aF9tZWFulIiMCHdpdGhfc3RklIiMBGNvcHmUiIwObl9mZWF0dXJlc19pbl+USwWMD25fc2FtcGxlc19zZWVuX5SMFm51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAZzY2FsYXKUk5SMBW51bXB5lIwFZHR5cGWUk5SMAmY4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGJDCAAAAAAA9rJAlIaUUpSMBW1lYW5flGgSjAxfcmVjb25zdHJ1Y3SUk5RoFYwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsFhZRoF4wCZjiUiYiHlFKUKEsDaBtOTk5K/////0r/////SwB0lGKJQygCU8GsBW1lQATGOjLV6lNAJazgqkTPPUCHs+6rRoblP8HKu5l89jpAlHSUYowEdmFyX5RoImgkSwCFlGgmh5RSlChLAUsFhZRoLIlDKIMzsU33HVRAgTigC0bcbkAQXE/4C9ZSQN6VJedZL8w/mxw2jWMdMECUdJRijAZzY2FsZV+UaCJoJEsAhZRoJoeUUpQoSwFLBYWUaCyJQyir61lQ2fAhQH/am4fQbC9A4l43qzhcIUAZAwzVLgjePxAGXQ2rDhBAlHSUYowQX3NrbGVhcm5fdmVyc2lvbpSMBTEuOC4wlHVihpSMBW1vZGVslIwTc2tsZWFybi5tdWx0aW91dHB1dJSMFE11bHRpT3V0cHV0UmVncmVzc29ylJOUKYGUfZQojAllc3RpbWF0b3KUjBtza2xlYXJuLmxpbmVhcl9tb2RlbC5fcmlkZ2WUjAVSaWRnZZSTlCmBlH2UKIwFYWxwaGGURz/wAAAAAAAAjA1maXRfaW50ZXJjZXB0lIiMBmNvcHlfWJSIjAhtYXhfaXRlcpROjAN0b2yURz8aNuLrHEMtjAZzb2x2ZXKUjARhdXRvlIwIcG9zaXRpdmWUiYwMcmFuZG9tX3N0YXRllE5oPmg/dWKMBm5fam9ic5ROjAtlc3RpbWF0b3JzX5RdlChoSimBlH2UKGhNRz/wAAAAAAAAaE6IaE+IaFBOaFFHPxo24uscQy1oUmhTaFSJaFVOaBBLBYwFY29lZl+UaCJoJEsAhZRoJoeUUpQoSwFLBYWUaCyJQygAcRgDWiDxP3ialceSPwxAFHjSP4PP9j+er2J6OCzovwaadvqGFxtAlHSUYowHbl9pdGVyX5ROjAdzb2x2ZXJflIwIY2hvbGVza3mUjAppbnRlcmNlcHRflGgUaBpDCGRjBFD52lZAlIaUUpRoPmg/dWJoSimBlH2UKGhNRz/wAAAAAAAAaE6IaE+IaFBOaFFHPxo24uscQy1oUmhTaFSJaFVOaBBLBWhbaCJoJEsAhZRoJoeUUpQoSwFLBYWUaCyJQyiRDd+vtKwQQNcziHaQvva/5IVCsKo41L8mgaI7QqIQwATii7aIByBAlHSUYmhiTmhjaGRoZWgUaBpDCJg35JKrfVlAlIaUUpRoPmg/dWJoSimBlH2UKGhNRz/wAAAAAAAAaE6IaE+IaFBOaFFHPxo24uscQy1oUmhTaFSJaFVOaBBLBWhbaCJoJEsAhZRoJoeUUpQoSwFLBYWUaCyJQyjdpNEWyFiHv+kkkAmKsJM/Ia4w7G2JkT/fc33VFg+dPztc2JGiAJE/lHSUYmhiTmhjaGRoZWgUaBpDCMeaLMLjouw/lIaUUpRoPmg/dWJlaBBLBWg+aD91YoaUZYwPdHJhbnNmb3JtX2lucHV0lE6MBm1lbW9yeZROjAd2ZXJib3NllIloPmg/dWIu"

@st.cache_resource
def charger_modele():
    """Charge le modele integre en base64."""
    try:
        import base64, pickle
        model_bytes = base64.b64decode(MODEL_B64)
        model = pickle.loads(model_bytes)
        return model, True
    except Exception as e:
        return None, False

model, model_ok = charger_modele()

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: #FAF8F3 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem !important; max-width: 1100px !important; margin: 0 auto !important; }
section[data-testid="stSidebar"] { display: none; }

.hero {
    background: linear-gradient(135deg, #0A4A3A 0%, #0E6655 60%, #0D5C4A 100%);
    padding: 3rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%; right: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(201,168,76,0.1) 0%, transparent 70%);
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #C9A84C; margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 300;
    color: white; line-height: 1.1;
    margin-bottom: 0.8rem;
}
.hero-title em { font-style: italic; color: #F0D080; }
.hero-subtitle { font-size: 0.95rem; color: rgba(255,255,255,0.7); }
.hero-stats {
    display: flex; gap: 2.5rem;
    margin-top: 2rem;
}
.hero-stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem; font-weight: 600;
    color: #F0D080; line-height: 1;
}
.hero-stat-label { font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 0.2rem; }
.hero-divider { width: 1px; background: rgba(255,255,255,0.15); align-self: stretch; }

.section-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #C9A84C; margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem; font-weight: 500;
    color: #0A4A3A; margin-bottom: 1.5rem;
}

.imc-preview {
    background: linear-gradient(135deg, #F0FDF8, #E6F7F3);
    border: 1px solid rgba(14,102,85,0.15);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin: 1rem 0;
}
.imc-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem; font-weight: 600; color: #0A4A3A; line-height: 1;
}
.imc-label { font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.1em; }
.imc-cat { font-size: 0.95rem; font-weight: 600; color: #0A4A3A; }

.stButton > button {
    background: linear-gradient(135deg, #0A4A3A, #0E6655) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(10,74,58,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(10,74,58,0.4) !important;
}

.risk-card {
    border-radius: 20px; padding: 2.5rem;
    text-align: center; margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
.risk-card.low    { background: linear-gradient(135deg, #0A4A3A, #0E6655); }
.risk-card.medium { background: linear-gradient(135deg, #7D4C00, #D68910); }
.risk-card.high   { background: linear-gradient(135deg, #7B1A1A, #C0392B); }
.risk-eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(255,255,255,0.6); }
.risk-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem; font-weight: 300; color: white; line-height: 1;
}
.risk-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15); color: white;
    padding: 0.4rem 1.2rem; border-radius: 30px;
    font-size: 0.9rem; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.2);
    margin-top: 0.5rem;
}

.metric-card {
    background: white; border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #F3F4F6;
    margin-bottom: 1rem;
    transition: transform 0.2s;
    position: relative; overflow: hidden;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    border-radius: 4px 0 0 4px;
}
.metric-card.ok::before     { background: linear-gradient(180deg, #1ABC9C, #0E6655); }
.metric-card.warn::before   { background: linear-gradient(180deg, #F39C12, #D68910); }
.metric-card.danger::before { background: linear-gradient(180deg, #E74C3C, #C0392B); }
.metric-icon   { font-size: 1.3rem; margin-bottom: 0.6rem; display: block; }
.metric-label  { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #9CA3AF; }
.metric-value  {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem; font-weight: 600; color: #1F2937; line-height: 1.1;
}
.metric-unit   { font-size: 0.9rem; color: #9CA3AF; font-family: 'DM Sans', sans-serif; }
.metric-badge  {
    display: inline-block; margin-top: 0.5rem;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600;
}
.badge-ok     { background: #D1FAE5; color: #065F46; }
.badge-warn   { background: #FEF3C7; color: #92400E; }
.badge-danger { background: #FEE2E2; color: #991B1B; }

.profil-card {
    background: linear-gradient(135deg, #0A4A3A, #0E6655);
    border-radius: 16px; padding: 1.5rem 2rem;
    margin-bottom: 1.5rem; color: white;
    display: flex; justify-content: space-between; align-items: center;
}
.profil-name { font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; }
.profil-sub  { font-size: 0.82rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem; }
.profil-time { font-size: 0.75rem; color: rgba(255,255,255,0.4); text-align: right; }

.reco-item {
    display: flex; align-items: flex-start; gap: 1rem;
    padding: 1.1rem 1.4rem; background: white;
    border-radius: 12px; margin-bottom: 0.7rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #F3F4F6;
    transition: transform 0.15s;
}
.reco-item:hover { transform: translateX(4px); }
.reco-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
.reco-dot.ok     { background: #1ABC9C; }
.reco-dot.warn   { background: #F39C12; }
.reco-dot.danger { background: #E74C3C; }
.reco-text { font-size: 0.9rem; line-height: 1.6; color: #4B5563; }

.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9A84C, transparent);
    margin: 2rem 0; opacity: 0.35;
}
.medical-warn {
    background: #FFFBEB; border: 1px solid rgba(201,168,76,0.3);
    border-radius: 10px; padding: 0.9rem 1.3rem;
    font-size: 0.82rem; color: #78350F; margin-top: 2rem;
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    color: #4B5563 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TRADUCTIONS
# ─────────────────────────────────────────────────────────────────
T = {
    "fr": {
        "eyebrow":"Plateforme de santé intelligente",
        "title":"Votre bilan\n<em>morphologique</em>\npersonnalisé",
        "subtitle":"Estimez vos indicateurs de santé en quelques secondes à partir de 4 données simples. Basé sur 6 068 mesures réelles.",
        "form_label":"Votre profil", "form_title":"Renseignez vos informations",
        "taille":"Taille (cm)", "poids":"Poids (kg)", "age":"Âge (années)",
        "sexe":"Sexe", "homme":"Homme", "femme":"Femme",
        "imc_calc":"IMC calculé",
        "btn":"Analyser mon profil",
        "calib_title":"Avez-vous vos vraies mesures ? (optionnel)",
        "calib_waist":"Mon vrai tour de taille (cm)",
        "calib_hip":"Mon vrai tour de hanches (cm)",
        "calib_badge":"Calibre avec vos mesures reelles",
        "loading":"Analyse en cours...",
        "res_label":"Vos résultats", "res_title":"Analyse complète",
        "score_label":"Score de risque cardio-métabolique",
        "risk_low":"Risque faible", "risk_med":"Risque modéré", "risk_high":"Risque élevé",
        "indicators":"Indicateurs de santé",
        "gauges_label":"Analyse visuelle", "gauges_title":"Ce que ça signifie pour vous",
        "recos_label":"Recommandations", "recos_title":"Conseils personnalisés",
        "hist_label":"Historique", "hist_title":"Évolution de vos indicateurs",
        "clear":"Effacer l'historique", "new":"← Nouvelle analyse",
        "normal":"Normal", "caution":"Attention", "high":"Élevé", "very_high":"Très élevé",
        "warning":"Cet outil est une aide à la prévention. Il ne remplace pas un avis médical.",
        "n_imc":"Indice de Masse Corporelle",
        "n_waist":"Tour de taille estimé",
        "n_hip":"Tour de hanches estimé",
        "n_whr":"Rapport Taille / Hanches (WHR)",
        "n_bf":"Masse grasse estimée",
        "n_score":"Score de risque",
        "imc_cats":{
            "Sous-poids":"Votre IMC indique un sous-poids. Un accompagnement nutritionnel est conseillé.",
            "Poids normal":"Votre IMC est dans la zone idéale. Continuez sur cette lancée !",
            "Surpoids":"Votre IMC indique un léger surpoids. Une activité physique régulière est bénéfique.",
            "Obesite grade I":"Votre IMC indique une obésité modérée. Un suivi médical est recommandé.",
            "Obesite grade II/III":"Votre IMC indique une obésité sévère. Consultez un médecin rapidement.",
        },
        "waist_cats":{
            "ok":"Votre tour de taille est dans la zone normale — bon signe pour votre santé cardiovasculaire.",
            "warn":"Votre tour de taille est légèrement au-dessus du seuil OMS. Réduire les graisses abdominales serait bénéfique.",
            "danger":"Votre tour de taille dépasse le seuil OMS élevé. Une consultation médicale est fortement conseillée.",
        },
        "whr_cats":{
            "ok":"Votre WHR est dans la norme. La répartition de vos graisses est favorable.",
            "warn":"Votre WHR dépasse le seuil OMS. Les graisses s'accumulent autour du ventre — facteur de risque cardiovasculaire.",
        },
        "global_cats":{
            "Faible":"Votre profil global est rassurant. Maintenir un mode de vie actif et équilibré est la clé.",
            "Modere":"Votre profil présente quelques signaux d'attention. De petits ajustements de mode de vie peuvent faire une grande différence.",
            "Eleve":"Votre profil présente un risque élevé. Une consultation médicale est fortement recommandée.",
        },
    },
    "en": {
        "eyebrow":"Intelligent health platform",
        "title":"Your personalized\n<em>morphological</em>\nassessment",
        "subtitle":"Estimate your health indicators in seconds from 4 simple inputs. Based on 6,068 real measurements.",
        "form_label":"Your profile", "form_title":"Enter your information",
        "taille":"Height (cm)", "poids":"Weight (kg)", "age":"Age (years)",
        "sexe":"Sex", "homme":"Male", "femme":"Female",
        "imc_calc":"Calculated BMI",
        "btn":"Analyze my profile",
        "calib_title":"Do you have your real measurements? (optional)",
        "calib_waist":"My real waist circumference (cm)",
        "calib_hip":"My real hip circumference (cm)",
        "calib_badge":"Calibrated with your real measurements",
        "loading":"Analyzing...",
        "res_label":"Your results", "res_title":"Full analysis",
        "score_label":"Cardio-metabolic risk score",
        "risk_low":"Low risk", "risk_med":"Moderate risk", "risk_high":"High risk",
        "indicators":"Health indicators",
        "gauges_label":"Visual analysis", "gauges_title":"What this means for you",
        "recos_label":"Recommendations", "recos_title":"Personalized advice",
        "hist_label":"History", "hist_title":"Evolution of your indicators",
        "clear":"Clear history", "new":"← New analysis",
        "normal":"Normal", "caution":"Caution", "high":"High", "very_high":"Very high",
        "warning":"This tool is a prevention aid. It does not replace professional medical advice.",
        "n_imc":"Body Mass Index",
        "n_waist":"Estimated waist circumference",
        "n_hip":"Estimated hip circumference",
        "n_whr":"Waist-to-Hip Ratio (WHR)",
        "n_bf":"Estimated body fat",
        "n_score":"Risk score",
        "imc_cats":{
            "Sous-poids":"Your BMI indicates underweight. Nutritional support is advised.",
            "Poids normal":"Your BMI is in the ideal range. Keep up the great work!",
            "Surpoids":"Your BMI indicates slight overweight. Regular physical activity is beneficial.",
            "Obesite grade I":"Your BMI indicates moderate obesity. Medical follow-up is recommended.",
            "Obesite grade II/III":"Your BMI indicates severe obesity. Please consult a doctor promptly.",
        },
        "waist_cats":{
            "ok":"Your waist circumference is in the normal range — a good sign for cardiovascular health.",
            "warn":"Your waist circumference is slightly above the WHO threshold. Reducing abdominal fat would be beneficial.",
            "danger":"Your waist circumference exceeds the WHO high-risk threshold. Medical consultation is strongly advised.",
        },
        "whr_cats":{
            "ok":"Your WHR is within normal range. Your fat distribution is favorable.",
            "warn":"Your WHR exceeds the WHO threshold. Fat is accumulating around the abdomen — a cardiovascular risk factor.",
        },
        "global_cats":{
            "Faible":"Your overall profile is reassuring. Maintaining an active, balanced lifestyle is key.",
            "Modere":"Your profile shows some warning signals. Small lifestyle adjustments can make a big difference.",
            "Eleve":"Your profile shows elevated risk. Medical consultation is strongly recommended.",
        },
    }
}

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
for k, v in [('hist', []), ('res', None), ('lang', 'fr'), ('sexe', 'male'), ('profil', '')]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# LOGIQUE MEDICALE
# ─────────────────────────────────────────────────────────────────
def predire(height, weight, age, gender, waist_reel=None, hip_reel=None):
    sex  = 1 if gender == "male" else 0
    bmi  = round(weight / (height/100)**2, 2)
    feat = np.array([[height, weight, age, sex, bmi]])
    pred = model.predict(feat)[0]
    waist_p = round(float(pred[0]), 1)
    hip_p   = round(float(pred[1]), 1)
    calibre = False
    if waist_reel and hip_reel:
        waist, hip, calibre = round(waist_reel,1), round(hip_reel,1), True
    elif waist_reel:
        ecart = waist_reel - waist_p
        waist, hip, calibre = round(waist_reel,1), round(hip_p + ecart*0.6, 1), True
    elif hip_reel:
        ecart = hip_reel - hip_p
        waist, hip, calibre = round(waist_p + ecart*0.6,1), round(hip_reel,1), True
    else:
        waist, hip = waist_p, hip_p
    whr   = round(waist / hip, 3) if hip > 0 else 0.0
    bf    = round(max(3.0, min(60.0, (1.20*bmi) + (0.23*age) - (10.8*sex) - 5.4)), 1)

    # Score de risque
    if sex == 1:
        st = 0.0 if waist<94 else (0.3+(waist-94)/8*0.4 if waist<102 else min(1.0,0.7+(waist-102)/30*0.3))
        sw = 0.90; sbn=19.0; sbo=25.0
    else:
        st = 0.0 if waist<80 else (0.3+(waist-80)/8*0.4 if waist<88 else min(1.0,0.7+(waist-88)/25*0.3))
        sw = 0.85; sbn=32.0; sbo=38.0

    sw_s = 0.0 if whr<sw else min(1.0,(whr-sw)/0.20)
    sbf  = (0.0 if bf<=sbn else (bf-sbn)/(sbo-sbn)*0.6 if bf<=sbo else min(1.0,0.6+(bf-sbo)/20*0.4))
    score = round((0.40*st + 0.40*sw_s + 0.20*sbf)*100, 1)

    def imc_cat(b):
        if b<18.5: return "Sous-poids"
        if b<25:   return "Poids normal"
        if b<30:   return "Surpoids"
        if b<35:   return "Obesite grade I"
        return "Obesite grade II/III"

    return {
        "BMI": bmi, "waist": waist, "hip": hip, "whr": whr,
        "bf": bf, "score": score, "sex": sex,
        "calibre": calibre if 'calibre' in dir() else False,
        "imc_cat": imc_cat(bmi),
        "waist_st": "ok" if (waist<94 if sex==1 else waist<80) else ("warn" if (waist<102 if sex==1 else waist<88) else "danger"),
        "whr_st"  : "ok" if whr < sw else "warn",
        "risk_lv" : "Faible" if score<30 else "Modere" if score<60 else "Eleve",
    }

def gauge(val, mn, mx, s1, s2, title, unit, h=190):
    c = "#1ABC9C" if val<s1 else "#F39C12" if val<s2 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={'text':title,'font':{'size':12,'color':'#4B5563','family':'DM Sans'}},
        number={'suffix':f' {unit}','font':{'size':20,'color':c,'family':'Cormorant Garamond, serif'}},
        gauge={
            'axis':{'range':[mn,mx],'tickfont':{'size':9},'tickcolor':'#9CA3AF'},
            'bar':{'color':c,'thickness':0.22}, 'bgcolor':'white','borderwidth':0,
            'steps':[
                {'range':[mn,s1],'color':'#D1FAE5'},
                {'range':[s1,s2],'color':'#FEF3C7'},
                {'range':[s2,mx],'color':'#FEE2E2'},
            ],
        }
    ))
    fig.update_layout(height=h, margin=dict(t=50,b=10,l=20,r=20),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def mc(icon, label, val, unit, badge_txt, badge_cls, card_cls):
    return f"""
<div class="metric-card {card_cls}">
    <span class="metric-icon">{icon}</span>
    <div class="metric-label">{label}</div>
    <div class="metric-value">{val}<span class="metric-unit"> {unit}</span></div>
    <span class="metric-badge badge-{badge_cls}">{badge_txt}</span>
</div>"""

# ─────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────
nc1, nc2, nc3 = st.columns([3, 5, 2])
with nc1:
    st.markdown('<div style="padding:0.5rem 0;font-family:Cormorant Garamond,serif;font-size:1.5rem;font-weight:600;color:#0A4A3A;">W<span style="color:#C9A84C">e</span>lyne</div>', unsafe_allow_html=True)
with nc3:
    lang = st.selectbox("", ["Français","English"],
                        index=0 if st.session_state.lang=='fr' else 1,
                        label_visibility="collapsed")
    st.session_state.lang = 'fr' if lang=="Français" else 'en'

t = T[st.session_state.lang]

# ─────────────────────────────────────────────────────────────────
# ERREUR MODELE
# ─────────────────────────────────────────────────────────────────
if not model_ok:
    st.error("Modèle non trouvé. Assurez-vous que welyne_model_ridge.joblib est dans le même dossier.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# PAGE FORMULAIRE
# ─────────────────────────────────────────────────────────────────
if st.session_state.res is None:

    title_html = t['title'].replace('\n','<br>').replace('<em>','<em style="font-style:italic;color:#F0D080;">')
    st.markdown(f"""
    <div class="hero">
        <div style="max-width:700px;margin:0 auto;">
            <div class="hero-eyebrow">✦ {t['eyebrow']}</div>
            <div class="hero-title">{title_html}</div>
            <p class="hero-subtitle">{t['subtitle']}</p>
            <div class="hero-stats">
                <div><div class="hero-stat-num">6 068</div><div class="hero-stat-label">{'mesures d\'entraînement' if st.session_state.lang=='fr' else 'training measurements'}</div></div>
                <div class="hero-divider"></div>
                <div><div class="hero-stat-num">86%</div><div class="hero-stat-label">{'précision du modèle' if st.session_state.lang=='fr' else 'model accuracy'}</div></div>
                <div class="hero-divider"></div>
                <div><div class="hero-stat-num">6</div><div class="hero-stat-label">{'indicateurs calculés' if st.session_state.lang=='fr' else 'indicators calculated'}</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{t["form_label"]}</div><div class="section-title">{t["form_title"]}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        taille = st.slider(t['taille'], 140, 210, 170, 1)
        poids  = st.slider(t['poids'],  40,  150,  70, 1)
    with c2:
        age    = st.slider(t['age'], 15, 80, 30, 1)
        sexe_l = st.radio(t['sexe'], [t['homme'], t['femme']], horizontal=True)
        sexe   = "male" if sexe_l == t['homme'] else "female"

    bmi_p = round(poids/(taille/100)**2, 1)
    def imc_cat_p(b):
        if b<18.5: return "Sous-poids"
        if b<25:   return "Poids normal"
        if b<30:   return "Surpoids"
        if b<35:   return "Obésité I"
        return "Obésité II/III"
    bmi_c = imc_cat_p(bmi_p)
    bmi_col = "#0E6655" if bmi_p<25 else "#D68910" if bmi_p<30 else "#C0392B"

    st.markdown(f"""
    <div class="imc-preview">
        <div>
            <div class="imc-label">{t['imc_calc']}</div>
            <div class="imc-num" style="color:{bmi_col};">{bmi_p}</div>
        </div>
        <div style="width:1px;background:#E5E7EB;margin:0 0.5rem;align-self:stretch;"></div>
        <div class="imc-cat" style="color:{bmi_col};">{bmi_c}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section calibration optionnelle
    use_calib = st.checkbox(f"📏  {t['calib_title']}", value=False)
    waist_reel = None
    hip_reel   = None
    if use_calib:
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            w_in = st.number_input(t['calib_waist'], min_value=50.0, max_value=200.0,
                                   value=None, step=0.5, placeholder="Ex: 88.0")
            if w_in: waist_reel = float(w_in)
        with cc2:
            h_in = st.number_input(t['calib_hip'], min_value=60.0, max_value=200.0,
                                   value=None, step=0.5, placeholder="Ex: 98.0")
            if h_in: hip_reel = float(h_in)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(f"🔍  {t['btn']}"):
        with st.spinner(t['loading']):
            res = predire(taille, poids, age, sexe, waist_reel, hip_reel)
        entry = {
            "date"  : datetime.now().strftime("%d/%m %H:%M"),
            "profil": f"{taille}cm/{poids}kg/{age}ans/{sexe_l}",
            **{k: res[k] for k in ['BMI','waist','hip','whr','bf','score','risk_lv']}
        }
        st.session_state.hist.append(entry)
        st.session_state.res    = res
        st.session_state.sexe   = sexe
        st.session_state.profil = f"{taille} cm  •  {poids} kg  •  {age} ans  •  {sexe_l}"
        st.rerun()

    st.markdown(f'<div class="medical-warn">⚕️ {t["warning"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE RESULTATS
# ─────────────────────────────────────────────────────────────────
else:
    res   = st.session_state.res
    sexe  = st.session_state.sexe
    profil= st.session_state.profil

    if st.button(t['new']):
        st.session_state.res = None
        st.rerun()

    # Profil
    st.markdown(f"""
    <div class="profil-card">
        <div>
            <div class="profil-name">{profil}</div>
            <div class="profil-sub">Welyne — Analyse morphologique</div>
        </div>
        <div class="profil-time">{datetime.now().strftime("%d %B %Y, %H:%M")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Score
    score = res['score']
    rlv   = res['risk_lv']
    rcls  = "low" if rlv=="Faible" else "medium" if rlv=="Modere" else "high"
    rlbl  = t['risk_low'] if rlv=="Faible" else t['risk_med'] if rlv=="Modere" else t['risk_high']

    st.markdown(f"""
    <div class="risk-card {rcls}">
        <div class="risk-eyebrow">{t['score_label']}</div>
        <div class="risk-number">{score:.0f}<span style="font-size:1.5rem;">/100</span></div>
        <span class="risk-badge">{rlbl}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Indicateurs
    st.markdown(f'<div class="section-label">{t["res_label"]}</div><div class="section-title">{t["indicators"]}</div>', unsafe_allow_html=True)

    bmi   = res['BMI']
    waist = res['waist']
    hip   = res['hip']
    whr   = res['whr']
    bf    = res['bf']
    w_st  = res['waist_st']
    whr_st= res['whr_st']
    bf_ok = 19 if sexe=="male" else 32
    bf_warn=25 if sexe=="male" else 38
    bf_st = "ok" if bf<bf_ok else "warn" if bf<bf_warn else "danger"

    imc_cls = {"Sous-poids":"warn","Poids normal":"ok","Surpoids":"warn",
               "Obesite grade I":"danger","Obesite grade II/III":"danger"}.get(res['imc_cat'],"ok")
    b2c = {"ok":"ok","warn":"warn","danger":"danger"}
    b2t_fr = {"ok":t['normal'],"warn":t['caution'],"danger":t['very_high']}

    c1,c2,c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(mc("⚖️",t['n_imc'],f"{bmi:.1f}","",res['imc_cat'],imc_cls,imc_cls), unsafe_allow_html=True)
    with c2:
        st.markdown(mc("📏",t['n_waist'],f"{waist:.1f}","cm",b2t_fr[w_st],w_st,w_st), unsafe_allow_html=True)
    with c3:
        st.markdown(mc("📐",t['n_whr'],f"{whr:.3f}","",t['normal'] if whr_st=="ok" else t['high'],whr_st,whr_st), unsafe_allow_html=True)

    c4,c5,c6 = st.columns(3, gap="medium")
    with c4:
        st.markdown(mc("🦴",t['n_hip'],f"{hip:.1f}","cm",t['normal'],"ok","ok"), unsafe_allow_html=True)
    with c5:
        st.markdown(mc("🔬",t['n_bf'],f"{bf:.1f}","%",b2t_fr[bf_st],bf_st,bf_st), unsafe_allow_html=True)
    with c6:
        st.markdown(mc("❤️",t['n_score'],f"{score:.0f}","/100",rlbl,
                       "ok" if rcls=="low" else "warn" if rcls=="medium" else "danger",
                       "ok" if rcls=="low" else "warn" if rcls=="medium" else "danger"), unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Jauges
    st.markdown(f'<div class="section-label">{t["gauges_label"]}</div><div class="section-title">{t["gauges_title"]}</div>', unsafe_allow_html=True)

    s_t1 = 94  if sexe=="male" else 80
    s_t2 = 102 if sexe=="male" else 88
    s_whr= 0.90 if sexe=="male" else 0.85

    j1,j2,j3 = st.columns(3, gap="medium")
    with j1:
        st.plotly_chart(gauge(waist,50,145,s_t1,s_t2,t['n_waist'],"cm"), use_container_width=True, config={'displayModeBar':False})
    with j2:
        st.plotly_chart(gauge(whr,0.55,1.30,s_whr,s_whr+0.12,t['n_whr'],""), use_container_width=True, config={'displayModeBar':False})
    with j3:
        st.plotly_chart(gauge(bf,0,60,bf_ok,bf_warn,t['n_bf'],"%"), use_container_width=True, config={'displayModeBar':False})

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Recommandations
    st.markdown(f'<div class="section-label">{t["recos_label"]}</div><div class="section-title">{t["recos_title"]}</div>', unsafe_allow_html=True)

    recos = [
        (t['imc_cats'].get(res['imc_cat'],""), imc_cls),
        (t['waist_cats'].get(w_st,""),          w_st),
        (t['whr_cats'].get(whr_st,""),           whr_st),
        (t['global_cats'].get(rlv,""),           "ok" if rlv=="Faible" else "warn" if rlv=="Modere" else "danger"),
    ]
    for txt, cls in recos:
        if txt:
            st.markdown(f'<div class="reco-item"><div class="reco-dot {cls}"></div><div class="reco-text">{txt}</div></div>', unsafe_allow_html=True)

    # Historique
    if len(st.session_state.hist) > 1:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{t["hist_label"]}</div><div class="section-title">{t["hist_title"]}</div>', unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.hist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['waist'],
            name=t['n_waist'],
            line=dict(color='#0E6655',width=3),
            mode='lines+markers',
            marker=dict(size=10,color='#0E6655',line=dict(color='white',width=2))
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['score'],
            name=t['n_score'],
            line=dict(color='#C9A84C',width=3,dash='dot'),
            mode='lines+markers',
            marker=dict(size=10,color='#C9A84C',line=dict(color='white',width=2)),
            yaxis='y2'
        ))
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            height=320, margin=dict(t=40,b=50,l=60,r=70),
            yaxis=dict(
                title='Tour de taille (cm)',
                title_font=dict(color='#0E6655',size=12,family='DM Sans'),
                tickfont=dict(color='#0E6655',size=11),
                gridcolor='#EEEEEE', showline=True, linecolor='#CCCCCC',
            ),
            yaxis2=dict(
                title='Score (/100)',
                title_font=dict(color='#C9A84C',size=12,family='DM Sans'),
                tickfont=dict(color='#C9A84C',size=11),
                overlaying='y', side='right', range=[0,100],
                showgrid=False, showline=True, linecolor='#CCCCCC',
            ),
            xaxis=dict(tickfont=dict(color='#333333',size=11),gridcolor='#EEEEEE'),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='center', x=0.5,
                bgcolor='white', bordercolor='#DDDDDD', borderwidth=1,
                font=dict(size=12,color='#333333',family='DM Sans'),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button(f"🗑  {t['clear']}"):
            st.session_state.hist = []
            st.rerun()

    st.markdown(f'<div class="medical-warn">⚕️ {t["warning"]}</div>', unsafe_allow_html=True)
