import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from pathlib import Path

st.set_page_config(
    page_title="Chelister Dewelopment",
    page_icon="🏢",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("xgb_model_polish.pkl")
    columns = joblib.load("model_columns_polish.pkl")
    return model, columns

model, model_columns = load_model()

# ── Background ───────────────────────────────────────────────────────────────
def get_bg_base64():
    p = Path("bg.png")
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None

bg     = get_bg_base64()
bg_css = f"url('data:image/png;base64,{bg}')" if bg else "none"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {{ background-color: #0a0a0a; }}

[data-testid="stAppViewContainer"] > .main {{
    background-image: linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.88)), {bg_css};
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    min-height: 100vh;
}}

[data-testid="stSidebar"] {{
    background: rgba(10,10,10,0.96) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}}

* {{ font-family: 'Outfit', sans-serif; color: #f0ede8; }}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {{
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase;
}}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 2px !important;
    color: #ffffff !important;
    font-size: 0.9rem !important;
}}

[data-testid="stButton"] > button {{
    background: #ffffff !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 14px 40px !important;
    width: 100%;
    margin-top: 24px;
    transition: all 0.2s ease;
}}
[data-testid="stButton"] > button:hover {{
    background: #e0ddd8 !important;
    transform: translateY(-1px);
}}
[data-testid="stButton"] > button *{{
    color: #0a0a0a !important;
}}

.kpi-card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-top: 2px solid #ffffff;
    border-radius: 2px;
    padding: 24px 20px 20px;
    margin-top: 24px;
}}
.kpi-label {{
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-bottom: 12px;
}}
.kpi-value {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 400;
    color: #ffffff;
    line-height: 1;
}}
.kpi-range {{
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-top: 6px;
}}
.kpi-sub {{
    font-size: 0.68rem;
    color: rgba(255,255,255,0.3);
    margin-top: 8px;
}}
.disclaimer {{
    font-size: 0.62rem;
    color: rgba(255,255,255,0.2);
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    line-height: 1.6;
}}

#footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language toggle
    lang = st.radio("🌐", ["Polski", "English"], horizontal=True, label_visibility="collapsed")
    pl   = lang == "Polski"

    T = {
        "sec_location":  "Lokalizacja"              if pl else "Location",
        "city":          "Miasto"                    if pl else "City",
        "distance":      "Odległość od centrum (km)" if pl else "Distance to Centre (km)",
        "sec_apt":       "Mieszkanie"                if pl else "Apartment",
        "area":          "Powierzchnia (m²)"         if pl else "Area (m²)",
        "rooms":         "Liczba pokoi"              if pl else "Number of Rooms",
        "floor":         "Piętro"                    if pl else "Floor",
        "floor_count":   "Liczba pięter w budynku"  if pl else "Total Floors in Building",
        "age":           "Wiek budynku (lata)"       if pl else "Building Age (years)",
        "poi":           "Punkty usługowe w pobliżu" if pl else "Points of Interest Nearby",
        "sec_building":  "Budynek"                   if pl else "Building",
        "type":          "Typ budynku"               if pl else "Building Type",
        "material":      "Materiał budynku"          if pl else "Building Material",
        "ownership":     "Forma własności"           if pl else "Ownership Type",
        "sec_condition": "Stan & Udogodnienia"       if pl else "Condition & Amenities",
        "condition":     "Stan mieszkania"            if pl else "Apartment Condition",
        "parking":       "Miejsce parkingowe"        if pl else "Parking Space",
        "balcony":       "Balkon"                    if pl else "Balcony",
        "elevator":      "Winda"                     if pl else "Elevator",
        "security":      "Ochrona"                   if pl else "Security",
        "storage":       "Komórka lokatorska"        if pl else "Storage Room",
        "btn":           "Oblicz Wycenę"             if pl else "Estimate Price",
        "hero_title":    "Inteligentna wycena nieruchomości"
                         if pl else "Intelligent Property Valuation",
        "hero_sub":      "Model oparty na 195\u202f000 transakcjach z polskiego rynku. Wypełnij parametry i kliknij przycisk."
                         if pl else "Model trained on 195,000 Polish real estate transactions. Fill in the parameters and click the button.",
        "kpi1_label":    "Szacowana wartość"         if pl else "Estimated Value",
        "kpi1_sub":      "przedział ufności ±10%"    if pl else "confidence interval ±10%",
        "kpi2_label":    "Wartość dodana remontu"    if pl else "Renovation Value Added",
        "kpi2_range":    "vs. stan niski"            if pl else "vs. low condition",
        "kpi2_sub":      "przyrost wartości"         if pl else "estimated value increase",
        "kpi3_label":    "Cena za m²"                if pl else "Price per m²",
        "kpi3_sub":      "cena jednostkowa"          if pl else "unit price",
        "rooms_unit":    "pok."                      if pl else "rooms",
        "placeholder":   "Wypełnij parametry i kliknij przycisk aby obliczyć wycenę."
                         if pl else "Fill in the parameters and click the button to get an estimate.",
        "disclaimer":    "Wycena ma charakter poglądowy (XGBoost, R²=0.94, dane 2023–2024). Nie stanowi oferty ani operatu szacunkowego. Chelister Dewelopment."
                         if pl else "This estimate is indicative only (XGBoost, R²=0.94, data 2023–2024). Not a formal valuation. Chelister Dewelopment.",
        "currency":      "zł" if pl else "PLN",
        "mln":           "mln zł" if pl else "M PLN",
    }

    st.divider()
    st.caption(T["sec_location"].upper())
    city = st.selectbox(T["city"], [
        'bialystok','bydgoszcz','czestochowa','gdansk','gdynia',
        'katowice','krakow','lodz','lublin','poznan',
        'radom','rzeszow','szczecin','warszawa','wroclaw'
    ], index=13)
    centre_distance = st.number_input(T["distance"], min_value=0.0, max_value=50.0, value=3.0, step=0.1)

    st.divider()
    st.caption(T["sec_apt"].upper())
    square_meters = st.number_input(T["area"],        min_value=10,  max_value=3000,  value=50)
    rooms         = st.number_input(T["rooms"],       min_value=1,   max_value=20,   value=3)
    floor         = st.number_input(T["floor"],       min_value=0,   max_value=50,   value=3)
    floor_count   = st.number_input(T["floor_count"], min_value=1,   max_value=50,   value=8)
    age           = st.number_input(T["age"],         min_value=0,   max_value=200,  value=15)
    poi_count     = st.number_input(T["poi"],         min_value=0,   max_value=100, value=1)

    st.divider()
    st.caption(T["sec_building"].upper())
    building_type = st.selectbox(T["type"],      ['apartmentBuilding','blockOfFlats','tenement','unknown'])
    material      = st.selectbox(T["material"],  ['brick','concreteSlab','unknown'])
    ownership     = st.selectbox(T["ownership"], ['condominium','cooperative','udział'])

    st.divider()
    st.caption(T["sec_condition"].upper())
    condition    = st.selectbox(T["condition"], ['low','premium','unknown'])
    has_parking  = st.checkbox(T["parking"])
    has_balcony  = st.checkbox(T["balcony"])
    has_elevator = st.checkbox(T["elevator"])
    has_security = st.checkbox(T["security"])
    has_storage  = st.checkbox(T["storage"])

    st.divider()
    estimate_btn = st.button(T["btn"])

# ── Main area ─────────────────────────────────────────────────────────────────
col_main, col_pad = st.columns([2, 1])

with col_main:
    # Header with logo
    logo_col, title_col = st.columns([1, 4])
    with logo_col:
        st.markdown("""
        <svg viewBox="0 0 280 240" xmlns="http://www.w3.org/2000/svg" width="110">
        <polygon points="140,5 275,230 5,230"  fill="none" stroke="white" stroke-width="12"/>
        <polygon points="140,55 235,215 45,215" fill="none" stroke="white" stroke-width="6"/>
        <text x="140" y="200" text-anchor="middle" fill="white"
                font-family="Helvetica Neue,sans-serif" font-size="18" font-weight="700"
                letter-spacing="3">CHELISTER</text>
        </svg>
        """, unsafe_allow_html=True)
    with title_col:
        st.markdown(f"""
        <div style="padding-top:8px">
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.8rem;
                        font-weight:600;letter-spacing:0.18em;color:#ffffff;text-transform:uppercase;">
                Chelister
            </div>
            <div style="font-size:0.72rem;letter-spacing:0.35em;
                        color:rgba(255,255,255,0.4);text-transform:uppercase;">
                Dewelopment &nbsp;·&nbsp; {"Kalkulator Wyceny" if pl else "Property Estimator"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:16px 0 28px'>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:8px 0 24px;">
        <div style="font-family:'Cormorant Garamond',serif;font-size:2.4rem;
                    font-weight:300;letter-spacing:0.04em;line-height:1.2;color:#ffffff;">
            {T["hero_title"]}
        </div>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.4);margin-top:12px;
                    letter-spacing:0.06em;max-width:420px;line-height:1.7;">
            {T["hero_sub"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if estimate_btn:
        # ── Build input dataframe ─────────────────────────────────────────────
        input_data = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)

        input_data['squareMeters']   = np.log1p(square_meters)
        input_data['rooms']          = rooms
        input_data['floor']          = np.log1p(floor)
        input_data['floorCount']     = np.log1p(floor_count)
        input_data['age']            = np.log1p(age)
        input_data['centreDistance'] = np.log1p(centre_distance)
        input_data['poiCount']       = np.log1p(poi_count)
        input_data['floorRatio']     = np.log1p(floor) / np.log1p(max(floor_count, 1))
        input_data['amenityScore']   = sum([has_parking, has_balcony, has_elevator,
                                            has_security, has_storage])
        input_data['cityCount']      = 1

        for prefix, value, ref in [
            ('city',             city,          'bialystok'),
            ('type',             building_type, 'apartmentBuilding'),
            ('buildingMaterial', material,      'brick'),
            ('condition',        condition,     'low'),
            ('ownership',        ownership,     'condominium'),
        ]:
            if value != ref:
                col_name = f'{prefix}_{value}'
                if col_name in model_columns:
                    input_data[col_name] = 1

        # ── Predictions ───────────────────────────────────────────────────────
        prediction        = np.expm1(model.predict(input_data)[0])
        input_before      = input_data.copy()
        input_before['condition_premium'] = 0
        input_before['condition_unknown'] = 0
        prediction_before = np.expm1(model.predict(input_before)[0])
        renovation_value  = prediction - prediction_before
        low_est           = prediction * 0.90
        high_est          = prediction * 1.10

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{T["kpi1_label"]}</div>
                <div class="kpi-value">{prediction/1_000_000:.2f}
                    <span style="font-size:1rem;color:rgba(255,255,255,0.5)">{T["mln"]}</span>
                </div>
                <div class="kpi-range">{low_est:,.0f} – {high_est:,.0f} {T["currency"]}</div>
                <div class="kpi-sub">{T["kpi1_sub"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{T["kpi2_label"]}</div>
                <div class="kpi-value">{renovation_value:,.0f}
                    <span style="font-size:1rem;color:rgba(255,255,255,0.5)">{T["currency"]}</span>
                </div>
                <div class="kpi-range">{T["kpi2_range"]}</div>
                <div class="kpi-sub">{T["kpi2_sub"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{T["kpi3_label"]}</div>
                <div class="kpi-value">{prediction/square_meters:,.0f}
                    <span style="font-size:1rem;color:rgba(255,255,255,0.5)">{T["currency"]}</span>
                </div>
                <div class="kpi-range">{square_meters} m² · {rooms} {T["rooms_unit"]}</div>
                <div class="kpi-sub">{T["kpi3_sub"]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="disclaimer">{T["disclaimer"]}</div>',
                    unsafe_allow_html=True)

    else:
        st.info(T["placeholder"])
