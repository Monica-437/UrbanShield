"""
UrbanShield — app.py (FULL MODIFIED FRONTEND + LIVE DECISION-SUPPORT LAYER)

IMPORTANT:
- The original dashboard structure, CSS, charts, maps, navigation style and
  existing image carousel are preserved.
- Existing image paths are NOT rejected: get_b64() tries the original paths
  first and then assets/images/.
- The original analytical pages remain available.
- New functionality adds:
    1. Live Operations / New Incident
    2. Database-backed incoming incidents
    3. Dynamic risk result display
    4. Explainable risk reasons
    5. Early-warning display
    6. Resource Priority / fixed-unit recommendation
- Historical data is still treated as historical data. Incoming incidents
  are the changing operational layer.
- Resource recommendations are decision support only and do not automatically
  dispatch or command law-enforcement resources.

This file is intentionally kept as one complete app.py so it can replace the
previous frontend file without deleting its existing dashboard work.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import database as db
import numpy as np
import base64
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# ── UrbanShield live-processing backend ─────────────────────────────────────
# These imports add the new decision-support layer without replacing the
# existing dashboard, charts, maps, CSS, or image-based frontend.
try:
    import database as db
    import pipeline as live_pipeline
    import risk_engine as live_risk_engine
    URBANSHIELD_BACKEND_OK = True
except Exception:
    db = None
    live_pipeline = None
    live_risk_engine = None
    URBANSHIELD_BACKEND_OK = False

from datetime import date

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

MLXTEND_OK = False
# ══════════════════════════════════════════
# DATABASE INITIALIZATION
# ══════════════════════════════════════════
db.init_db()

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
st.set_page_config(page_title="UrbanShield", layout="wide",
                   page_icon="🔍", initial_sidebar_state="expanded")

def get_b64(fp):
    """Read an image from the old location first, then the new project assets.

    The old paths are intentionally preserved so the previous frontend keeps
    working. The assets fallback allows the same images to be moved into:
        UrbanShield/assets/images/
    """
    candidates = [fp]

    try:
        old_name = str(fp).replace("\\", "/").split("/")[-1]
        candidates.append(str(__file__ and
                             (Path(__file__).resolve().parent / "assets" / "images" / old_name)))
    except Exception:
        pass

    # Also try the exact basename in assets/images using several common
    # extensions because Windows paths copied from the old project can be
    # awkwardly named.
    try:
        base = Path(str(fp)).name
        assets_dir = Path(__file__).resolve().parent / "assets" / "images"
        candidates.extend([
            str(assets_dir / base),
            str(assets_dir / Path(base).stem),
        ])
    except Exception:
        pass

    for candidate in candidates:
        try:
            p = Path(candidate)
            if p.is_file():
                raw = p.read_bytes()
                return base64.b64encode(raw).decode()
        except Exception:
            continue
    return ""


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "assets" / "images"
splash_bg = get_b64(str(IMAGE_DIR / "download (5).jpeg"))
inner_bg = get_b64(str(IMAGE_DIR / "Detective Scene.jpeg"))


CRIME_IMGS = [
    IMAGE_DIR / "img1.jpeg",
    IMAGE_DIR / "The Hidden World of Cybercrime_ The Dark Web.jpeg",
    IMAGE_DIR / "img2.jpeg",
    IMAGE_DIR / "thieves tried to steal the store’s cash.jpeg",
    IMAGE_DIR / "img5.jpeg",
    IMAGE_DIR / "img6.jpeg",
]

crime_imgs_b64 = [get_b64(str(p)) for p in CRIME_IMGS]
PREDEFINED_USERS = {
    st.secrets["auth"]["username"]: st.secrets["auth"]["password"]
}
MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

def authenticate(u, p): return PREDEFINED_USERS.get(u) == p

# ══════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════
def inject_css(b64, opacity=0.62):
    bg = (f".stApp{{background-image:url('data:image/jpeg;base64,{b64}');background-size:cover;"
          f"background-position:center;background-repeat:no-repeat;background-attachment:fixed;}}"
          if b64 else ".stApp{background:#030811;}")
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Exo+2:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
{bg}
.stApp::before{{content:"";position:fixed;top:0;left:0;right:0;bottom:0;
    background:rgba(0,0,0,{opacity});z-index:0;}}

*,p,div,span,label{{font-family:'Exo 2',sans-serif !important;font-size:17px !important;color:white !important;}}
h1{{font-size:72px !important;font-weight:700 !important;color:white !important;
    text-shadow:0 0 40px rgba(0,200,255,0.5),3px 3px 12px black;
    font-family:'Rajdhani',sans-serif !important;letter-spacing:4px;}}
h2{{font-size:28px !important;font-weight:700 !important;color:white !important;
    text-shadow:2px 2px 8px black;letter-spacing:1px;}}
h3{{font-size:19px !important;font-weight:600 !important;color:white !important;}}
h4,h5,h6{{font-size:17px !important;color:white !important;font-weight:600 !important;}}

[data-testid="stSidebar"]{{background:rgba(2,6,18,0.96) !important;
    border-right:1px solid rgba(0,200,255,0.18) !important;}}
[data-testid="stSidebar"] *{{color:white !important;font-size:15px !important;}}

.stTextInput>label,.stSelectbox>label,.stSlider>label,
.stRadio>label,.stCheckbox>label{{font-size:13px !important;font-weight:700 !important;
    color:rgba(0,200,255,0.85) !important;letter-spacing:0.8px;text-transform:uppercase;}}
.stSelectbox [data-baseweb="select"]{{background:rgba(0,10,30,0.75) !important;
    border:1px solid rgba(0,200,255,0.3) !important;border-radius:6px !important;}}

.stTabs [data-baseweb="tab"]{{font-size:14px !important;font-weight:600 !important;
    color:rgba(255,255,255,0.55) !important;padding:10px 18px !important;}}
.stTabs [data-baseweb="tab-list"]{{background:rgba(0,10,25,0.65) !important;
    border-radius:10px;gap:3px;border:1px solid rgba(0,200,255,0.14) !important;}}
.stTabs [aria-selected="true"]{{background:rgba(0,200,255,0.13) !important;
    color:#00C8FF !important;border-bottom:2px solid #00C8FF !important;
    border-radius:8px 8px 0 0 !important;}}

.stButton>button{{background:linear-gradient(135deg,rgba(0,200,255,0.13),rgba(0,100,200,0.09)) !important;
    color:white !important;font-weight:700 !important;font-size:15px !important;
    border:1px solid rgba(0,200,255,0.4) !important;border-radius:8px !important;
    padding:11px 20px !important;width:100% !important;transition:all 0.2s ease !important;}}
.stButton>button:hover{{background:linear-gradient(135deg,rgba(0,200,255,0.28),rgba(0,100,200,0.18)) !important;
    border-color:#00C8FF !important;transform:scale(1.02) !important;
    box-shadow:0 0 16px rgba(0,200,255,0.25) !important;}}

.intel-card{{background:linear-gradient(135deg,rgba(0,8,24,0.82),rgba(0,18,45,0.55));
    border:1px solid rgba(0,200,255,0.22);border-radius:14px;padding:18px 20px;
    margin-bottom:14px;box-shadow:0 4px 24px rgba(0,0,0,0.45);backdrop-filter:blur(12px);}}
.intel-red{{background:linear-gradient(135deg,rgba(226,75,74,0.18),rgba(0,8,24,0.75));
    border:1px solid rgba(226,75,74,0.5);border-radius:14px;padding:18px 20px;margin-bottom:14px;}}
.intel-green{{background:linear-gradient(135deg,rgba(29,158,117,0.18),rgba(0,8,24,0.75));
    border:1px solid rgba(29,158,117,0.5);border-radius:14px;padding:18px 20px;margin-bottom:14px;}}
.intel-amber{{background:linear-gradient(135deg,rgba(239,159,39,0.18),rgba(0,8,24,0.75));
    border:1px solid rgba(239,159,39,0.5);border-radius:14px;padding:18px 20px;margin-bottom:14px;}}
.intel-cyan{{background:linear-gradient(135deg,rgba(0,200,255,0.11),rgba(0,8,24,0.75));
    border:1px solid rgba(0,200,255,0.38);border-radius:14px;padding:18px 20px;margin-bottom:14px;}}

.alert-crit{{background:rgba(226,75,74,0.2);border-left:5px solid #E24B4A;
    border-radius:0 10px 10px 0;padding:14px 20px;margin-bottom:10px;}}
.alert-warn{{background:rgba(239,159,39,0.17);border-left:5px solid #EF9F27;
    border-radius:0 10px 10px 0;padding:14px 20px;margin-bottom:10px;}}
.alert-info{{background:rgba(0,200,255,0.11);border-left:5px solid #00C8FF;
    border-radius:0 10px 10px 0;padding:14px 20px;margin-bottom:10px;}}
.alert-green{{background:rgba(29,158,117,0.17);border-left:5px solid #1D9E75;
    border-radius:0 10px 10px 0;padding:14px 20px;margin-bottom:10px;}}

.mcard{{background:linear-gradient(135deg,rgba(0,8,24,0.85),rgba(0,28,58,0.5));
    border:1px solid rgba(0,200,255,0.3);border-radius:13px;padding:16px 12px;text-align:center;}}
.mcard-num{{font-size:30px !important;font-weight:700 !important;color:#00C8FF !important;
    display:block;font-family:'Rajdhani',sans-serif !important;
    text-shadow:0 0 16px rgba(0,200,255,0.45);}}
.mcard-lbl{{font-size:12px !important;color:#88aacc !important;display:block;
    margin-top:4px;letter-spacing:0.9px;text-transform:uppercase;}}

.insight-box{{background:linear-gradient(135deg,rgba(0,200,255,0.07),rgba(0,8,24,0.6));
    border:1px solid rgba(0,200,255,0.28);border-radius:10px;padding:14px 18px;margin:10px 0;}}
.insight-box p{{font-size:15px !important;color:#d4eeff !important;margin:0;line-height:1.75;}}

.warn-box{{background:linear-gradient(135deg,rgba(239,159,39,0.1),rgba(0,8,24,0.6));
    border:1px solid rgba(239,159,39,0.35);border-radius:10px;padding:14px 18px;margin:10px 0;}}
.warn-box p{{font-size:15px !important;color:#ffe0a0 !important;margin:0;line-height:1.75;}}

.red-box{{background:linear-gradient(135deg,rgba(226,75,74,0.1),rgba(0,8,24,0.6));
    border:1px solid rgba(226,75,74,0.35);border-radius:10px;padding:14px 18px;margin:10px 0;}}
.red-box p{{font-size:15px !important;color:#ffcccc !important;margin:0;line-height:1.75;}}

.green-box{{background:linear-gradient(135deg,rgba(29,158,117,0.1),rgba(0,8,24,0.6));
    border:1px solid rgba(29,158,117,0.35);border-radius:10px;padding:14px 18px;margin:10px 0;}}
.green-box p{{font-size:15px !important;color:#aaffdd !important;margin:0;line-height:1.75;}}

.sec-hdr{{background:linear-gradient(90deg,rgba(0,200,255,0.1),transparent);
    border-left:3px solid #00C8FF;border-radius:0 6px 6px 0;padding:8px 16px;margin-bottom:14px;}}
.sec-hdr p{{font-size:14px !important;color:#00C8FF !important;margin:0;
    letter-spacing:1.2px;text-transform:uppercase;font-weight:700 !important;}}

.briefing{{background:linear-gradient(135deg,rgba(0,8,24,0.88),rgba(0,30,65,0.6));
    border:1px solid rgba(0,200,255,0.3);border-radius:14px;padding:22px 26px;
    margin-bottom:16px;line-height:2.0;}}
.briefing p{{font-size:17px !important;color:#e0f4ff !important;margin:0;}}

.action-card{{background:linear-gradient(135deg,rgba(0,8,24,0.75),rgba(0,25,55,0.45));
    border-left:3px solid #00C8FF;border-radius:0 12px 12px 0;
    padding:14px 18px;margin-bottom:10px;transition:transform 0.2s;}}
.action-card:hover{{transform:translateX(4px);}}
.action-card h4{{font-size:16px !important;color:#00C8FF !important;margin:0 0 5px;}}
.action-card p{{font-size:14px !important;color:#aaccdd !important;margin:0;line-height:1.7;}}

.ldot{{display:inline-block;width:8px;height:8px;border-radius:50%;
    background:#1D9E75;margin-right:6px;animation:blink 1.6s infinite;}}
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0.2;}}}}

.ftag{{display:inline-block;background:rgba(0,200,255,0.12);border:1px solid rgba(0,200,255,0.35);
    border-radius:20px;padding:4px 14px;font-size:13px !important;color:#00C8FF !important;
    font-weight:700 !important;margin:0 4px 4px 0;letter-spacing:0.5px;}}

[data-testid="stDataFrame"]{{border-radius:10px;overflow:hidden;}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════
def pcfg(h=280):
    return {"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,5,15,0.38)",
            "font_color":"white","font":{"size":13,"family":"Exo 2"},
            "margin":{"t":38,"b":16,"l":16,"r":16},"height":h}

def insight(txt, kind="blue"):
    cls = {"blue":"insight-box","warn":"warn-box","red":"red-box","green":"green-box"}.get(kind,"insight-box")
    icon = {"blue":"🔍","warn":"⚠️","red":"🚨","green":"✅"}.get(kind,"🔍")
    st.markdown(f"<div class='{cls}'><p>{icon} {txt}</p></div>", unsafe_allow_html=True)

def sec(title):
    st.markdown(f"<div class='sec-hdr'><p>{title}</p></div>", unsafe_allow_html=True)

def filter_tag(crime, district, time_val, count):
    st.markdown(
        f"<div style='margin-bottom:14px;'>"
        f"<span class='ldot'></span>"
        f"<span class='ftag'>{crime}</span>"
        f"<span class='ftag'>{district}</span>"
        f"<span class='ftag'>{time_val}</span>"
        f"<span style='font-size:13px !important;color:#aaa !important;margin-left:6px;'>"
        f"{count:,} matching incidents</span></div>",
        unsafe_allow_html=True)
# ══════════════════════════════════════════ 
#  SESSION STATE 
# ══════════════════════════════════════════ 
DEFAULTS = {
    "page": "splash",
    "logged_in": False,
    "username": "",
    "slide_idx": 0,
    "g_crime": "Theft",
    "g_district": "Central",
    "g_time": "Evening",
    "g_weapon": "Unknown",
    "g_result": None,
    "live_notice": None,
    "live_result": None
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
# ══════════════════════════════════════════
#  DATA + MODELS
# ══════════════════════════════════════════
@st.cache_data
def load_and_train():
    # New project location first; original location remains a fallback.
    project_csv = Path(__file__).resolve().parent / "data" / "crime_analysis_final_v4.csv"
    legacy_csv = Path(r"D:\chrome\DA\crime_analysis_final_v4.csv")

    if project_csv.is_file():
        csv_path = project_csv
    elif legacy_csv.is_file():
        csv_path = legacy_csv
    else:
        raise FileNotFoundError(
            "crime_analysis_final_v4.csv was not found. Put it in "
            "UrbanShield/data/ or restore the original dataset path."
        )

    df = pd.read_csv(csv_path)
    df = df.ffill()
    df = df.copy()

    if 'month' in df.columns:
        df['month'] = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)

    encoders = {}
    for col in ["crime_type","crime_category","crime_severity","district","time_category","season",
                "day_of_week","weapon_used","gang_related","arrest_made","weather","priority_level",
                "domestic_related","hate_crime","premises_type","victim_gender","suspect_gender",
                "neighborhood","reported_by","is_holiday","evidence_collected","video_footage"]:
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── CART (gang prediction) ────────────────────────────────────────────────
    gf = ["crime_type_enc","district_enc","time_category_enc","weapon_used_enc"]
    X = df[gf]; y = df["gang_related"]
    Xt,Xe,yt,ye = train_test_split(X, y, test_size=0.2, random_state=42)
    gang_clf = DecisionTreeClassifier(max_depth=15, random_state=42)
    gang_clf.fit(Xt, yt)
    gang_acc = round(accuracy_score(ye, gang_clf.predict(Xe))*100, 2)

    # ── Arrest prediction (GaussianNB) ────────────────────────────────────────
    af = ["crime_type_enc","district_enc","time_category_enc","weapon_used_enc",
          "crime_severity_enc","gang_related_enc","evidence_collected_enc"]
    Xa = df[af]; ya = df["arrest_made"]
    Xat,Xae,yat,yae = train_test_split(Xa, ya, test_size=0.2, random_state=42)
    arrest_clf = GaussianNB()
    arrest_clf.fit(Xat, yat)
    arrest_acc = round(accuracy_score(yae, arrest_clf.predict(Xae))*100, 2)

    # ── KNN (crime similarity matching, k=5) ─────────────────────────────────
    knn_feats = ["crime_type_enc","district_enc","time_category_enc",
                 "weapon_used_enc","crime_severity_enc"]
    Xk = df[knn_feats]; yk = df["arrest_made"]
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(Xk, yk)

    # ── KMeans (crime zone clustering, k=5) ───────────────────────────────────
    coord_cols = ["latitude","longitude","crime_type_enc","crime_severity_enc"]
    coord_df = df[coord_cols].dropna()
    sc_km = StandardScaler()
    km_X  = sc_km.fit_transform(coord_df)
    kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_model.fit(km_X)
    coord_df = coord_df.copy()
    coord_df["zone_cluster"] = kmeans_model.labels_
    df = df.copy()
    df.loc[coord_df.index, "zone_cluster"] = coord_df["zone_cluster"]
    df["zone_cluster"] = df["zone_cluster"].fillna(-1).astype(int)

    # ── Agglomerative clustering (district crime grouping) ────────────────────
    dist_feat = df.groupby("district")[["crime_type_enc","crime_severity_enc",
                                        "gang_related_enc","weapon_used_enc"]].mean()
    agg_model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    dist_feat = dist_feat.copy()
    dist_feat["agg_group"] = agg_model.fit_predict(dist_feat)

    # ── Loss regression ───────────────────────────────────────────────────────
    lf = ["crime_type_enc","crime_severity_enc","district_enc","weapon_used_enc",
          "gang_related_enc","victim_count","suspect_count"]
    Xr = df[lf].dropna(); yr = df.loc[Xr.index,"estimated_loss"]
    Xrt,Xre,yrt,yre = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    lscaler = StandardScaler(); lreg = LinearRegression()
    lreg.fit(lscaler.fit_transform(Xrt), yrt)

    # ── Pre-computed aggregations ─────────────────────────────────────────────
    loss_stats       = df.groupby("crime_type")["estimated_loss"].agg(["mean","min","max","std"]).round(2)
    month_crime      = df.groupby(["month","crime_type"]).size().reset_index(name="Count")
    dist_month       = df.groupby(["district","month"]).size().unstack(fill_value=0)
    crime_month_pivot= df.groupby(["month","crime_type"]).size().unstack(fill_value=0)

    return (df, encoders,
            gang_clf, gang_acc, gf,
            arrest_clf, arrest_acc, af,
            knn_clf, knn_feats,
            lreg, lscaler, lf,
            loss_stats, month_crime, dist_month, crime_month_pivot,
            dist_feat, None)


(df, encoders,
 gang_clf, gang_acc, gang_feats,
 arrest_clf, arrest_acc, arrest_feats,
 knn_clf, knn_feats,
 lreg, lscaler, loss_feats,
 loss_stats, month_crime, dist_month, crime_month_pivot,
 dist_feat_agg, apriori_rules) = load_and_train()

def enc(col, val):
    try: return int(encoders[col].transform([val])[0])
    except: return 0

# ══════════════════════════════════════════
#  INTELLIGENCE ENGINE
# ══════════════════════════════════════════
def run_analysis(crime, district, time_of_day, weapon):
    inp = pd.DataFrame([[enc("crime_type",crime), enc("district",district),
                         enc("time_category",time_of_day), enc("weapon_used",weapon)]],
                       columns=gang_feats)
    gang_pred  = gang_clf.predict(inp)[0]
    gang_proba = gang_clf.predict_proba(inp)[0]
    gang_conf  = round(max(gang_proba)*100, 1)

    fdf     = df[(df["crime_type"]==crime)&(df["district"]==district)&(df["time_category"]==time_of_day)].copy()
    area_df = df[(df["district"]==district)&(df["time_category"]==time_of_day)].copy()
    crime_freq = len(fdf); area_freq = max(len(area_df), 1)

    risk_score = round(min((crime_freq/area_freq)*100*3, 100), 1)
    if gang_pred=="Yes":              risk_score = min(risk_score+20, 100)
    if weapon in ["Firearm","Knife"]: risk_score = min(risk_score+15, 100)
    risk_level = ("CRITICAL" if risk_score>=70 else "HIGH" if risk_score>=45
                  else "MODERATE" if risk_score>=25 else "LOW")

    if crime in loss_stats.index:
        ls   = loss_stats.loc[crime]
        base = round(ls["mean"], 2)
        lmin = round(max(0, ls["min"]), 2)
        lmax = round(ls["max"], 2)
    else:
        base = 0.0; lmin = 0.0; lmax = 0.0

    resolved_count = int(fdf["arrest_made"].value_counts().get("Yes", 0)) if crime_freq > 0 else 0
    arrest_rate    = round(fdf["arrest_made"].value_counts(normalize=True).get("Yes", 0)*100, 1) if crime_freq > 0 else 0.0
    arrest_display = f"{resolved_count}/{crime_freq} resolved ({arrest_rate}%)"

    city_arrest = round(df["arrest_made"].value_counts(normalize=True).get("Yes", 0)*100, 1)
    arrest_gap  = round(arrest_rate - city_arrest, 1)

    # Predicted arrest probability from similar incidents
    knn_inp = pd.DataFrame([[enc("crime_type",crime), enc("district",district),
                              enc("time_category",time_of_day), enc("weapon_used",weapon),
                              enc("crime_severity", fdf["crime_severity"].mode()[0] if crime_freq>0 else "Misdemeanor")]],
                           columns=knn_feats)
    knn_proba = knn_clf.predict_proba(knn_inp)[0]
    knn_arrest_prob = round(knn_proba[list(knn_clf.classes_).index("Yes")]*100, 1) if "Yes" in knn_clf.classes_ else 0.0

    gang_label = "Gang Involved" if gang_pred=="Yes" else "Gang Not Involved"
    gang_txt   = (f"{gang_label} — Cross-reference gang database immediately." if gang_pred=="Yes"
                  else f"{gang_label} — Standard response applies.")
    risk_txt   = {"CRITICAL":"CRITICAL threat level. Deploy rapid response unit immediately.",
                  "HIGH":"HIGH risk. Increase patrol presence and escalate to senior officer.",
                  "MODERATE":"MODERATE risk. Step up surveillance for 48 hours.",
                  "LOW":"LOW risk. Standard protocol applies."}.get(risk_level,"")
    gap_txt    = f"Resolution rate is {abs(arrest_gap)}% {'above' if arrest_gap>0 else 'below'} city average ({city_arrest}%)."

    top_weapon_in_area = fdf["weapon_used"].mode()[0] if crime_freq>0 else weapon
    peak_time_in_area  = area_df["time_category"].mode()[0] if len(area_df)>0 else time_of_day

    city_crime_counts = df.groupby("crime_type").size()
    dist_crime_counts = df[df["district"]==district].groupby("crime_type").size()
    dist_share = (dist_crime_counts / city_crime_counts.reindex(dist_crime_counts.index, fill_value=1) * 100).round(1)
    top_threat     = dist_share.idxmax() if len(dist_share)>0 else crime
    top_threat_pct = dist_share.max()   if len(dist_share)>0 else 0

    return {
        "crime":crime,"district":district,"time":time_of_day,"weapon":weapon,
        "gang_pred":gang_pred,"gang_conf":gang_conf,
        "gang_label":gang_label,
        "gang_proba":dict(zip(gang_clf.classes_, gang_proba)),
        "risk_score":risk_score,"risk_level":risk_level,
        "loss_min":lmin,"loss_mean":base,"loss_max":lmax,
        "similar_count":crime_freq,
        "resolved_count":resolved_count,
        "arrest_rate":arrest_rate,
        "arrest_display":arrest_display,
        "knn_arrest_prob":knn_arrest_prob,
        "city_arrest":city_arrest,"arrest_gap":arrest_gap,
        "gang_txt":gang_txt,"risk_txt":risk_txt,"gap_txt":gap_txt,
        "top_weapon_in_area":top_weapon_in_area,"peak_time_in_area":peak_time_in_area,
        "top_threat":top_threat,"top_threat_pct":top_threat_pct,
        "fdf":fdf,"area_df":area_df,
        "city_crime":df[df["crime_type"]==crime].copy(),
        "city_dist":df[df["district"]==district].copy()
    }

# ══════════════════════════════════════════
#  AUTO-CONCLUSION ENGINE
# ══════════════════════════════════════════
def auto_conclude(data, chart_type, context={}):
    if chart_type == "crime_by_district":
        top = data.idxmax(); val = data.max(); avg = data.mean()
        pct = round((val-avg)/avg*100)
        return f"<b>{top}</b> district records the highest incidents — <b>{int(val)}</b> cases, which is <b>{pct}% above</b> the city average of {int(avg)}. Priority resource deployment recommended."
    elif chart_type == "time_pattern":
        top = data.idxmax(); val = data.max(); low = data.idxmin()
        return f"<b>{top}</b> is the highest-risk time period with <b>{int(val)}</b> incidents. <b>{low}</b> sees the lowest activity. Patrol density should be highest during {top} shifts."
    elif chart_type == "weapon_dist":
        top = data.idxmax(); pct = round(data[top]/data.sum()*100)
        return f"<b>{top}</b> is the most common weapon type, involved in <b>{pct}%</b> of incidents. Armed response readiness should reflect this pattern."
    elif chart_type == "arrest_by_day":
        top = data.idxmax(); val = round(data[top], 1)
        low = data.idxmin(); lval = round(data[low], 1)
        return f"<b>{top}</b> has the highest case resolution rate at <b>{val}%</b>. <b>{low}</b> performs lowest at {lval}% — investigate resource allocation on this day."
    elif chart_type == "seasonal":
        top = data.idxmax(); val = data.max(); low = data.idxmin()
        return f"Crime peaks in <b>{top}</b> season ({int(val)} incidents). <b>{low}</b> is the safest season. Seasonal resource planning should account for this <b>{round((val-data[low])/data[low]*100)}% variance</b>."
    elif chart_type == "loss_by_crime":
        top = data.idxmax(); val = data.max()
        return f"<b>{top}</b> causes the highest average financial damage at <b>${val:,.0f}</b> per incident. Insurance and victim support resources should be prioritised for this crime type."
    elif chart_type == "evidence_resolution":
        if "with_evidence" in context and "without_evidence" in context:
            diff = round(context["with_evidence"] - context["without_evidence"], 1)
            return f"Cases with physical evidence collected are resolved at <b>{context['with_evidence']}%</b> vs only <b>{context['without_evidence']}%</b> without — a <b>{diff}% gap</b>. Evidence collection training is the single highest-impact intervention."
    elif chart_type == "victim_profile":
        return f"The most vulnerable group is <b>{context.get('top_gender','Male')}</b>, aged <b>{context.get('top_age','25-40')}</b>. Targeted victim protection should focus on this demographic."
    elif chart_type == "monthly_trend":
        top = context.get("peak","N/A"); low = context.get("trough","N/A")
        val = context.get("peak_val",0); lval = context.get("trough_val",0)
        return f"Crime peaks in <b>{top}</b> ({val} incidents) and is lowest in <b>{low}</b> ({lval} incidents) — a <b>{round((val-lval)/max(lval,1)*100)}% seasonal swing</b>. Pre-emptive deployment before {top} is strongly recommended."
    elif chart_type == "gang_by_district":
        top = data.idxmax(); val = data.max()
        city_total = data.sum()
        pct = round(val/city_total*100)
        return f"<b>{top}</b> accounts for <b>{pct}%</b> of all gang-related incidents in the city ({int(val)} cases). Anti-gang task force resources should be concentrated here."
    elif chart_type == "hotspot":
        return f"Crime zone <b>{context.get('zone',0)}</b> in <b>{context.get('district','N/A')}</b> is the highest-density area with <b>{context.get('count',0)}</b> incidents. This zone requires dedicated patrol assignment."
    return ""


# ══════════════════════════════════════════
#  URBANSHIELD LIVE DECISION-SUPPORT HELPERS
# ══════════════════════════════════════════

def _backend_ready():
    return bool(URBANSHIELD_BACKEND_OK and db is not None and live_pipeline is not None)


def _risk_color(level):
    return {
        "CRITICAL": "#E24B4A",
        "HIGH": "#EF9F27",
        "MODERATE": "#FAC775",
        "LOW": "#1D9E75",
    }.get(str(level).upper(), "#00C8FF")


def _safe_backend_call(payload):
    """Call the project's pipeline while supporting both pipeline API styles.

    Preferred final API:
        pipeline.process_new_incident(payload, source="streamlit")

    Compatibility fallback:
        pipeline.process_incident(...)
    """
    if not _backend_ready():
        return {
            "success": False,
            "errors": [
                "UrbanShield backend modules are not available. "
                "Check database.py and pipeline.py."
            ],
        }

    # Preferred UrbanShield pipeline.
    fn = getattr(live_pipeline, "process_new_incident", None)
    if callable(fn):
        try:
            return fn(payload, source="streamlit")
        except TypeError:
            return fn(payload)

    # Compatibility with the earlier pipeline signature.
    fn = getattr(live_pipeline, "process_incident", None)
    if callable(fn):
        try:
            coverage_bounds = utils.compute_coverage_bounds(df)

            baseline_df = live_risk_engine.compute_historical_baseline(df)

            # If the compatibility pipeline needs an anomaly model, use the
            # project's anomaly module when available.
            try:
                import anomaly_detection as _ad
                anomaly_model, anomaly_encoders = _ad.train_anomaly_model(df)
            except Exception:
                anomaly_model, anomaly_encoders = None, None

            return fn(
                payload,
                df,
                coverage_bounds,
                baseline_df,
                anomaly_model,
                anomaly_encoders,
            )
        except Exception as exc:
            return {"success": False, "errors": [str(exc)]}

    return {
        "success": False,
        "errors": ["No supported incident-processing function was found in pipeline.py."],
    }


def _fetch_recent_live_incidents(limit=50):
    if not _backend_ready():
        return pd.DataFrame()

    try:
        rows = db.fetch_incidents(limit=limit)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        try:
            with db.get_session() as session:
                rows = session.query(db.Incident).order_by(
                    db.Incident.submitted_at.desc()
                ).limit(limit).all()
                data = []
                for r in rows:
                    data.append({
                        "incident_id": r.incident_id,
                        "crime_type": r.crime_type,
                        "crime_severity": r.crime_severity,
                        "district": r.district,
                        "neighborhood": r.neighborhood,
                        "latitude": r.latitude,
                        "longitude": r.longitude,
                        "occurred_date": r.occurred_date,
                        "occurred_time": r.occurred_time,
                        "weapon_used": r.weapon_used,
                        "priority_level": r.priority_level,
                        "source": r.source,
                        "status": r.status,
                        "submitted_at": r.submitted_at,
                    })
                return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()


def _fetch_live_alerts(limit=20):
    if not _backend_ready():
        return []

    try:
        return db.fetch_alerts(limit=limit)
    except Exception:
        try:
            with db.get_session() as session:
                rows = session.query(db.Alert).order_by(
                    db.Alert.created_at.desc()
                ).limit(limit).all()
                return [{
                    "alert_id": r.alert_id,
                    "incident_id": r.incident_id,
                    "location": r.location,
                    "alert_type": r.alert_type,
                    "risk_level": r.risk_level,
                    "message": r.message,
                    "recommendation": r.recommendation,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                } for r in rows]
        except Exception:
            return []


def _extract_result_risk(result):
    assessment = result.get("assessment", {}) if isinstance(result, dict) else {}
    risk = assessment.get("dynamic_risk", result.get("dynamic_risk", 0))
    level = assessment.get("risk_level", result.get("risk_level", "LOW"))
    return float(risk or 0), str(level or "LOW")


def _historical_area_scores():
    """Return district historical scores using the project's risk engine."""
    if live_risk_engine is None:
        return pd.DataFrame()

    try:
        base = live_risk_engine.compute_historical_baseline(df)
        if len(base):
            return base.copy()
    except Exception:
        pass

    # Safe visual fallback only if the backend engine cannot be evaluated.
    rows = []
    for district, sub in df.groupby("district"):
        rows.append({
            "district": district,
            "incident_count": len(sub),
            "historical_risk": round(
                min(100.0, len(sub) / max(len(df), 1) * 700), 1
            ),
        })
    return pd.DataFrame(rows)


def _resource_allocation(total_units=10):
    """Decision-support allocation for a fixed pool of patrol units.

    This is a recommendation only. It does not automatically dispatch units.
    The allocation is derived from historical district risk plus recent
    incoming-incident activity.
    """
    base = _historical_area_scores()
    if base.empty:
        return pd.DataFrame()

    recent = _fetch_recent_live_incidents(limit=2000)
    recent_counts = (
        recent.groupby("district").size()
        if not recent.empty and "district" in recent.columns
        else pd.Series(dtype=float)
    )

    rows = []
    for _, r in base.iterrows():
        district = r["district"]
        hist = float(r.get("historical_risk", 0) or 0)
        incoming = float(recent_counts.get(district, 0))
        score = hist + min(incoming * 2.0, 20.0)
        rows.append({
            "Zone": district,
            "Risk Score": round(score, 1),
            "Risk": (
                "CRITICAL" if score >= 75 else
                "HIGH" if score >= 55 else
                "MODERATE" if score >= 35 else
                "LOW"
            ),
            "Incoming Incidents": int(incoming),
        })

    alloc = pd.DataFrame(rows).sort_values(
        ["Risk Score", "Incoming Incidents"], ascending=False
    ).reset_index(drop=True)

    if alloc.empty:
        return alloc

    # Start with one unit per area, then distribute remaining units by score.
    n = len(alloc)
    units = [1] * n
    remaining = max(0, int(total_units) - n)

    for idx in alloc.index:
        if remaining <= 0:
            break
        # Highest-risk zones receive extra units first.
        extra = min(
            remaining,
            max(0, int(round(float(alloc.loc[idx, "Risk Score"]) / 25)) - 1)
        )
        units[idx] += extra
        remaining -= extra

    # If units remain, distribute one-by-one from highest to lowest score.
    idx_pos = 0
    while remaining > 0 and n > 0:
        units[alloc.index[idx_pos % n]] += 1
        remaining -= 1
        idx_pos += 1

    alloc["Recommended Units"] = units
    alloc["Decision Support Note"] = (
        "Recommended priority only — an authorized decision-maker must approve deployment."
    )
    return alloc


def _render_live_result(result):
    """Render the result of the most recent incoming incident."""
    if not result:
        return

    if not result.get("success", False):
        errors = result.get("errors", result.get("detail", ["Processing failed."]))
        if not isinstance(errors, list):
            errors = [str(errors)]
        st.markdown(
            "<div class='red-box'><p>🚨 <b>Incident was not processed</b></p>"
            f"<p>{'<br>'.join(map(str, errors))}</p></div>",
            unsafe_allow_html=True,
        )
        return

    incident = result.get("incident", {})
    risk, level = _extract_result_risk(result)
    color = _risk_color(level)

    st.markdown(
        f"""<div class='intel-card' style='border-color:{color};'>
            <h3 style='color:{color} !important;'>LIVE INCIDENT PROCESSED</h3>
            <p><b>Incident:</b> {incident.get('incident_id', 'N/A')}</p>
            <p><b>Area:</b> {incident.get('district', 'N/A')} /
               {incident.get('neighborhood', 'N/A')}</p>
            <p><b>Crime:</b> {incident.get('crime_type', 'N/A')} |
               <b>Severity:</b> {incident.get('crime_severity', 'N/A')}</p>
            <p><b>Dynamic Risk:</b>
               <span style='color:{color};font-weight:800;font-size:26px !important;'>
               {risk:.1f}/100 — {level}</span></p>
        </div>""",
        unsafe_allow_html=True,
    )

    reasons = result.get("reasons", [])
    if not reasons:
        assessment = result.get("assessment", {})
        raw = assessment.get("reasons", "")
        reasons = raw.split(" | ") if raw else []

    if reasons:
        st.markdown("<div class='intel-cyan'><h3>Why did the risk change?</h3>",
                    unsafe_allow_html=True)
        for reason in reasons[:8]:
            st.markdown(f"• {reason}")
        st.markdown("</div>", unsafe_allow_html=True)

    priority = result.get("priority", {})
    if priority:
        st.markdown(
            f"""<div class='action-card'>
                <h4>Recommended Decision-Support Priority</h4>
                <p><b>{priority.get('recommended_focus', 'Monitoring')}</b><br>
                {priority.get('action', 'Review the incident and current area risk.')}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    alert = result.get("alert")
    if alert:
        alert_color = "#E24B4A" if level == "CRITICAL" else "#EF9F27"
        st.markdown(
            f"""<div class='alert-crit' style='border-left-color:{alert_color};'>
                <b>⚠️ EARLY WARNING</b><br>
                {alert.get('message', 'Elevated risk detected.')}<br>
                <b>Recommendation:</b> {alert.get('recommendation', '')}
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════
#  PAGE — LIVE OPERATIONS
# ══════════════════════════════════════════
if st.session_state.page == "live":
    st.markdown("<h2>Live Operations</h2>", unsafe_allow_html=True)

    st.markdown(
        """<div class='briefing'><p>
        <b style='color:#00C8FF;'>HISTORICAL + INCOMING DATA LAYER</b><br>
        Historical incidents remain the learning foundation. New incidents
        entered through this interface are stored in the database and processed
        through the UrbanShield risk, anomaly, hotspot and alert pipeline.
        </p></div>""",
        unsafe_allow_html=True,
    )

    if not _backend_ready():
        st.markdown(
            """<div class='red-box'><p>⚠️ <b>Backend connection not ready.</b><br>
            Keep database.py, pipeline.py, risk_engine.py and the other backend
            files in the same UrbanShield folder, then restart Streamlit.</p></div>""",
            unsafe_allow_html=True,
        )
    else:
        sec("Submit New Incident")

        c1, c2, c3 = st.columns(3)
        with c1:
            live_crime = st.selectbox(
                "Crime Type",
                sorted(df["crime_type"].dropna().unique().tolist()),
                key="live_crime",
            )
            live_category_default = (
                df.loc[df["crime_type"] == live_crime, "crime_category"].mode()[0]
                if "crime_category" in df.columns and
                len(df.loc[df["crime_type"] == live_crime]) > 0
                else ""
            )
            live_category = st.text_input(
                "Crime Category",
                value=str(live_category_default),
                key="live_category",
            )

        with c2:
            severity_values = ["Infraction", "Misdemeanor", "Felony"]
            live_severity = st.selectbox(
                "Severity",
                severity_values,
                index=1,
                key="live_severity",
            )
            live_district = st.selectbox(
                "District",
                sorted(df["district"].dropna().unique().tolist()),
                key="live_district",
            )

        with c3:
            live_date = st.date_input("Incident Date", value=date.today(), key="live_date")
            live_time = st.time_input("Incident Time", key="live_time")

        c4, c5, c6 = st.columns(3)
        with c4:
            live_neighborhood = st.text_input("Neighborhood", key="live_neighborhood")
            live_lat = st.number_input(
                "Latitude",
                value=float(df["latitude"].median()),
                format="%.6f",
                key="live_lat",
            )

        with c5:
            live_lon = st.number_input(
                "Longitude",
                value=float(df["longitude"].median()),
                format="%.6f",
                key="live_lon",
            )
            live_weapon = st.selectbox(
                "Weapon",
                ["Unknown", "None", "Firearm", "Knife"] +
                sorted([
                    str(x) for x in df["weapon_used"].dropna().unique()
                    if str(x) not in {"Unknown", "None", "Firearm", "Knife"}
                ]),
                key="live_weapon",
            )

        with c6:
            live_domestic = st.selectbox("Domestic Related", ["No", "Yes"], key="live_domestic")
            live_gang = st.selectbox("Gang Related", ["No", "Yes"], key="live_gang")
            live_damage = st.selectbox("Property Damage", ["No", "Yes"], key="live_damage")

        live_loss = st.number_input(
            "Estimated Loss",
            min_value=0,
            value=0,
            step=100,
            key="live_loss",
        )
        live_priority = st.selectbox(
            "Reported Priority",
            ["Low", "Medium", "High"],
            key="live_priority",
        )

        st.markdown(
            "<p style='font-size:12px !important;color:#8faec0 !important;'>"
            "This interface is for authorized operational decision support. "
            "Recommendations do not automatically dispatch personnel.</p>",
            unsafe_allow_html=True,
        )

        if st.button("SUBMIT INCIDENT", key="submit_live_incident"):
            payload = {
                "crime_type": live_crime,
                "crime_category": live_category,
                "crime_severity": live_severity,
                "district": live_district,
                "neighborhood": live_neighborhood or None,
                "latitude": float(live_lat),
                "longitude": float(live_lon),
                "occurred_date": str(live_date),
                "occurred_time": live_time.strftime("%H:%M"),
                "weapon_used": live_weapon,
                "domestic_related": live_domestic,
                "gang_related": live_gang,
                "property_damage": live_damage,
                "estimated_loss": int(live_loss),
                "priority_level": live_priority,
                "source": "streamlit",
            }

            result = _safe_backend_call(payload)
            st.session_state.live_result = result

            if result.get("success"):
                incident = result.get("incident", {})
                st.session_state.live_incident_id = incident.get("incident_id")
                st.session_state.live_notice = "Incident stored and processed successfully."
            else:
                st.session_state.live_notice = "Incident processing returned an error."

            st.rerun()

        if st.session_state.live_notice:
            st.markdown(
                f"<div class='alert-info'><b>{st.session_state.live_notice}</b></div>",
                unsafe_allow_html=True,
            )

        _render_live_result(st.session_state.live_result)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Recent Incoming Incidents")

        recent_live = _fetch_recent_live_incidents(limit=25)
        if recent_live.empty:
            st.info("No incoming incidents have been stored yet.")
        else:
            display_cols = [
                c for c in [
                    "incident_id", "crime_type", "crime_severity", "district",
                    "neighborhood", "occurred_date", "occurred_time",
                    "priority_level", "source", "status", "submitted_at"
                ] if c in recent_live.columns
            ]
            st.dataframe(
                recent_live[display_cols],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Early-Warning Alerts")
        alerts = _fetch_live_alerts(limit=10)
        if not alerts:
            st.markdown(
                "<div class='green-box'><p>✅ No active risk-escalation alerts in the latest records.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            for a in alerts:
                cls = "alert-crit" if a.get("risk_level") == "CRITICAL" else "alert-warn"
                st.markdown(
                    f"<div class='{cls}'><b>{a.get('risk_level','ALERT')}</b> — "
                    f"{a.get('message','')}<br>"
                    f"<b>Recommendation:</b> {a.get('recommendation','')}</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════
#  PAGE — RESOURCE ALLOCATION
# ══════════════════════════════════════════
if st.session_state.page == "resources":
    st.markdown("<h2>Resource Priority</h2>", unsafe_allow_html=True)

    st.markdown(
        """<div class='briefing'><p>
        <b style='color:#00C8FF;'>DECISION-SUPPORT RESOURCE ALLOCATION</b><br>
        The system estimates how a fixed pool of resources could be prioritized
        across areas using historical risk and incoming incident activity.
        These are recommendations for an authorized decision-maker, not
        automatic deployment commands.
        </p></div>""",
        unsafe_allow_html=True,
    )

    total_units = st.slider(
        "Available patrol / response units",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        key="resource_units",
    )

    alloc = _resource_allocation(total_units)
    if alloc.empty:
        st.info("Resource allocation will appear after the risk baseline is available.")
    else:
        top = alloc.iloc[0]
        top_color = _risk_color(top["Risk"])

        st.markdown(
            f"""<div class='intel-card' style='border-color:{top_color};'>
                <h3 style='color:{top_color} !important;'>Highest Priority Area: {top['Zone']}</h3>
                <p><b>Risk:</b> {top['Risk']} |
                   <b>Score:</b> {top['Risk Score']}/100 |
                   <b>Recommended units:</b> {top['Recommended Units']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.dataframe(
            alloc[
                ["Zone", "Risk Score", "Risk", "Incoming Incidents", "Recommended Units"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        fig_ra = px.bar(
            alloc,
            x="Zone",
            y="Recommended Units",
            color="Risk",
            title=f"Recommended Allocation of {total_units} Available Units",
            color_discrete_map={
                "CRITICAL": "#E24B4A",
                "HIGH": "#EF9F27",
                "MODERATE": "#FAC775",
                "LOW": "#1D9E75",
            },
        )
        fig_ra.update_layout(**pcfg(330))
        st.plotly_chart(fig_ra, use_container_width=True)

        st.markdown(
            """<div class='warn-box'><p>
            ⚠️ <b>Decision-support notice:</b> The allocation above is an
            analytical recommendation based on observed area-level patterns.
            It must be reviewed by an authorized human decision-maker before
            any real-world resource action.
            </p></div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════
#  SPLASH
# ══════════════════════════════════════════
if st.session_state.page == "splash":
    inject_css(splash_bg, 0.38)
    st.markdown("""
    <div style='text-align:center;padding:44px 0 4px;'>
        <h1 style='font-size:88px !important;letter-spacing:8px;
            text-shadow:0 0 70px rgba(0,200,255,0.65),4px 4px 18px black;'>UrbanShield</h1>
        <p style='font-size:22px !important;color:rgba(255,255,255,0.88) !important;
            letter-spacing:2px;margin-top:4px;font-weight:500 !important;'>
            AI-Powered Urban Crime Intelligence & Decision Support System</p>
        <hr style='border:none;height:1px;width:40%;margin:14px auto;
            background:linear-gradient(90deg,transparent,rgba(0,200,255,0.55),transparent);'>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1,2,1])
    with col:
        valid = [(i,b) for i,b in enumerate(crime_imgs_b64) if b]
        if valid:
            idx = st.session_state.slide_idx % len(valid); _, ib = valid[idx]
            st.markdown(f"""
            <div style='text-align:center;margin-bottom:12px;'>
                <div style='display:inline-block;border-radius:14px;overflow:hidden;
                    border:1px solid rgba(0,200,255,0.42);
                    box-shadow:0 0 34px rgba(0,200,255,0.2),0 10px 36px rgba(0,0,0,0.65);'>
                    <img src="data:image/jpeg;base64,{ib}"
                        style='height:250px;max-width:100%;object-fit:cover;display:block;'/>
                </div>
                <p style='font-size:13px !important;color:rgba(0,200,255,0.58) !important;
                    margin:7px 0 0;letter-spacing:1px;font-family:"Share Tech Mono",monospace !important;'>
                    Crime Types &nbsp;|&nbsp; {idx+1} / {len(valid)}</p>
            </div>""", unsafe_allow_html=True)
            b1,_,b2 = st.columns([1,1,1])
            with b1:
                if st.button("Previous",key="prev"):
                    st.session_state.slide_idx=(st.session_state.slide_idx-1)%len(valid); st.rerun()
            with b2:
                if st.button("Next",key="nxt"):
                    st.session_state.slide_idx=(st.session_state.slide_idx+1)%len(valid); st.rerun()

        st.markdown("""
        <div style='background:rgba(0,5,18,0.80);border:1px solid rgba(0,200,255,0.22);
            border-radius:14px;padding:24px 30px;margin:14px 0;backdrop-filter:blur(12px);'>
            <p style='font-size:17px !important;line-height:2.0;color:#ccddeeff !important;
                text-align:justify;margin:0;'>
                Urban crime is a growing and constantly changing challenge that affects
                <b style='color:#00C8FF;'>public safety</b>, law enforcement efficiency,
                and community well-being. With increasing urbanization and population growth,
                crime patterns are becoming more complex and harder to manage in real time.
                There is a need to better understand <b style='color:#00C8FF;'>where and when
                crimes are happening</b> so that authorities can respond faster and
                prevent further incidents.
            </p>
        </div>""", unsafe_allow_html=True)

        s1,s2,s3 = st.columns(3)
        for c,v,l in [(s1,"5,000","Incidents"),(s2,"7","Districts"),(s3,"15","Crime Types")]:
            with c: st.markdown(f"<div class='mcard'><span class='mcard-num'>{v}</span>"
                                f"<span class='mcard-lbl'>{l}</span></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ENTER SYSTEM", key="go"):
            st.session_state.page="login"; st.rerun()

# ══════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════
# ══════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════
if st.session_state.page == "login":
    inject_css(inner_bg, 0.64)

    st.markdown("""
    <div style='text-align:center;padding:44px 0 10px;'>
        <h1 style='font-size:54px !important;'>UrbanShield</h1>
        <p style='font-size:14px !important;color:rgba(0,200,255,0.7) !important;
            letter-spacing:3px;text-transform:uppercase;
            font-family:"Share Tech Mono",monospace !important;'>
            Authorised Personnel Only
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, c2, _ = st.columns([1, 1.1, 1])

    with c2:
        st.markdown("""
        <div style='background:rgba(0,5,18,0.85);
            border:1px solid rgba(0,200,255,0.3);
            border-radius:14px;
            padding:32px 36px;
            backdrop-filter:blur(22px);'>
        """, unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align:center;color:#00C8FF !important;"
            "letter-spacing:2px;'>SECURE ACCESS</h3>",
            unsafe_allow_html=True
        )

        u = st.text_input(
            "Badge ID / Username",
            key="lu"
        )

        p = st.text_input(
            "Password",
            type="password",
            key="lp"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("AUTHENTICATE", key="login_btn"):

            # Normalize username so MONICA, Monica and monica
            # are treated as the same username.
            username = u.strip().upper()

            if authenticate(username, p):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.session_state.g_result = run_analysis(
                    st.session_state.g_crime,
                    st.session_state.g_district,
                    st.session_state.g_time,
                    st.session_state.g_weapon
                )

                st.session_state.page = "command"
                st.rerun()

            else:
                st.error("Invalid credentials.")

        st.markdown(
            "<p style='font-size:12px !important;color:#445 !important;"
            "text-align:center;margin-top:12px;'>"
            "Access restricted to authorised personnel only."
            "</p>",
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Back", key="back"):
            st.session_state.page = "splash"
            st.rerun()

# ══════════════════════════════════════════
#  MAIN SHELL
# ══════════════════════════════════════════
PAGES = ["command","map","threats","targets","timing","damage","directives","live","resources"]
if st.session_state.page in PAGES:
    inject_css(inner_bg, 0.60)
    if not st.session_state.logged_in:
        st.session_state.page="login"; st.rerun()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:10px 0 4px;'>
            <h2 style='font-size:19px !important;letter-spacing:3px;color:#00C8FF !important;'>URBANSHIELD</h2>
            <p style='font-size:10px !important;color:rgba(0,200,255,0.4) !important;letter-spacing:2px;
                margin:0;text-transform:uppercase;font-family:"Share Tech Mono",monospace !important;'>
                Intelligence System</p>
        </div>
        <hr style='border-color:rgba(0,200,255,0.18);margin:8px 0;'>
        <p style='font-size:13px !important;color:#aaa !important;'>
            <span class='ldot'></span>Officer: <b style='color:#00C8FF !important;'>{st.session_state.username}</b></p>
        <hr style='border-color:rgba(255,255,255,0.06);margin:6px 0 10px;'>
        """, unsafe_allow_html=True)

        st.markdown("""<div style='background:rgba(0,200,255,0.055);border:1px solid rgba(0,200,255,0.18);
            border-radius:9px;padding:12px;margin-bottom:12px;'>
            <p style='font-size:11px !important;color:#00C8FF !important;letter-spacing:1px;
                text-transform:uppercase;margin-bottom:8px;font-weight:700 !important;'>
                Intelligence Filter</p>""", unsafe_allow_html=True)

        def sb_idx(lst,val): return lst.index(val) if val in lst else 0
        ct_list = sorted(df["crime_type"].unique().tolist())
        di_list = sorted(df["district"].unique().tolist())
        ti_list = sorted(df["time_category"].unique().tolist())
        wp_list = ["Unknown"]+sorted(df["weapon_used"].unique().tolist())

        nc = st.selectbox("Crime Type",  ct_list, index=sb_idx(ct_list, st.session_state.g_crime),    key="sb_ct")
        nd = st.selectbox("District",    di_list, index=sb_idx(di_list, st.session_state.g_district), key="sb_di")
        nt = st.selectbox("Time of Day", ti_list, index=sb_idx(ti_list, st.session_state.g_time),     key="sb_ti")
        nw = st.selectbox("Weapon",      wp_list, index=sb_idx(wp_list, st.session_state.g_weapon),   key="sb_wp")

        changed = (nc!=st.session_state.g_crime or nd!=st.session_state.g_district or
                   nt!=st.session_state.g_time  or nw!=st.session_state.g_weapon)
        if changed:
            st.session_state.g_crime=nc; st.session_state.g_district=nd
            st.session_state.g_time=nt;  st.session_state.g_weapon=nw
            st.session_state.g_result=run_analysis(nc,nd,nt,nw); st.rerun()
        if st.button("RUN ANALYSIS", key="run"):
            st.session_state.g_result=run_analysis(nc,nd,nt,nw); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── NAV ──
        st.markdown("<p style='font-size:10px !important;color:#445 !important;"
                    "letter-spacing:1px;text-transform:uppercase;margin-bottom:7px;'>Modules</p>",
                    unsafe_allow_html=True)
        nav = [("Command Center",            "command"),
               ("Crime Location Map",        "map"),
               ("Threat Intelligence",       "threats"),
               ("Target Profile Analysis",   "targets"),
               ("Operational Timing",        "timing"),
               ("Economic Damage Assessment","damage"),
               ("Command Directives",        "directives"),
               ("Live Operations",            "live"),
               ("Resource Priority",          "resources")]
        for label,pk in nav:
            lbl = f"**{label}**" if st.session_state.page==pk else label
            if st.button(lbl, key=f"nav_{pk}"):
                st.session_state.page=pk; st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:10px 0;'>",
                    unsafe_allow_html=True)
        R2 = st.session_state.g_result
        if R2:
            rc0 = {"CRITICAL":"#E24B4A","HIGH":"#EF9F27","MODERATE":"#FAC775","LOW":"#1D9E75"}.get(R2["risk_level"],"#888")
            st.markdown(f"<div style='background:rgba(0,5,18,0.5);border:1px solid rgba(0,200,255,0.12);"
                        f"border-radius:8px;padding:9px;margin-bottom:10px;'>"
                        f"<p style='font-size:13px !important;color:{rc0} !important;margin:0;font-weight:700 !important;'>"
                        f"Risk: {R2['risk_level']} ({R2['risk_score']}%)</p>"
                        f"<p style='font-size:12px !important;color:#888 !important;margin:3px 0 0;'>"
                        f"{R2['similar_count']} matching incidents</p></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("LOGOUT", key="lo"):
            st.session_state.logged_in=False; st.session_state.page="login"; st.rerun()

    if not st.session_state.g_result:
        st.session_state.g_result=run_analysis(
            st.session_state.g_crime,st.session_state.g_district,
            st.session_state.g_time,st.session_state.g_weapon)
    R = st.session_state.g_result

# ══════════════════════════════════════════
#  PAGE 1 — COMMAND CENTER
# ══════════════════════════════════════════
if st.session_state.page == "command":
    st.markdown("<h2>Command Center</h2>", unsafe_allow_html=True)
    st.metric("Total Records in Dataset", f"{len(df):,}")
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    rc  = {"CRITICAL":"#E24B4A","HIGH":"#EF9F27","MODERATE":"#FAC775","LOW":"#1D9E75"}.get(R["risk_level"],"#888")
    gc  = "#E24B4A" if R["gang_pred"]=="Yes" else "#1D9E75"
    gang_status = (f"Gang involvement: <b style='color:#E24B4A;'>GANG INVOLVED</b>"
                   if R["gang_pred"]=="Yes"
                   else f"Gang involvement: <b style='color:#1D9E75;'>GANG NOT INVOLVED</b>")

    briefing_html = (f"<b style='color:#00C8FF;'>OPERATIONAL BRIEFING</b> — "
                     f"{R['crime'].upper()} incident in <b>{R['district']} District</b> "
                     f"during <b>{R['time']}</b> hours. "
                     f"{gang_status}. "
                     f"Threat level is <b style='color:{rc};'>{R['risk_level']}</b> "
                     f"(score: {R['risk_score']}%). "
                     f"Based on <b>{R['similar_count']}</b> historical similar incidents, "
                     f"estimated financial impact: "
                     f"<b style='color:#1D9E75;'>${R['loss_min']:,.0f}</b> — "
                     f"<b style='color:#EF9F27;'>${R['loss_mean']:,.0f}</b> — "
                     f"<b style='color:#E24B4A;'>${R['loss_max']:,.0f}</b>. "
                     f"Case resolution: <b>{R['arrest_display']}</b> "
                     f"({abs(R['arrest_gap'])}% {'above' if R['arrest_gap']>=0 else 'below'} city average). "
                     f"Predicted arrest probability: <b>{R['knn_arrest_prob']}%</b>.")
    st.markdown(f"<div class='briefing'><p>{briefing_html}</p></div>", unsafe_allow_html=True)

    if R["risk_level"]=="CRITICAL":
        st.markdown(f"<div class='alert-crit'><b>CRITICAL ALERT</b> — {R['risk_txt']} {R['gang_txt']}</div>", unsafe_allow_html=True)
    elif R["risk_level"]=="HIGH":
        st.markdown(f"<div class='alert-warn'><b>HIGH RISK</b> — {R['risk_txt']} {R['gang_txt']}</div>", unsafe_allow_html=True)
    elif R["gang_pred"]=="Yes":
        st.markdown(f"<div class='alert-warn'><b>GANG INTELLIGENCE ALERT</b> — {R['gang_txt']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-info'><b>ACTIVE CASE</b> — {R['risk_txt']} {R['gang_txt']}</div>", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    for col,lbl,val,sub,clr in [
        (k1,"Gang Status",              R["gang_label"],           "Based on historical patterns",                       gc),
        (k2,"Threat Level",             R["risk_level"],           f"Risk score: {R['risk_score']}%",                    rc),
        (k3,"Similar Cases",            f"{R['similar_count']}",   R["arrest_display"],                                  "#00C8FF"),
        (k4,"Est. Impact",              f"${R['loss_mean']:,.0f}", f"Range: ${R['loss_min']:,.0f}–${R['loss_max']:,.0f}", "#EF9F27"),
        (k5,"Predicted Arrest Likelihood", f"{R['knn_arrest_prob']}%","Based on similar past incidents",                "#1D9E75"),
    ]:
        with col:
            st.markdown(f"""<div class='mcard' style='border-color:{clr};'>
                <span class='mcard-num' style='color:{clr} !important;font-size:20px !important;'>{val}</span>
                <span class='mcard-lbl'>{lbl}</span>
                <p style='font-size:12px !important;color:#888 !important;margin:6px 0 0;'>{sub}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    g1,g2 = st.columns(2)

    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=R["risk_score"],
            title={"text":"Threat Level Score","font":{"color":"white","size":13}},
            number={"suffix":"%","font":{"color":"white","size":36}},
            gauge={"axis":{"range":[0,100],"ticksuffix":"%","tickcolor":"white"},
                   "bar":{"color":rc,"thickness":0.28},
                   "steps":[{"range":[0,25],"color":"rgba(29,158,117,0.12)"},
                             {"range":[25,45],"color":"rgba(250,199,117,0.12)"},
                             {"range":[45,70],"color":"rgba(239,159,39,0.12)"},
                             {"range":[70,100],"color":"rgba(226,75,74,0.12)"}],
                   "threshold":{"line":{"color":"white","width":3},"thickness":0.8,"value":R["risk_score"]}}))
        fig.update_layout(**pcfg(240)); st.plotly_chart(fig, use_container_width=True)
        insight(f"A score of <b>{R['risk_score']}%</b> places this incident in the "
                f"<b style='color:{rc};'>{R['risk_level']}</b> category. {R['risk_txt']}")

    with g2:
        gang_counts = df["gang_related"].value_counts(normalize=True) * 100
        gp = pd.DataFrame({
            "Outcome": ["Gang Involved","Gang Not Involved"],
            "Probability": [round(gang_counts.get("Yes",0),1), round(gang_counts.get("No",0),1)]
        })
        fig2 = px.bar(gp, x="Outcome", y="Probability", color="Outcome",
                      color_discrete_map={"Gang Involved":"#E24B4A","Gang Not Involved":"#1D9E75"},
                      title="Gang Involvement — City-Wide Distribution", text="Probability")
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(**pcfg(240), showlegend=False); st.plotly_chart(fig2, use_container_width=True)
        insight(R["gang_txt"], "red" if R["gang_pred"]=="Yes" else "green")

    # ── Average Victim Count by Crime Type ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Average Victim Count by Crime Type")
    q1_district_sel = st.selectbox(
        "Filter by district", ["All"] + sorted(df["district"].unique().tolist()), key="q1_dist")
    q1_df = df if q1_district_sel=="All" else df[df["district"]==q1_district_sel]
    q1_data = q1_df.groupby("crime_type")["victim_count"].mean().reset_index()
    q1_data.columns = ["Crime Type","Avg Victims"]
    q1_data = q1_data.sort_values("Avg Victims", ascending=False)
    fig_q1 = px.bar(q1_data, x="Crime Type", y="Avg Victims",
                    title=f"Average Victim Count per Crime Type — {q1_district_sel} District",
                    color="Crime Type",
                    color_discrete_sequence=px.colors.qualitative.Set1)
    fig_q1.update_layout(**pcfg(280), xaxis_tickangle=-35)
    st.plotly_chart(fig_q1, use_container_width=True)
    top_q1 = q1_data.iloc[0]
    insight(f"<b>{top_q1['Crime Type']}</b> has the highest average victim count "
            f"(<b>{top_q1['Avg Victims']:.2f}</b> per incident) in "
            f"<b>{q1_district_sel}</b>. "
            f"Multi-victim response protocols and victim support units should be pre-positioned for this crime type.", "warn")

    # ── Gang Status vs Weapon Usage ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Gang Status vs Weapon Usage")
    q8_crime_sel = st.selectbox(
        "Filter by crime type", ["All"] + sorted(df["crime_type"].unique().tolist()),
        index=(["All"]+sorted(df["crime_type"].unique().tolist())).index(R["crime"])
        if R["crime"] in df["crime_type"].unique() else 0, key="q8_crime")
    q8_df = df if q8_crime_sel=="All" else df[df["crime_type"]==q8_crime_sel]
    q8_data = q8_df.groupby(["gang_related","weapon_used"]).size().reset_index(name="Incidents")
    fig_q8 = px.bar(q8_data, x="weapon_used", y="Incidents", color="gang_related",
                    barmode="group",
                    title=f"Gang Involvement vs Weapon Usage — {q8_crime_sel}",
                    color_discrete_map={"Yes":"#E24B4A","No":"#1D9E75"},
                    labels={"gang_related":"Gang Involved","weapon_used":"Weapon"})
    fig_q8.update_layout(**pcfg(290), xaxis_tickangle=-30)
    st.plotly_chart(fig_q8, use_container_width=True)
    gang_weapon = q8_df[q8_df["gang_related"]=="Yes"]["weapon_used"].mode()
    if len(gang_weapon)>0:
        insight(f"Gang-involved incidents in <b>{q8_crime_sel}</b> most frequently use "
                f"<b>{gang_weapon[0]}</b>. Armed response protocols must be elevated when gang "
                f"involvement is suspected — especially for this weapon type.", "red")

    fdf = R["fdf"]
    if len(fdf)>0:
        st.markdown("<br>", unsafe_allow_html=True)
        sec(f"Pattern Analysis — {R['crime']} in {R['district']} during {R['time']}")
        c1,c2 = st.columns(2)
        with c1:
            wc = fdf["weapon_used"].value_counts().reset_index(); wc.columns=["Weapon","Incidents"]
            fig = px.bar(wc, x="Weapon", y="Incidents", color="Incidents",
                         color_continuous_scale="Reds", title="Weapon Profile — Filtered Incidents")
            fig.update_layout(**pcfg(240)); st.plotly_chart(fig, use_container_width=True)
            insight(auto_conclude(fdf["weapon_used"].value_counts(), "weapon_dist"))
        with c2:
            sev = fdf["crime_severity"].value_counts().reset_index(); sev.columns=["Severity","Count"]
            fig2 = px.pie(sev, values="Count", names="Severity", title="Severity Breakdown",
                          color_discrete_map={"Felony":"#E24B4A","Misdemeanor":"#EF9F27","Infraction":"#1D9E75"})
            fig2.update_layout(**pcfg(240)); st.plotly_chart(fig2, use_container_width=True)
            top_sev = sev.iloc[0]["Severity"]; top_sev_pct = round(sev.iloc[0]["Count"]/sev["Count"].sum()*100)
            insight(f"<b>{top_sev}</b> charges account for <b>{top_sev_pct}%</b> of {R['crime']} incidents "
                    f"in {R['district']}.")

        sec("Incident Summary — Matching Cases")
        m1,m2,m3,m4 = st.columns(4)
        top_severity = fdf["crime_severity"].mode()[0] if len(fdf)>0 else "N/A"
        top_weapon   = fdf["weapon_used"].mode()[0]    if len(fdf)>0 else "N/A"
        top_premises = fdf["premises_type"].mode()[0]  if "premises_type" in fdf.columns and len(fdf)>0 else "N/A"
        with m1:
            st.markdown(f"<div class='mcard'><span class='mcard-num' style='font-size:22px !important;'>"
                        f"{R['similar_count']}</span><span class='mcard-lbl'>Total Cases</span></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='mcard'><span class='mcard-num' style='font-size:16px !important;"
                        f"color:#EF9F27 !important;'>{top_severity}</span>"
                        f"<span class='mcard-lbl'>Most Common Severity</span></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='mcard'><span class='mcard-num' style='font-size:16px !important;"
                        f"color:#E24B4A !important;'>{top_weapon}</span>"
                        f"<span class='mcard-lbl'>Common Weapon</span></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='mcard'><span class='mcard-num' style='font-size:16px !important;"
                        f"color:#1D9E75 !important;'>{R['resolved_count']}/{R['similar_count']}</span>"
                        f"<span class='mcard-lbl'>Cases Resolved ({R['arrest_rate']}%)</span></div>", unsafe_allow_html=True)

        sev2 = fdf.groupby("crime_severity").size().reset_index(name="Count")
        fig_sev = px.bar(sev2, x="crime_severity", y="Count",
                         title="Severity Distribution of Matching Incidents",
                         color="crime_severity",
                         color_discrete_map={"Felony":"#E24B4A","Misdemeanor":"#EF9F27","Infraction":"#1D9E75"})
        fig_sev.update_layout(**pcfg(220)); st.plotly_chart(fig_sev, use_container_width=True)
        insight(R["gap_txt"] + f" ({R['arrest_display']})")

        sec("Geospatial Distribution — Incident Locations")
        mdf = fdf.dropna(subset=["latitude","longitude"])
        if len(mdf)>0:
            m = folium.Map(location=[mdf["latitude"].mean(),mdf["longitude"].mean()],
                           zoom_start=13, tiles="CartoDB dark_matter")
            HeatMap([[r["latitude"],r["longitude"]] for _,r in mdf.iterrows()],
                    radius=14, blur=16, min_opacity=0.45).add_to(m)
            for _,row in mdf.sample(min(80,len(mdf)),random_state=42).iterrows():
                folium.CircleMarker(location=[row["latitude"],row["longitude"]],radius=5,
                    color="#E24B4A",fill=True,fill_opacity=0.72,weight=1,
                    popup=folium.Popup(f"<b>{row['crime_type']}</b><br>{row['district']}<br>{row['time_category']}",
                                       max_width=180)).add_to(m)
            st_folium(m, width=None, height=360, returned_objects=[])
    else:
        st.info("No exact matches for the current filter. Showing district-wide distribution.")
        w2 = df[df["crime_type"]==R["crime"]]["district"].value_counts().reset_index()
        w2.columns=["District","Incidents"]
        fig = px.bar(w2,x="District",y="Incidents",title=f"{R['crime']} — City-Wide Distribution",
                     color="Incidents",color_continuous_scale="Reds")
        fig.update_layout(**pcfg(260)); st.plotly_chart(fig,use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Recommended Police Actions")
    actions=[]
    if R["risk_level"] in ["CRITICAL","HIGH"]:
        actions+=[("Immediate Deployment","Deploy rapid response unit. Establish perimeter and secure scene."),
                  ("Escalate to Command","Notify senior officer and adjacent patrol units immediately.")]
    if R["gang_pred"]=="Yes":
        actions+=[("Gang Database Check","Cross-reference gang databases. Coordinate with anti-gang task force."),
                  ("Tactical Briefing","Minimum 3 officers before engagement. Request armed backup.")]
    if R["weapon"] in ["Firearm","Knife"]:
        actions+=[("Armed Response Protocol","Follow armed-incident SOP. Request specialist unit if available.")]
    if R["risk_level"]=="MODERATE":
        actions+=[("Surveillance Increase","Step up patrols 48 hours. Flag for pattern review."),
                  ("Evidence Priority","Prioritise evidence collection — directly improves resolution rate.")]
    if R["risk_level"]=="LOW":
        actions+=[("Standard Response","Follow standard protocol. Community liaison recommended.")]
    actions+=[("Cross-Reference History",f"Compare against {R['similar_count']} similar past incidents."),
              ("Update Operational Map","Log to duty system and mark on operational map.")]
    for i in range(0,min(len(actions),6),2):
        c1,c2 = st.columns(2)
        for j,(title,desc) in enumerate(actions[i:i+2]):
            with [c1,c2][j]:
                st.markdown(f"<div class='action-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  PAGE 2 — CRIME LOCATION MAP
# ══════════════════════════════════════════
if st.session_state.page == "map":
    st.markdown("<h2>Crime Location Map</h2>", unsafe_allow_html=True)
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    fc1,fc2 = st.columns(2)
    with fc1: show_all  = st.checkbox("Show all crime types — this district", value=False)
    with fc2: show_city = st.checkbox("City-wide view", value=False)

    if show_city:     mdf=df.copy()
    elif show_all:    mdf=R["city_dist"].copy()
    else:             mdf=R["fdf"].copy()
    mdf = mdf.dropna(subset=["latitude","longitude"])
    st.markdown(f"<p style='font-size:14px !important;color:#aaa !important;'>"
                f"Displaying <b style='color:#00C8FF !important;'>{len(mdf):,}</b> incidents "
                f"matching current filter.</p>", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["High-Risk Crime Zones","District Priority Levels",
                             "Crime Density Areas","Concentrated Crime Regions"])

    with t1:
        if len(mdf)>0:
            m = folium.Map(location=[mdf["latitude"].mean(),mdf["longitude"].mean()],
                           zoom_start=12, tiles="CartoDB dark_matter")
            HeatMap([[r["latitude"],r["longitude"]] for _,r in mdf.iterrows()],
                    radius=14, blur=17, min_opacity=0.42).add_to(m)
            for _,row in mdf.sample(min(600,len(mdf)),random_state=42).iterrows():
                folium.CircleMarker(location=[row["latitude"],row["longitude"]],radius=4,
                    color="#E24B4A",fill=True,fill_opacity=0.65,weight=1,
                    popup=folium.Popup(f"<b>{row['crime_type']}</b><br>{row['district']}<br>{row['time_category']}",
                                       max_width=180)).add_to(m)
            st_folium(m, width=None, height=520, returned_objects=[])
            top_zone = mdf["district"].mode()[0] if "district" in mdf.columns else R["district"]
            insight(f"Highest incident density in <b>{top_zone}</b>. "
                    f"<b>{len(mdf):,} incidents</b> plotted in this view.")
        else: st.warning("No incidents match current filters.")

    with t2:
        q6 = df.groupby(["district","priority_level"]).size().reset_index(name="Incidents")
        fig = px.bar(q6,x="district",y="Incidents",color="priority_level",barmode="group",
                     title="Response Priority Distribution by District",
                     color_discrete_map={"High":"#E24B4A","Medium":"#EF9F27","Low":"#1D9E75"})
        fig.update_layout(**pcfg(320)); st.plotly_chart(fig,use_container_width=True)
        dist_high = q6[q6["priority_level"]=="High"].set_index("district")["Incidents"]
        if len(dist_high)>0:
            insight(auto_conclude(dist_high,"crime_by_district"))

    with t3:
        sec("Crime Density Analysis")
        zone_df = df.dropna(subset=["latitude","longitude","zone_cluster"]).copy()
        zone_df = zone_df[zone_df["zone_cluster"]>=0]
        zone_sel = st.selectbox("Filter by crime type",
                                ["All"]+sorted(df["crime_type"].unique().tolist()),
                                key="km_sel")
        zone_plot = zone_df if zone_sel=="All" else zone_df[zone_df["crime_type"]==zone_sel]

        if len(zone_plot)>0:
            fig_km = px.scatter_mapbox(
                zone_plot.sample(min(800,len(zone_plot)), random_state=42),
                lat="latitude", lon="longitude",
                color="zone_cluster",
                color_continuous_scale="Turbo",
                zoom=11, height=480,
                title=f"Crime Density Areas — {zone_sel}",
                hover_data={"crime_type":True,"district":True,"zone_cluster":True}
            )
            fig_km.update_layout(mapbox_style="carto-darkmatter",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="white", margin={"t":36,"b":0,"l":0,"r":0})
            st.plotly_chart(fig_km, use_container_width=True)

            zone_summary = zone_plot.groupby("zone_cluster").agg(
                Incidents=("crime_type","count"),
                Top_Crime=("crime_type", lambda x: x.mode()[0]),
                Avg_Severity=("crime_severity_enc","mean")
            ).reset_index().sort_values("Incidents", ascending=False)
            zone_summary = zone_summary.rename(columns={
                "zone_cluster":"Zone","Top_Crime":"Dominant Crime","Avg_Severity":"Avg Severity"})
            st.dataframe(zone_summary, use_container_width=True, hide_index=True)
            top_z = zone_summary.iloc[0]
            insight(f"<b>Zone {int(top_z['Zone'])}</b> has the highest crime density with "
                    f"<b>{int(top_z['Incidents'])}</b> incidents. Dominant crime: "
                    f"<b>{top_z['Dominant Crime']}</b>. Permanent patrol post recommended.", "warn")
        else:
            st.info("No data for selected filter.")

    with t4:
        sec("Areas with High Incident Concentration")
        db_crime_sel = st.selectbox("Select crime type",
                                    sorted(df["crime_type"].unique().tolist()),
                                    index=sorted(df["crime_type"].unique().tolist()).index(R["crime"])
                                    if R["crime"] in df["crime_type"].unique() else 0,
                                    key="dbscan_sel")
        db_eps = st.slider("Detection sensitivity (radius)", 0.002, 0.05, 0.01, 0.001, key="dbscan_eps")
        db_df  = df[df["crime_type"]==db_crime_sel].dropna(subset=["latitude","longitude"]).copy()

        if len(db_df) >= 10:
            coords = db_df[["latitude","longitude"]].values
            db_model = DBSCAN(eps=db_eps, min_samples=5, algorithm='ball_tree', metric='haversine')
            db_df["_area_id"] = db_model.fit_predict(np.radians(coords))

            n_clusters = len(set(db_df["_area_id"])) - (1 if -1 in db_df["_area_id"].values else 0)
            n_noise    = (db_df["_area_id"]==-1).sum()

            c1_,c2_ = st.columns(2)
            with c1_:
                st.markdown(f"<div class='mcard'><span class='mcard-num'>{n_clusters}</span>"
                            f"<span class='mcard-lbl'>High-Concentration Areas Found</span></div>", unsafe_allow_html=True)
            with c2_:
                st.markdown(f"<div class='mcard'><span class='mcard-num'>{n_noise}</span>"
                            f"<span class='mcard-lbl'>Isolated Incidents</span></div>", unsafe_allow_html=True)

            db_plot = db_df[db_df["_area_id"]>=0].copy()
            db_plot["area_label"] = "Area " + db_plot["_area_id"].astype(str)
            if len(db_plot)>0:
                fig_db = px.scatter_mapbox(
                    db_plot.sample(min(600,len(db_plot)), random_state=42),
                    lat="latitude", lon="longitude",
                    color="area_label",
                    zoom=11, height=460,
                    title=f"High-Concentration Crime Areas — {db_crime_sel}",
                    hover_data={"district":True,"time_category":True}
                )
                fig_db.update_layout(mapbox_style="carto-darkmatter",
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      font_color="white", margin={"t":36,"b":0,"l":0,"r":0})
                st.plotly_chart(fig_db, use_container_width=True)

                area_counts = db_plot.groupby("_area_id").size()
                top_h = area_counts.idxmax()
                top_h_dist = db_plot[db_plot["_area_id"]==top_h]["district"].mode()[0]
                insight(auto_conclude(None,"hotspot",
                        {"zone":top_h,"district":top_h_dist,"count":int(area_counts[top_h])}), "red")
            else:
                st.info("No dense areas found with current sensitivity — try increasing the radius.")
        else:
            st.warning("Insufficient data points. Select a more common crime type.")

# ══════════════════════════════════════════
#  PAGE 3 — THREAT INTELLIGENCE
# ══════════════════════════════════════════
if st.session_state.page == "threats":
    st.markdown("<h2>Threat Intelligence</h2>", unsafe_allow_html=True)
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    sec("Rising Threats — Auto-Detected from Current Data")
    city_monthly = df.groupby(["crime_type","month"]).size().unstack(fill_value=0)
    if len(city_monthly.columns) >= 2:
        recent  = city_monthly.iloc[:,-3:].sum(axis=1)
        earlier = city_monthly.iloc[:,:int(len(city_monthly.columns)/2)].sum(axis=1)
        growth  = ((recent - earlier) / earlier.replace(0,1) * 100).round(1)
        rising  = growth.nlargest(3); falling = growth.nsmallest(3)

        r1,r2,r3 = st.columns(3)
        for col,ct,g in zip([r1,r2,r3],rising.index,rising.values):
            clr = "#E24B4A" if g>10 else "#EF9F27"
            with col:
                st.markdown(f"<div class='mcard' style='border-color:{clr};'>"
                            f"<span class='mcard-num' style='color:{clr} !important;font-size:20px !important;'>{ct}</span>"
                            f"<span class='mcard-lbl'>+{g:.1f}% growth trend</span></div>", unsafe_allow_html=True)
        insight(f"<b>{rising.index[0]}</b> is the fastest-rising crime type city-wide "
                f"(+{rising.values[0]:.1f}%). Followed by <b>{rising.index[1]}</b> and "
                f"<b>{rising.index[2]}</b>. Intelligence resources should be pre-positioned for these categories.","warn")

        st.markdown("<br>", unsafe_allow_html=True)
        r1b,r2b,r3b = st.columns(3)
        for col,ct,g in zip([r1b,r2b,r3b],falling.index,falling.values):
            with col:
                st.markdown(f"<div class='mcard' style='border-color:#1D9E75;'>"
                            f"<span class='mcard-num' style='color:#1D9E75 !important;font-size:20px !important;'>{ct}</span>"
                            f"<span class='mcard-lbl'>{g:.1f}% declining</span></div>", unsafe_allow_html=True)
        insight(f"<b>{falling.index[0]}</b> shows the strongest decline — current prevention strategies working. "
                f"Document and replicate the approach.","green")

    st.markdown("<br>", unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs(["Monthly Crime Flow","Crime Distribution by District",
                                    "Month-by-Month Comparison","Seasonal Patterns",
                                    "Holiday vs Non-Holiday Crime Trends",
                                    "Priority vs Severity Analysis"])

    with t1:
        sel = st.multiselect("Focus on specific crime types",
                             sorted(df["crime_type"].unique().tolist()),
                             default=sorted(df["crime_type"].unique().tolist())[:5])
        mc_f = month_crime[month_crime["crime_type"].isin(sel)] if sel else month_crime
        fig = px.line(mc_f,x="month",y="Count",color="crime_type",markers=True,
                      title="Monthly Crime Volume — Trend Lines",
                      color_discrete_sequence=px.colors.qualitative.Set1,
                      category_orders={"month":MONTH_ORDER})
        fig.update_traces(line_width=2.5,marker_size=8)
        fig.update_layout(**pcfg(360)); st.plotly_chart(fig,use_container_width=True)
        mt = df.groupby("month").size().reindex(MONTH_ORDER,fill_value=0)
        peak_m = mt.idxmax(); low_m = mt.idxmin()
        insight(auto_conclude(None,"monthly_trend",
                {"peak":peak_m,"trough":low_m,"peak_val":int(mt[peak_m]),"trough_val":int(mt[low_m])}))
        mt_df = mt.reset_index(); mt_df.columns=["Month","Incidents"]
        colors_list = ["#E24B4A" if m==peak_m else "#00C8FF" for m in mt_df["Month"]]
        fig2 = go.Figure(go.Bar(x=mt_df["Month"],y=mt_df["Incidents"],
                                marker_color=colors_list,text=mt_df["Incidents"],textposition="outside"))
        fig2.update_layout(**pcfg(240),title="Total Monthly Incident Volume (Red = Peak Month)")
        st.plotly_chart(fig2,use_container_width=True)

    with t2:
        dist_month_reset = dist_month.reindex(columns=MONTH_ORDER,fill_value=0).reset_index()
        dist_month_melt  = dist_month_reset.melt(id_vars="district",var_name="Month",value_name="Incidents")
        dist_month_melt["Month"] = pd.Categorical(dist_month_melt["Month"],categories=MONTH_ORDER,ordered=True)
        dist_month_melt  = dist_month_melt.sort_values("Month")
        fig = px.line(dist_month_melt,x="Month",y="Incidents",color="district",markers=True,
                      title="Crime Distribution by District — Monthly Trends",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(line_width=2.5,marker_size=7)
        fig.update_layout(**pcfg(360)); st.plotly_chart(fig,use_container_width=True)
        dist_totals = dist_month_melt.groupby("district")["Incidents"].sum().reset_index().sort_values("Incidents",ascending=False)
        fig2 = px.bar(dist_totals,x="district",y="Incidents",
                      title="Total Incidents by District (All Months)",
                      color="Incidents",color_continuous_scale="Reds")
        fig2.update_layout(**pcfg(260)); st.plotly_chart(fig2,use_container_width=True)
        top_dist = dist_totals.iloc[0]
        insight(f"<b>{top_dist['district']}</b> district recorded the highest total incidents "
                f"(<b>{int(top_dist['Incidents'])}</b>). Deploy maximum resources here year-round.","warn")

        sec("Districts Grouped by Crime Profile")
        agg_display = dist_feat_agg.reset_index()[["district","agg_group"]]
        agg_merged  = dist_totals.merge(agg_display, on="district", how="left")
        agg_merged["Group"] = "Group " + agg_merged["agg_group"].astype(str)
        fig_agg = px.bar(agg_merged, x="district", y="Incidents", color="Group",
                         title="Districts Grouped by Crime Profile",
                         color_discrete_sequence=["#E24B4A","#00C8FF","#1D9E75"])
        fig_agg.update_layout(**pcfg(270)); st.plotly_chart(fig_agg, use_container_width=True)
        for g in sorted(agg_merged["agg_group"].dropna().unique()):
            grp_dists = agg_merged[agg_merged["agg_group"]==g]["district"].tolist()
            insight(f"<b>Group {int(g)}</b> includes: <b>{', '.join(grp_dists)}</b> — "
                    f"districts sharing similar crime-type composition and severity profiles. "
                    f"Cross-district intelligence sharing within each group is strongly recommended.")

    with t3:
        c1,c2 = st.columns(2)
        with c1: m1s = st.selectbox("Month 1",MONTH_ORDER,index=0,key="cm1")
        with c2: m2s = st.selectbox("Month 2",MONTH_ORDER,index=6,key="cm2")
        if m1s in crime_month_pivot.index and m2s in crime_month_pivot.index:
            comp = pd.DataFrame({"Crime Type":crime_month_pivot.columns,
                                  m1s:crime_month_pivot.loc[m1s].values,
                                  m2s:crime_month_pivot.loc[m2s].values})
            comp["Change"] = comp[m2s]-comp[m1s]
            comp["Direction"] = comp["Change"].apply(lambda x:"Increased" if x>0 else "Decreased" if x<0 else "Stable")
            fig = go.Figure()
            fig.add_trace(go.Bar(name=m1s,x=comp["Crime Type"],y=comp[m1s],marker_color="rgba(0,200,255,0.72)"))
            fig.add_trace(go.Bar(name=m2s,x=comp["Crime Type"],y=comp[m2s],marker_color="rgba(226,75,74,0.72)"))
            fig.update_layout(**pcfg(300),barmode="group",title=f"Crime Volume: {m1s} vs {m2s}",
                              xaxis_tickangle=-30,legend={"font":{"color":"white"}})
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(comp[["Crime Type","Change","Direction"]].sort_values("Change",ascending=False),
                         use_container_width=True,hide_index=True)
            br=comp.loc[comp["Change"].idxmax()]; bf=comp.loc[comp["Change"].idxmin()]
            if br["Change"]>0:
                insight(f"<b>{br['Crime Type']}</b> increased most between {m1s} and {m2s} "
                        f"(+{int(br['Change'])} incidents). Increase patrol allocation.","warn")
            if bf["Change"]<0:
                insight(f"<b>{bf['Crime Type']}</b> decreased most "
                        f"({int(bf['Change'])} incidents). Prevention measures working.","green")
        else: st.info("Select two months to compare.")

    with t4:
        available_seasons = sorted(df["season"].unique().tolist())
        selected_season   = st.selectbox("Select Season to Explore",["All"]+available_seasons,
                                         index=0,key="seas_sel")
        seas_df = df if selected_season=="All" else df[df["season"]==selected_season]
        c1,c2 = st.columns(2)
        with c1:
            seas_ct = seas_df.groupby("crime_type").size().reset_index(name="Incidents").sort_values("Incidents",ascending=False)
            fig = px.bar(seas_ct,x="crime_type",y="Incidents",
                         title=f"Crime Type Breakdown — {selected_season} Season",
                         color="Incidents",color_continuous_scale="Reds")
            fig.update_layout(**pcfg(280),xaxis_tickangle=-35); st.plotly_chart(fig,use_container_width=True)
            if selected_season!="All":
                top_ct = seas_ct.iloc[0]
                insight(f"In <b>{selected_season}</b>, <b>{top_ct['crime_type']}</b> is dominant "
                        f"with <b>{int(top_ct['Incidents'])}</b> incidents.")
            else:
                seas_total = df.groupby("season").size()
                insight(auto_conclude(seas_total,"seasonal"))
        with c2:
            seas2 = df.groupby("season")["estimated_loss"].mean().reset_index()
            seas2.columns=["Season","Avg Loss ($)"]
            colors_seas = ["#E24B4A" if s==selected_season else "#00C8FF" for s in seas2["Season"]]
            fig2 = go.Figure(go.Bar(x=seas2["Season"],y=seas2["Avg Loss ($)"],
                                    marker_color=colors_seas,text=seas2["Avg Loss ($)"].round(0),
                                    texttemplate="$%{text:,.0f}",textposition="outside"))
            fig2.update_layout(**pcfg(280),title="Average Financial Loss by Season")
            st.plotly_chart(fig2,use_container_width=True)
            worst_seas = seas2.loc[seas2["Avg Loss ($)"].idxmax()]
            insight(f"<b>{worst_seas['Season']}</b> causes highest avg damage "
                    f"per incident (${worst_seas['Avg Loss ($)']:,.0f}).","warn")

        sec("District Activity by Season")
        ds_bar = df.groupby(["district","season"]).size().reset_index(name="Incidents")
        if selected_season!="All": ds_bar = ds_bar[ds_bar["season"]==selected_season]
        fig3 = px.bar(ds_bar,x="district",y="Incidents",color="season",barmode="group",
                      title=f"Incident Volume: District × Season ({selected_season})",
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig3.update_layout(**pcfg(300)); st.plotly_chart(fig3,use_container_width=True)
        ds_pivot = df.groupby(["district","season"]).size().unstack(fill_value=0)
        ds_top   = ds_pivot.max(axis=1).idxmax()
        insight(f"<b>{ds_top}</b> district consistently records highest seasonal incident volume.")

    with t5:
        # Holiday vs Non-Holiday Crime Analysis
        sec("Holiday vs Non-Holiday Crime Patterns")
        hol_crime_sel = st.selectbox(
            "Filter by crime type", ["All"]+sorted(df["crime_type"].unique().tolist()), key="hol_crime")
        hol_df = df if hol_crime_sel=="All" else df[df["crime_type"]==hol_crime_sel]

        c1,c2 = st.columns(2)
        with c1:
            hol_count = hol_df.groupby(["is_holiday","crime_type"]).size().reset_index(name="Incidents")
            hol_count["Holiday"] = hol_count["is_holiday"].map({"Yes":"Holiday","No":"Non-Holiday"})
            fig_h1 = px.bar(hol_count, x="crime_type", y="Incidents", color="Holiday",
                            barmode="group",
                            title=f"Crime Volume: Holiday vs Non-Holiday — {hol_crime_sel}",
                            color_discrete_map={"Holiday":"#E24B4A","Non-Holiday":"#00C8FF"})
            fig_h1.update_layout(**pcfg(300), xaxis_tickangle=-30)
            st.plotly_chart(fig_h1, use_container_width=True)

        with c2:
            hol_arr = hol_df.groupby("is_holiday")["arrest_made"].apply(
                lambda x: round(x.value_counts(normalize=True).get("Yes",0)*100,1)
            ).reset_index()
            hol_arr.columns = ["is_holiday","Resolution %"]
            hol_arr["Holiday"] = hol_arr["is_holiday"].map({"Yes":"Holiday","No":"Non-Holiday"})
            fig_h2 = px.bar(hol_arr, x="Holiday", y="Resolution %",
                            color="Holiday",
                            title="Case Resolution Rate: Holiday vs Non-Holiday",
                            color_discrete_map={"Holiday":"#E24B4A","Non-Holiday":"#1D9E75"},
                            text="Resolution %")
            fig_h2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_h2.update_layout(**pcfg(300), showlegend=False)
            st.plotly_chart(fig_h2, use_container_width=True)

        # Crime increase during holidays
        hol_per_type = df.groupby(["is_holiday","crime_type"]).size().unstack(fill_value=0)
        if "Yes" in hol_per_type.index and "No" in hol_per_type.index:
            holiday_ratio = (hol_per_type.loc["Yes"] / hol_per_type.loc["No"]).round(3).reset_index()
            holiday_ratio.columns = ["Crime Type","Holiday Ratio"]
            holiday_ratio = holiday_ratio.sort_values("Holiday Ratio", ascending=False)
            fig_h3 = px.bar(holiday_ratio, x="Crime Type", y="Holiday Ratio",
                            title="Crime Increase During Holidays (Compared to Normal Days)",
                            color="Holiday Ratio",
                            color_continuous_scale="RdYlGn_r")
            fig_h3.add_hline(y=1.0, line_dash="dash", line_color="white",
                             annotation_text="Baseline (No Holiday Effect)")
            fig_h3.update_layout(**pcfg(280), xaxis_tickangle=-30)
            st.plotly_chart(fig_h3, use_container_width=True)
            top_hol = holiday_ratio.iloc[0]
            insight(f"<b>{top_hol['Crime Type']}</b> is {top_hol['Holiday Ratio']:.2f}x more likely "
                    f"on holidays vs regular days. Extra patrols and community safety messaging "
                    f"should be deployed on all public holidays for this crime type.", "red")

        hol_res = hol_arr[hol_arr["is_holiday"]=="Yes"]["Resolution %"].values
        non_res = hol_arr[hol_arr["is_holiday"]=="No"]["Resolution %"].values
        if len(hol_res)>0 and len(non_res)>0:
            diff_h = round(float(hol_res[0]) - float(non_res[0]), 1)
            insight(f"Resolution rate on holidays: <b>{hol_res[0]}%</b> vs non-holidays: "
                    f"<b>{non_res[0]}%</b> — a <b>{abs(diff_h)}% {'drop' if diff_h<0 else 'improvement'}</b>. "
                    f"{'Staffing shortfalls on holidays may be reducing investigative effectiveness.' if diff_h<0 else 'Holiday staffing strategy is working.'}", "warn")

    with t6:
        # Priority Level vs Crime Severity
        sec("Priority Level vs Crime Severity")
        q16_dist_sel = st.selectbox(
            "Filter by district", ["All"]+sorted(df["district"].unique().tolist()), key="q16_dist")
        q16_crime_sel = st.selectbox(
            "Filter by crime type", ["All"]+sorted(df["crime_type"].unique().tolist()), key="q16_crime")
        q16_df = df.copy()
        if q16_dist_sel!="All":  q16_df = q16_df[q16_df["district"]==q16_dist_sel]
        if q16_crime_sel!="All": q16_df = q16_df[q16_df["crime_type"]==q16_crime_sel]

        c1,c2 = st.columns(2)
        with c1:
            ps_cross = q16_df.groupby(["priority_level","crime_severity"]).size().reset_index(name="Incidents")
            fig_ps = px.bar(ps_cross, x="priority_level", y="Incidents", color="crime_severity",
                            barmode="group",
                            title=f"Priority Level vs Crime Severity — {q16_dist_sel} / {q16_crime_sel}",
                            color_discrete_map={"Felony":"#E24B4A","Misdemeanor":"#EF9F27","Infraction":"#1D9E75"})
            fig_ps.update_layout(**pcfg(300))
            st.plotly_chart(fig_ps, use_container_width=True)

        with c2:
            # Stacked bar instead of heatmap
            ps_stack = q16_df.groupby(["priority_level","crime_severity"]).size().reset_index(name="Incidents")
            ps_total = ps_stack.groupby("priority_level")["Incidents"].transform("sum")
            ps_stack["Share (%)"] = (ps_stack["Incidents"] / ps_total * 100).round(1)
            fig_stack = px.bar(ps_stack, x="priority_level", y="Share (%)",
                               color="crime_severity", barmode="stack",
                               title="Crime Severity Breakdown by Priority Level (%)",
                               color_discrete_map={"Felony":"#E24B4A","Misdemeanor":"#EF9F27","Infraction":"#1D9E75"},
                               text="Share (%)")
            fig_stack.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
            fig_stack.update_layout(**pcfg(300))
            st.plotly_chart(fig_stack, use_container_width=True)

        # Resolution rate by priority
        q16_res = q16_df.groupby("priority_level")["arrest_made"].apply(
            lambda x: round(x.value_counts(normalize=True).get("Yes",0)*100,1)
        ).reset_index()
        q16_res.columns = ["Priority Level","Resolution %"]
        fig_pr = px.bar(q16_res, x="Priority Level", y="Resolution %",
                        title="Case Resolution Rate by Priority Level",
                        color="Priority Level",
                        color_discrete_map={"High":"#E24B4A","Medium":"#EF9F27","Low":"#1D9E75"},
                        text="Resolution %")
        fig_pr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_pr.update_layout(**pcfg(260), showlegend=False)
        st.plotly_chart(fig_pr, use_container_width=True)

        if len(q16_res)>0:
            best_p  = q16_res.loc[q16_res["Resolution %"].idxmax()]
            worst_p = q16_res.loc[q16_res["Resolution %"].idxmin()]
            insight(f"<b>{best_p['Priority Level']}</b> priority cases achieve the highest resolution "
                    f"(<b>{best_p['Resolution %']}%</b>), while <b>{worst_p['Priority Level']}</b> priority "
                    f"cases resolve at only <b>{worst_p['Resolution %']}%</b>. "
                    f"Triage processes should be reviewed to ensure high-severity incidents are "
                    f"correctly classified as High priority.", "warn")

# ══════════════════════════════════════════
#  PAGE 4 — TARGET PROFILE ANALYSIS
# ══════════════════════════════════════════
if st.session_state.page == "targets":
    st.markdown("<h2>Target Profile Analysis</h2>", unsafe_allow_html=True)
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    edf = R["city_crime"] if len(R["city_crime"])>0 else df[df["crime_type"]==R["crime"]]
    if len(edf)==0: edf=df.copy()

    top_vg = edf["victim_gender"].mode()[0] if len(edf)>0 else "N/A"
    top_sg = edf["suspect_gender"].mode()[0] if len(edf)>0 else "N/A"
    top_va = edf["victim_age_group"].mode()[0] if "victim_age_group" in edf.columns and len(edf)>0 else "N/A"
    top_sa = edf["suspect_age_group"].mode()[0] if "suspect_age_group" in edf.columns and len(edf)>0 else "N/A"
    top_pr = edf["premises_type"].mode()[0] if len(edf)>0 else "N/A"
    arr_count_t = int(edf["arrest_made"].value_counts().get("Yes",0))
    arr_yes     = round(edf["arrest_made"].value_counts(normalize=True).get("Yes",0)*100,1)
    arr_display_t = f"{arr_count_t}/{len(edf)} resolved ({arr_yes}%)"

    st.markdown(f"""<div class='briefing'>
        <p><b style='color:#00C8FF;'>TARGET PROFILE — {R['crime'].upper()} IN {R['district'].upper()}</b><br>
        Primary victims: <b>{top_vg}</b>, aged <b>{top_va}</b>.
        Typical offender: <b>{top_sg}</b>, aged <b>{top_sa}</b>.
        Most common location: <b>{top_pr}</b>.
        Case resolution: <b>{arr_display_t}</b>.</p>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["Who is Being Targeted?",
                             "Who is Committing These Crimes?",
                             "What Drives Case Resolution?",
                             "Key Factors Influencing Arrests"])

    with t1:
        c1,c2 = st.columns(2)
        with c1:
            vg = edf["victim_gender"].value_counts()
            fig = px.pie(values=vg.values,names=vg.index,title="Victim Gender Distribution",
                         color_discrete_sequence=["#00C8FF","#E24B4A","#1D9E75"])
            fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
            top_vg2 = vg.idxmax(); pct_vg = round(vg[top_vg2]/vg.sum()*100)
            insight(f"<b>{top_vg2}</b> victims account for <b>{pct_vg}%</b> of {R['crime']} cases.")
        with c2:
            if "victim_age_group" in edf.columns:
                va = edf["victim_age_group"].value_counts().reset_index(); va.columns=["Age","Count"]
                fig2 = px.bar(va,x="Age",y="Count",title="Victim Age Group Distribution",
                              color="Count",color_continuous_scale="Oranges")
                fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
                top_va2 = va.iloc[0]["Age"]
                insight(f"Age group <b>{top_va2}</b> is most targeted. "
                        f"Community safety messaging should target this group.","warn")
        sec("Victim Gender × Crime Type — Vulnerability Overview")
        vc_bar = df.groupby(["victim_gender","crime_type"]).size().reset_index(name="Incidents")
        fig3   = px.bar(vc_bar,x="crime_type",y="Incidents",color="victim_gender",barmode="group",
                        title="Victim Gender Distribution Across Crime Types",
                        color_discrete_sequence=["#00C8FF","#E24B4A","#1D9E75"])
        fig3.update_layout(**pcfg(320),xaxis_tickangle=-35); st.plotly_chart(fig3,use_container_width=True)
        insight(auto_conclude(None,"victim_profile",{"top_gender":top_vg,"top_age":top_va}))

    with t2:
        c1,c2 = st.columns(2)
        with c1:
            sg2 = edf["suspect_gender"].value_counts()
            fig = px.pie(values=sg2.values,names=sg2.index,title="Suspect Gender Profile",
                         color_discrete_sequence=["#E24B4A","#00C8FF","#1D9E75"])
            fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
            top_sg2 = sg2.idxmax(); pct_sg = round(sg2[top_sg2]/sg2.sum()*100)
            insight(f"<b>{pct_sg}%</b> of {R['crime']} offenders identified as <b>{top_sg2}</b>.")
        with c2:
            if "suspect_age_group" in edf.columns:
                sa2 = edf["suspect_age_group"].value_counts().reset_index(); sa2.columns=["Age","Count"]
                fig2 = px.bar(sa2,x="Age",y="Count",title="Suspect Age Group Profile",
                              color="Count",color_continuous_scale="Purples")
                fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
                top_sa2 = sa2.iloc[0]["Age"]
                insight(f"Offenders aged <b>{top_sa2}</b> most frequently involved in {R['crime']} cases.")
        sec("Crime Type Composition by District")
        dist_susp = df.groupby(["district","crime_type"]).size().reset_index(name="Incidents")
        fig3 = px.bar(dist_susp,x="district",y="Incidents",color="crime_type",barmode="stack",
                      title="Crime Type Composition by District",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(**pcfg(300)); st.plotly_chart(fig3,use_container_width=True)
        dist_top = dist_susp.groupby("district")["Incidents"].sum().idxmax()
        dist_top_val = dist_susp.groupby("district")["Incidents"].sum().max()
        insight(f"<b>{dist_top}</b> district has the highest overall crime activity "
                f"(<b>{int(dist_top_val)}</b> incidents). Multi-agency intervention recommended.")

    with t3:
        c1,c2 = st.columns(2)
        with c1:
            ev = df.groupby("evidence_collected")["arrest_made"].apply(
                lambda x: round(x.value_counts(normalize=True).get("Yes",0)*100,1)
            ).reset_index()
            ev.columns=["Evidence Collected","Resolution Rate (%)"]
            fig = px.bar(ev,x="Evidence Collected",y="Resolution Rate (%)",
                         title="Does Evidence Collection Drive Resolution?",
                         color="Resolution Rate (%)",color_continuous_scale="Greens")
            fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
            ev_yes = ev[ev["Evidence Collected"]=="Yes"]["Resolution Rate (%)"].values
            ev_no  = ev[ev["Evidence Collected"]=="No"]["Resolution Rate (%)"].values
            if len(ev_yes)>0 and len(ev_no)>0:
                insight(auto_conclude(None,"evidence_resolution",
                        {"with_evidence":float(ev_yes[0]),"without_evidence":float(ev_no[0])}),"green")
        with c2:
            sec("Case Outcome Prediction — by District and Crime")
            rows=[]
            for d in df["district"].unique():
                for c_ in df["crime_type"].unique():
                    sub = df[(df["district"]==d)&(df["crime_type"]==c_)]
                    if len(sub)>5:
                        arr_r = round(sub["arrest_made"].value_counts(normalize=True).get("Yes",0)*100,1)
                        ev_r  = round(sub["evidence_collected"].value_counts(normalize=True).get("Yes",0)*100,1)
                        rows.append({"District":d,"Crime":c_,"Resolution %":arr_r,"Evidence %":ev_r})
            pred_df = pd.DataFrame(rows)
            if len(pred_df)>0:
                res_by_dist = pred_df.groupby("District")["Resolution %"].mean().reset_index().sort_values("Resolution %",ascending=False)
                fig2 = px.bar(res_by_dist,x="District",y="Resolution %",
                              title="Average Case Resolution Rate by District",
                              color="Resolution %",color_continuous_scale="RdYlGn",range_color=[0,100])
                fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
                best  = pred_df.loc[pred_df["Resolution %"].idxmax()]
                worst = pred_df.loc[pred_df["Resolution %"].idxmin()]
                insight(f"<b>{best['District']}</b> resolves <b>{best['Crime']}</b> at "
                        f"<b>{best['Resolution %']}%</b>. Adopt in <b>{worst['District']}</b>.","green")

    with t4:
        sec("Key Factors Influencing Arrests")
        q5_crime_sel = st.selectbox(
            "Filter by crime type",
            ["All"]+sorted(df["crime_type"].unique().tolist()),
            index=(["All"]+sorted(df["crime_type"].unique().tolist())).index(R["crime"])
            if R["crime"] in df["crime_type"].unique() else 0, key="q5_crime")
        q5_df = df if q5_crime_sel=="All" else df[df["crime_type"]==q5_crime_sel]

        feature_labels = {
            "crime_type_enc":        "Crime Type",
            "district_enc":          "District",
            "time_category_enc":     "Time of Day",
            "weapon_used_enc":       "Weapon Used",
            "crime_severity_enc":    "Crime Severity",
            "gang_related_enc":      "Gang Related",
            "evidence_collected_enc":"Evidence Collected"
        }
        y_enc = q5_df["arrest_made"].map({"Yes":1,"No":0}).dropna()
        corr_rows = []
        for feat,label in feature_labels.items():
            if feat in q5_df.columns:
                x_col = q5_df.loc[y_enc.index, feat]
                corr_val = abs(float(x_col.corr(y_enc)))
                corr_rows.append({"Factor":label,"Importance Score":round(corr_val,4)})
        fi_df = pd.DataFrame(corr_rows).sort_values("Importance Score",ascending=True)

        c1_,c2_ = st.columns(2)
        with c1_:
            fig_fi = px.bar(fi_df, x="Importance Score", y="Factor",
                            orientation="h",
                            title=f"Factors Influencing Arrest Outcome — {q5_crime_sel}",
                            color="Importance Score",
                            color_continuous_scale="Blues")
            fig_fi.update_layout(**pcfg(320))
            st.plotly_chart(fig_fi, use_container_width=True)

        with c2_:
            # Arrest Rate by Crime Type bar chart
            arr_by_crime = df.groupby("crime_type")["arrest_made"].apply(
                lambda x: round((x=="Yes").mean()*100, 1)
            ).reset_index()
            arr_by_crime.columns = ["Crime Type","Arrest Rate (%)"]
            arr_by_crime = arr_by_crime.sort_values("Arrest Rate (%)", ascending=False)
            fig_arr = px.bar(arr_by_crime, x="Crime Type", y="Arrest Rate (%)",
                             color="Arrest Rate (%)",
                             color_continuous_scale="RdYlGn",
                             title="Arrest Rate by Crime Type")
            fig_arr.update_layout(**pcfg(320), xaxis_tickangle=-35)
            st.plotly_chart(fig_arr, use_container_width=True)

        top_f  = fi_df.iloc[-1]; bot_f = fi_df.iloc[0]
        insight(f"<b>{top_f['Factor']}</b> is the strongest factor influencing arrest outcomes "
                f"in <b>{q5_crime_sel}</b> cases. "
                f"<b>{bot_f['Factor']}</b> has the weakest impact. "
                f"Investigators should prioritise improving <b>{top_f['Factor']}</b> quality "
                f"for maximum impact on case resolution rates.","green")

# ══════════════════════════════════════════
#  PAGE 5 — OPERATIONAL TIMING
# ══════════════════════════════════════════
if st.session_state.page == "timing":
    st.markdown("<h2>Operational Timing Analysis</h2>", unsafe_allow_html=True)
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    tf1,tf2 = st.columns(2)
    with tf1:
        t_dist  = st.selectbox("District focus",["All"]+sorted(df["district"].unique().tolist()),
                                index=(["All"]+sorted(df["district"].unique().tolist())).index(R["district"])
                                if R["district"] in df["district"].unique() else 0, key="tp_d")
    with tf2:
        t_crime = st.selectbox("Crime type focus",["All"]+sorted(df["crime_type"].unique().tolist()),
                                index=(["All"]+sorted(df["crime_type"].unique().tolist())).index(R["crime"])
                                if R["crime"] in df["crime_type"].unique() else 0, key="tp_c")
    tdf = df.copy()
    if t_dist!="All":  tdf=tdf[tdf["district"]==t_dist]
    if t_crime!="All": tdf=tdf[tdf["crime_type"]==t_crime]

    if len(tdf)>0:
        peak_t = tdf["time_category"].mode()[0]
        peak_d = tdf["day_of_week"].mode()[0]
        peak_s = tdf["season"].mode()[0]
        resolved_t = int(tdf["arrest_made"].value_counts().get("Yes",0))
        arr_r      = round(tdf["arrest_made"].value_counts(normalize=True).get("Yes",0)*100,1)
        arr_disp_t = f"{resolved_t}/{len(tdf)} resolved ({arr_r}%)"
        st.markdown(f"""<div class='briefing'>
            <p><b style='color:#00C8FF;'>TIMING INTELLIGENCE — {t_dist.upper()} | {t_crime.upper()}</b><br>
            Peak activity: <b>{peak_t}</b> period, <b>{peak_d}s</b>, during <b>{peak_s}</b> season.
            Resolution: <b>{arr_disp_t}</b>.</p>
        </div>""", unsafe_allow_html=True)
        s1,s2,s3,s4 = st.columns(4)
        for col,lbl,val,clr in [(s1,"Peak Time",peak_t,"#E24B4A"),(s2,"Peak Day",peak_d,"#EF9F27"),
                                  (s3,"Peak Season",peak_s,"#00C8FF"),
                                  (s4,"Cases Resolved",f"{resolved_t}/{len(tdf)}","#1D9E75")]:
            with col: st.markdown(f"<div class='mcard' style='border-color:{clr};'>"
                                  f"<span class='mcard-num' style='color:{clr} !important;"
                                  f"font-size:18px !important;'>{val}</span>"
                                  f"<span class='mcard-lbl'>{lbl}</span></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    t1,t2,t3,t4 = st.tabs(["When Does Crime Peak?","Day-by-Day Breakdown",
                             "Resolution Timing Patterns","Multi-District Comparison"])
    with t1:
        c1,c2 = st.columns(2)
        with c1:
            tc = tdf.groupby("time_category").size().reset_index(name="Incidents")
            tc = tc.sort_values("Incidents",ascending=False)
            fig = px.bar(tc,x="time_category",y="Incidents",color="Incidents",
                         color_continuous_scale="Reds",title="Incident Volume by Time of Day")
            fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
            insight(auto_conclude(tdf.groupby("time_category").size(),"time_pattern"))
        with c2:
            tt = tdf["time_category"].value_counts().reset_index(); tt.columns=["Time","Share"]
            fig2 = px.pie(tt,values="Share",names="Time",title="Time Period Share of All Incidents",
                          color_discrete_sequence=["#E24B4A","#EF9F27","#00C8FF","#1D9E75"])
            fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
            top_time = tt.iloc[0]["Time"]; top_pct = round(tt.iloc[0]["Share"]/tt["Share"].sum()*100)
            insight(f"<b>{top_pct}%</b> of incidents occur during <b>{top_time}</b> hours.")

        sec("Crime Timing — Day-of-Week Trends by Time Period")
        dt_line = tdf.groupby(["day_of_week","time_category"]).size().reset_index(name="Incidents")
        dt_line["day_of_week"] = pd.Categorical(dt_line["day_of_week"],categories=day_order,ordered=True)
        dt_line = dt_line.sort_values("day_of_week")
        fig3 = px.line(dt_line,x="day_of_week",y="Incidents",color="time_category",markers=True,
                       title="Incident Volume by Day and Time Period",
                       color_discrete_sequence=["#E24B4A","#EF9F27","#00C8FF","#1D9E75"])
        fig3.update_traces(line_width=2.5,marker_size=8)
        fig3.update_layout(**pcfg(320)); st.plotly_chart(fig3,use_container_width=True)
        if len(dt_line)>0:
            peak_row = dt_line.loc[dt_line["Incidents"].idxmax()]
            insight(f"Peak: <b>{peak_row['day_of_week']}</b> during <b>{peak_row['time_category']}</b> "
                    f"— <b>{int(peak_row['Incidents'])}</b> incidents. Maximum patrol density required.","warn")

    with t2:
        dc = tdf.groupby("day_of_week").size().reset_index(name="Incidents")
        dc["day_of_week"] = pd.Categorical(dc["day_of_week"],categories=day_order,ordered=True)
        dc = dc.sort_values("day_of_week")
        fig = px.bar(dc,x="day_of_week",y="Incidents",title="Incidents by Day of Week",
                     color="Incidents",color_continuous_scale="Blues")
        fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
        peak_day_s = dc.set_index("day_of_week")["Incidents"]
        insight(f"<b>{peak_day_s.idxmax()}</b> is the highest-crime day "
                f"({int(peak_day_s.max())} incidents). Weekend surge requires adjusted deployment.","warn")
        q3 = tdf.groupby(["time_category","crime_severity"]).size().reset_index(name="Incidents")
        fig2 = px.bar(q3,x="time_category",y="Incidents",color="crime_severity",barmode="group",
                      title="Incident Severity by Time Period",
                      color_discrete_map={"Felony":"#E24B4A","Misdemeanor":"#EF9F27","Infraction":"#1D9E75"})
        fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
        insight("Night-time incidents carry disproportionately higher <b>Felony</b> share. "
                "Night shift officers require enhanced equipment and backup.")

    with t3:
        c1,c2 = st.columns(2)
        with c1:
            q2 = tdf.groupby(["day_of_week","arrest_made"]).size().reset_index(name="Incidents")
            q2["day_of_week"] = pd.Categorical(q2["day_of_week"],categories=day_order,ordered=True)
            q2 = q2.sort_values("day_of_week")
            fig = px.bar(q2,x="day_of_week",y="Incidents",color="arrest_made",barmode="group",
                         title="Resolution Rate by Day",
                         color_discrete_map={"Yes":"#1D9E75","No":"#E24B4A"})
            fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
            if len(tdf)>0:
                abd = tdf.groupby("day_of_week").apply(
                    lambda x: x["arrest_made"].value_counts(normalize=True).get("Yes",0)*100)
                insight(auto_conclude(abd,"arrest_by_day"),"green")
        with c2:
            q12 = tdf.groupby(["time_category","crime_severity"]).size().unstack(fill_value=0)
            if len(q12)>0:
                q12_pct = q12.div(q12.sum(axis=1),axis=0)*100
                q12_pct_reset = q12_pct.reset_index().melt(id_vars="time_category",
                                                            var_name="Severity",value_name="Percentage")
                fig3 = px.bar(q12_pct_reset,x="time_category",y="Percentage",color="Severity",
                              barmode="stack",title="Severity Proportion by Time of Day",
                              color_discrete_map={"Felony":"#E24B4A","Misdemeanor":"#EF9F27","Infraction":"#1D9E75"})
                fig3.update_layout(**pcfg(270)); st.plotly_chart(fig3,use_container_width=True)
                insight("Felony-level incidents concentrated in <b>Night</b> and <b>Evening</b>. "
                        "Senior officers on-call during these periods.")

    with t4:
        sel_dists = st.multiselect("Compare districts",sorted(df["district"].unique().tolist()),
                                    default=sorted(df["district"].unique().tolist())[:3])
        if sel_dists:
            dm2 = df[df["district"].isin(sel_dists)].groupby(["district","month"]).size().reset_index(name="Incidents")
            fig = px.line(dm2,x="month",y="Incidents",color="district",markers=True,
                          title="Monthly Volume Comparison — Selected Districts",
                          category_orders={"month":MONTH_ORDER},
                          color_discrete_sequence=["#E24B4A","#00C8FF","#1D9E75","#EF9F27","#AFA9EC"])
            fig.update_traces(line_width=2.5,marker_size=8)
            fig.update_layout(**pcfg(340),legend={"font":{"color":"white"}}); st.plotly_chart(fig,use_container_width=True)
            rank_rows=[]
            for d in sel_dists:
                sub_d = df[df["district"]==d]
                res_c = int(sub_d["arrest_made"].value_counts().get("Yes",0))
                res_pct = round(sub_d["arrest_made"].value_counts(normalize=True).get("Yes",0)*100,1)
                rank_rows.append({"District":d,"Total Incidents":len(sub_d),
                                   "Resolved":f"{res_c}/{len(sub_d)} ({res_pct}%)",
                                   "Avg Loss":f"${sub_d['estimated_loss'].mean():,.0f}",
                                   "Dominant Crime":sub_d["crime_type"].mode()[0]})
            rank = pd.DataFrame(rank_rows).sort_values("Total Incidents",ascending=False)
            st.dataframe(rank,use_container_width=True,hide_index=True)
            insight(f"<b>{rank.iloc[0]['District']}</b> has highest incident volume. "
                    f"Cross-district knowledge transfer recommended.","green")
        else: st.info("Select at least one district above.")

# ══════════════════════════════════════════
#  PAGE 6 — ECONOMIC DAMAGE ASSESSMENT
# ══════════════════════════════════════════
if st.session_state.page == "damage":
    st.markdown("<h2>Economic Damage Assessment</h2>", unsafe_allow_html=True)
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    if R["crime"] in loss_stats.index:
        ls = loss_stats.loc[R["crime"]]
        auto_base = round(ls["mean"],2); auto_lo = round(max(0,ls["min"]),2); auto_hi = round(ls["max"],2)
    else:
        auto_base = 0.0; auto_lo = 0.0; auto_hi = 0.0

    st.markdown(f"""<div class='intel-amber'>
        <h3>Estimated Financial Impact — {R["crime"]} in {R["district"]}</h3>
        <p style='font-size:13px !important;color:#aaa !important;margin-bottom:8px;'>
            Based on actual historical data from {R["similar_count"]} similar incidents.</p>
        <div style='display:flex;gap:26px;align-items:center;flex-wrap:wrap;margin-top:10px;'>
            <div><span style='font-size:12px !important;color:#aaa !important;display:block;text-transform:uppercase;'>Minimum</span>
                <span style='font-size:28px !important;font-weight:700;color:#1D9E75 !important;font-family:"Rajdhani",sans-serif !important;'>${auto_lo:,.0f}</span></div>
            <span style='color:#555 !important;font-size:20px !important;'>→</span>
            <div><span style='font-size:12px !important;color:#aaa !important;display:block;text-transform:uppercase;'>Average</span>
                <span style='font-size:38px !important;font-weight:700;color:#EF9F27 !important;font-family:"Rajdhani",sans-serif !important;'>${auto_base:,.0f}</span></div>
            <span style='color:#555 !important;font-size:20px !important;'>→</span>
            <div><span style='font-size:12px !important;color:#aaa !important;display:block;text-transform:uppercase;'>Maximum</span>
                <span style='font-size:28px !important;font-weight:700;color:#E24B4A !important;font-family:"Rajdhani",sans-serif !important;'>${auto_hi:,.0f}</span></div>
        </div>
    </div>""", unsafe_allow_html=True)

    t1,t2 = st.tabs(["Financial Impact by Crime and District","What Drives Investigation Cost?"])
    with t1:
        dmg_district_sel = st.selectbox("Filter district",
                                         ["All"]+sorted(df["district"].unique().tolist()), key="dmg_d")
        dmg_df = df if dmg_district_sel=="All" else df[df["district"]==dmg_district_sel]
        c1,c2 = st.columns(2)
        with c1:
            al = dmg_df.groupby("crime_type")["estimated_loss"].mean().reset_index().sort_values("estimated_loss",ascending=False)
            fig = px.bar(al,x="crime_type",y="estimated_loss",
                         title=f"Average Financial Damage by Crime Type — {dmg_district_sel}",
                         color="estimated_loss",color_continuous_scale="Reds")
            fig.update_layout(**pcfg(290),xaxis_tickangle=-35); st.plotly_chart(fig,use_container_width=True)
            insight(auto_conclude(al.set_index("crime_type")["estimated_loss"],"loss_by_crime"))
        with c2:
            dl = dmg_df.groupby("district")["estimated_loss"].mean().reset_index().sort_values("estimated_loss",ascending=False) if dmg_district_sel=="All" else df.groupby("district")["estimated_loss"].mean().reset_index().sort_values("estimated_loss",ascending=False)
            fig2 = px.bar(dl,x="district",y="estimated_loss",
                          title="Average Damage per Incident by District",
                          color="estimated_loss",color_continuous_scale="Oranges")
            fig2.update_layout(**pcfg(290)); st.plotly_chart(fig2,use_container_width=True)
            top_dl = dl.iloc[0]; bot_dl = dl.iloc[-1]
            insight(f"<b>{top_dl['district']}</b> has highest avg damage per incident "
                    f"(${top_dl['estimated_loss']:,.0f}), "
                    f"{round(top_dl['estimated_loss']/max(bot_dl['estimated_loss'],1),1)}× higher than "
                    f"<b>{bot_dl['district']}</b>.")

        sec("Financial Damage Overview — District × Crime Type")
        loss_bar = df.groupby(["district","crime_type"])["estimated_loss"].mean().reset_index()
        loss_bar.columns=["District","Crime Type","Avg Loss ($)"]
        fig3 = px.bar(loss_bar,x="District",y="Avg Loss ($)",color="Crime Type",barmode="group",
                      title="Average Financial Loss by District and Crime Type",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig3.update_layout(**pcfg(340),xaxis_tickangle=-15); st.plotly_chart(fig3,use_container_width=True)
        loss_heat_max = loss_bar.loc[loss_bar["Avg Loss ($)"].idxmax()]
        insight(f"Highest damage combo: <b>{loss_heat_max['District']}</b> + "
                f"<b>{loss_heat_max['Crime Type']}</b> — avg ${loss_heat_max['Avg Loss ($)']:,.0f}/incident.","warn")

    with t2:
        from sklearn.linear_model import LinearRegression as LR2
        Xq9 = df[["victim_count","suspect_count","crime_severity_enc","crime_type_enc"]]
        yq9 = df["investigation_hours"]
        Xq9t,_,yq9t,_ = train_test_split(Xq9,yq9,test_size=0.2,random_state=42)
        sc9=StandardScaler(); lr9=LR2(); lr9.fit(sc9.fit_transform(Xq9t),yq9t)
        cd9=pd.DataFrame({"Factor":["No. of Victims","No. of Suspects","Crime Severity","Crime Type"],
                          "Impact (hours)":np.abs(lr9.coef_)}).sort_values("Impact (hours)")
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(cd9,x="Impact (hours)",y="Factor",orientation="h",
                       title="Factors That Extend Investigation Duration",
                       color="Impact (hours)",color_continuous_scale="Blues")
            fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
            top_f2=cd9.iloc[-1]["Factor"]
            insight(f"<b>{top_f2}</b> is the strongest driver of investigation duration.","warn")
        with c2:
            fig2=px.scatter(df.sample(min(500,len(df))),x="victim_count",y="investigation_hours",
                            color="crime_severity",title="Victim Count vs Investigation Hours",
                            color_discrete_sequence=["#E24B4A","#EF9F27","#00C8FF"])
            fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
            insight("Cases with higher victim counts require significantly more investigation hours. "
                    "Multi-victim incidents should be assigned dedicated investigation teams.")

# ══════════════════════════════════════════
#  PAGE 7 — COMMAND DIRECTIVES
# ══════════════════════════════════════════
if st.session_state.page == "directives":
    st.markdown("<h2>Command Directives</h2>", unsafe_allow_html=True)
    filter_tag(R["crime"],R["district"],R["time"],R["similar_count"])

    rc2 = {"CRITICAL":"#E24B4A","HIGH":"#EF9F27","MODERATE":"#FAC775","LOW":"#1D9E75"}.get(R["risk_level"],"#888")
    st.markdown(f"""<div class='briefing'>
        <p><b style='color:#00C8FF;'>COMMAND DIRECTIVES — EVIDENCE-BASED</b><br>
        Active filter: <b>{R['crime']}</b> in <b>{R['district']}</b> during <b>{R['time']}</b> hours.
        Threat level: <b style='color:{rc2};'>{R['risk_level']}</b>.
        Directives auto-generated from <b>{len(df):,}</b> historical incidents.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        dm = df.groupby("district").size().reset_index(name="Total")
        da_rows=[]
        for d in df["district"].unique():
            sub=df[df["district"]==d]
            res_c=int(sub["arrest_made"].value_counts().get("Yes",0))
            res_pct=round(sub["arrest_made"].value_counts(normalize=True).get("Yes",0)*100,1)
            da_rows.append({"district":d,"Resolved":res_c,"Resolution %":res_pct})
        da=pd.DataFrame(da_rows)
        dm=dm.merge(da,on="district")
        dm["Unresolved"]=dm["Total"]-dm["Resolved"]
        fig=px.bar(dm,x="district",y=["Resolved","Unresolved"],barmode="stack",
                   title="Case Resolution Status by District",
                   color_discrete_sequence=["#1D9E75","#E24B4A"])
        fig.update_layout(**pcfg(270)); st.plotly_chart(fig,use_container_width=True)
        worst_dist=dm.loc[dm["Resolution %"].idxmin()]; best_dist=dm.loc[dm["Resolution %"].idxmax()]
        insight(f"<b>{worst_dist['district']}</b> lowest resolution ({worst_dist['Resolution %']}%). "
                f"<b>{best_dist['district']}</b> leads at {best_dist['Resolution %']}%.","warn")

        gd=df[df["gang_related"]=="Yes"].groupby("district").size().reset_index(name="Gang Incidents")
        fig2=px.bar(gd,x="district",y="Gang Incidents",title="Gang Activity Concentration by District",
                    color="Gang Incidents",color_continuous_scale="Reds")
        fig2.update_layout(**pcfg(270)); st.plotly_chart(fig2,use_container_width=True)
        insight(auto_conclude(gd.set_index("district")["Gang Incidents"],"gang_by_district"),"red")

    with c2:
        wd=df.groupby(["district","weapon_used"]).size().reset_index(name="Incidents")
        fig3=px.bar(wd,x="district",y="Incidents",color="weapon_used",barmode="stack",
                    title="Weapon Type Distribution by District",
                    color_discrete_sequence=px.colors.qualitative.Set1)
        fig3.update_layout(**pcfg(270)); st.plotly_chart(fig3,use_container_width=True)
        top_w_dist=wd.loc[wd["Incidents"].idxmax()]
        insight(f"<b>{top_w_dist['weapon_used']}</b> most prevalent in "
                f"<b>{top_w_dist['district']}</b>. Armed readiness elevated.","red")

        ev=df.groupby("evidence_collected")["arrest_made"].apply(
            lambda x: round(x.value_counts(normalize=True).get("Yes",0)*100,1)
        ).reset_index()
        ev.columns=["Evidence Collected","Resolution %"]
        fig4=px.bar(ev,x="Evidence Collected",y="Resolution %",
                    title="Evidence Collection Impact on Resolution",
                    color_discrete_sequence=["#E24B4A","#1D9E75"])
        fig4.update_layout(**pcfg(270)); st.plotly_chart(fig4,use_container_width=True)
        ev_y=ev[ev["Evidence Collected"]=="Yes"]["Resolution %"].values
        ev_n=ev[ev["Evidence Collected"]=="No"]["Resolution %"].values
        if len(ev_y)>0 and len(ev_n)>0:
            insight(f"Evidence collection lifts resolution by "
                    f"<b>{round(float(ev_y[0])-float(ev_n[0]),1)} percentage points</b>. "
                    f"Evidence training is highest-ROI intervention.","green")

    # ── Reporting Source vs Arrest Outcome ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("How Reporting Source Affects Case Resolution")
    q13_crime_sel = st.selectbox(
        "Select crime type",
        ["All"]+sorted(df["crime_type"].unique().tolist()),
        index=(["All"]+sorted(df["crime_type"].unique().tolist())).index(R["crime"])
        if R["crime"] in df["crime_type"].unique() else 0, key="q13_crime")
    q13_df = df if q13_crime_sel=="All" else df[df["crime_type"]==q13_crime_sel]

    c1_,c2_ = st.columns(2)
    with c1_:
        rep_arr = q13_df.groupby(["reported_by","arrest_made"]).size().reset_index(name="Incidents")
        fig_ra  = px.bar(rep_arr, x="reported_by", y="Incidents", color="arrest_made",
                         barmode="group",
                         title=f"Reporting Source vs Case Resolution — {q13_crime_sel}",
                         color_discrete_map={"Yes":"#1D9E75","No":"#E24B4A"},
                         labels={"reported_by":"Reported By","arrest_made":"Arrested"})
        fig_ra.update_layout(**pcfg(300), xaxis_tickangle=-25)
        st.plotly_chart(fig_ra, use_container_width=True)

    with c2_:
        rep_rate = q13_df.groupby("reported_by")["arrest_made"].apply(
            lambda x: round(x.value_counts(normalize=True).get("Yes",0)*100,1)
        ).reset_index()
        rep_rate.columns = ["Reported By","Resolution %"]
        rep_rate = rep_rate.sort_values("Resolution %", ascending=False)
        fig_rr = px.bar(rep_rate, x="Reported By", y="Resolution %",
                        title="Case Resolution Rate by Reporting Source",
                        color="Resolution %", color_continuous_scale="RdYlGn",
                        range_color=[0,100], text="Resolution %")
        fig_rr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rr.update_layout(**pcfg(300), xaxis_tickangle=-20)
        st.plotly_chart(fig_rr, use_container_width=True)

    if len(rep_rate)>0:
        best_rep  = rep_rate.iloc[0]; worst_rep = rep_rate.iloc[-1]
        insight(f"Cases reported by <b>{best_rep['Reported By']}</b> have the highest resolution rate "
                f"(<b>{best_rep['Resolution %']}%</b>), while <b>{worst_rep['Reported By']}</b>-reported "
                f"cases resolve at only <b>{worst_rep['Resolution %']}%</b>. "
                f"Anonymous and community reporting channels should be promoted to increase case solvability — "
                f"especially for under-reported crime types.", "green")

    # ── Intelligence-Driven Directives ───────────────────────────────────────
    st.markdown("<br><h2 style='text-align:center;color:#00C8FF !important;letter-spacing:2px;'>"
                "INTELLIGENCE-DRIVEN DIRECTIVES</h2><br>", unsafe_allow_html=True)

    directives = [
        ("Strengthen Patrol Density in High-Incident Zones","PRIORITY 1",
         f"Pattern analysis identifies {R['top_threat']} as dominant threat in {R['district']} district "
         f"({R['top_threat_pct']:.1f}% of incidents). Increase visible presence during {R['peak_time_in_area']} hours."),
        ("Target Evidence Collection in Every Incident","PRIORITY 1",
         "Data shows evidence collection is the single strongest predictor of case resolution. "
         "Officers must be trained and equipped to collect evidence at every scene."),
        ("Anti-Gang Task Force Deployment",
         "PRIORITY 1" if R["gang_pred"]=="Yes" else "PRIORITY 2",
         f"Gang involvement is {'CONFIRMED' if R['gang_pred']=='Yes' else 'possible'} for current filter. "
         f"Coordinate with specialist gang intelligence unit. Cross-reference known activity in {R['district']}."),
        ("Seasonal Resource Pre-Positioning","PRIORITY 2",
         "Crime volume shows strong seasonal patterns. Deploy additional units 2 weeks before peak months."),
        ("Weekend and Night Shift Reinforcement","PRIORITY 2",
         "Pattern analysis confirms crime spikes on weekends and during evening/night hours. "
         "Shift scheduling must prioritise these windows with experienced officers."),
        ("Community Intelligence Network","PRIORITY 2",
         f"Incidents reported directly by victims have higher resolution rates in {R['district']}. "
         "Invest in anonymous reporting channels and community liaison."),
        ("Data-Aligned Zone Realignment","PRIORITY 3",
         "Crime density analysis reveals high-incident areas that don't align with current patrol zones. "
         "Operational planning should redraw patrol zones to match actual crime density patterns."),
    ]
    for title,priority,desc in directives:
        pc = {"PRIORITY 1":"#E24B4A","PRIORITY 2":"#EF9F27","PRIORITY 3":"#00C8FF"}.get(priority,"#888")
        st.markdown(f"""
        <div class='action-card' style='border-left-color:{pc};margin-bottom:12px;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;'>
                <h4 style='color:{pc} !important;font-size:17px !important;'>{title}</h4>
                <span style='background:{pc}22;border:1px solid {pc};border-radius:20px;padding:3px 13px;
                    font-size:12px !important;color:{pc} !important;font-weight:700 !important;'>{priority}</span>
            </div>
            <p style='font-size:15px !important;'>{desc}</p>
        </div>""", unsafe_allow_html=True)