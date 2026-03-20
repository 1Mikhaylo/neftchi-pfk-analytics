import streamlit as st
import pandas as pd
import base64

st.set_page_config(
    page_title="Neftçi PFK Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

[data-testid="metric-container"] {
    background: #0B1220;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 18px 22px 14px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: rgba(56,189,248,0.18); }
[data-testid="stMetricLabel"] > div {
    font-size: 0.60rem !important; font-weight: 700 !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    color: #334155 !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 2.2rem !important; font-weight: 800 !important;
    color: #F1F5F9 !important; letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
}
hr { border-color: rgba(255,255,255,0.05) !important; margin: 1.5rem 0 !important; }
section[data-testid="stSidebar"] {
    background: #080C14;
    border-right: 1px solid rgba(255,255,255,0.04);
}
[data-testid="stPageLink"] a {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    padding: 8px 0 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: #94A3B8 !important;
    text-decoration: none !important;
    margin-top: 4px !important;
    transition: background 0.18s, border-color 0.18s, color 0.18s !important;
}
[data-testid="stPageLink"] a:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: #F1F5F9 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/neftchi_shots.csv")
    df["xgot"] = pd.to_numeric(df["xgot"], errors="coerce")
    return df

@st.cache_data
def load_logo_b64():
    with open("assets/neftchi_logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

df       = load_data()
logo_b64 = load_logo_b64()

# ── Stats ─────────────────────────────────────────────────────────────────────
total_matches = df["match"].nunique()
total_shots   = len(df)
total_goals   = int((df["shot_type"] == "goal").sum())
total_xg      = df["xg"].sum()
avg_xg_pm     = total_xg / total_matches if total_matches else 0
conversion    = total_goals / total_xg if total_xg > 0 else 0
sot           = df["shot_type"].isin(["goal", "save"]).sum()
sot_pct       = sot / total_shots * 100 if total_shots else 0

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;align-items:center;gap:28px;
            padding:32px 0 28px;border-bottom:1px solid rgba(255,255,255,0.05);
            margin-bottom:32px">
  <img src="data:image/png;base64,{logo_b64}" width="90"
       style="flex-shrink:0;filter:drop-shadow(0 0 18px rgba(56,189,248,0.18))">
  <div>
    <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.22em;
              text-transform:uppercase;color:#38BDF8;margin:0 0 6px">
      Premyer Liqa &nbsp;·&nbsp; 2025/26 Mövsümü
    </p>
    <h1 style="font-size:2.4rem;font-weight:800;color:#F1F5F9;
               letter-spacing:-0.03em;margin:0;line-height:1.05">
      Neftçi PFK
    </h1>
    <p style="font-size:0.82rem;color:#334155;margin:8px 0 0">
      Hücum performansı &nbsp;·&nbsp; xG analitikası &nbsp;·&nbsp; Oyunçu qiymətləndirməsi
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Oyun",          total_matches)
c2.metric("Zərbə",         total_shots)
c3.metric("Qol",           total_goals)
c4.metric("xG",            f"{total_xg:.1f}")
c5.metric("xG / oyun",     f"{avg_xg_pm:.2f}")
c6.metric("xG konversiya", f"{conversion:.2f}x", help="Qol ÷ xG")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<p style='font-size:0.58rem;font-weight:700;letter-spacing:0.18em;"
    "text-transform:uppercase;color:#334155;margin:0 0 18px'>Analiz Bölmələri</p>",
    unsafe_allow_html=True,
)

pages = [
    {
        "icon":  "⚽",
        "title": "Zərbə Xəritəsi",
        "path":  "pages/1_Zərbə_Xəritəsi.py",
        "desc":  "Hər zərbənin meydandakı yeri, istilik xəritəsi və keyfiyyət göstəriciləri. "
                 "Oyunçu, vəziyyət və oyun aralığına görə filtrləyin.",
        "color": "#F5C518",
        "tag":   "Meydança · İstilik · Filterlər",
    },
    {
        "icon":  "🎯",
        "title": "Finişinq Analizi",
        "path":  "pages/0_Finişinq_Analizi.py",
        "desc":  "Yaradılan şanslar nə qədər qola çevrilir? Hansı vəziyyətlər ən çox "
                 "israf olunur, kim şanslarını dəyərləndirir, kim boşa keçirir.",
        "color": "#38BDF8",
        "tag":   "Realizasiya · Oyunçu · İsraf",
    },
    {
        "icon":  "📊",
        "title": "Matç Performansı",
        "path":  "pages/2_Matç_Performansı.py",
        "desc":  "Oyun-başına xG və qol dinamikası, kumulativ momentum və "
                 "mövsüm boyunca komanda performansının dəyişim trendi.",
        "color": "#4ADE80",
        "tag":   "Oyun · Timeline · Trend",
    },
    {
        "icon":  "⚡",
        "title": "Aboubakar Effekti",
        "path":  "pages/3_Aboubakar_Effekti.py",
        "desc":  "Vincent Aboubakar transferindən əvvəl və sonra komanda hücumu necə dəyişdi? "
                 "Fərdi töhfə və komandanın ona asılılıq dərəcəsi.",
        "color": "#FB923C",
        "tag":   "Transfer · Əvvəl/Sonra · Asılılıq",
    },
]

cols = st.columns(4, gap="medium")
for col, p in zip(cols, pages):
    with col:
        st.markdown(f"""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-top:3px solid {p['color']};border-radius:10px;
     padding:22px 20px 14px;min-height:200px">
  <p style="font-size:1.35rem;margin:0 0 10px;color:{p['color']}">{p['icon']}</p>
  <p style="font-size:0.80rem;font-weight:700;color:#F1F5F9;
     letter-spacing:-0.01em;margin:0 0 8px">{p['title']}</p>
  <p style="font-size:0.78rem;color:#475569;line-height:1.65;margin:0 0 12px">
    {p['desc']}</p>
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
     text-transform:uppercase;color:#1E3A5F;margin:0 0 14px">{p['tag']}</p>
</div>
""", unsafe_allow_html=True)
        st.page_link(p["path"], label="Bölməyə keç →")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEASON SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<p style='font-size:0.58rem;font-weight:700;letter-spacing:0.18em;"
    "text-transform:uppercase;color:#334155;margin:0 0 18px'>Mövsüm Xülasəsi</p>",
    unsafe_allow_html=True,
)

season_diff  = total_goals - total_xg
diff_c       = "#4ADE80" if season_diff >= 0 else "#EF5350"
diff_verdict = (
    "Komanda xG-ni aşaraq potensialından yuxarı göstərici nümayiş etdirib."
    if season_diff > 0 else
    "Komanda yaratdığı şanslara baxmayaraq qol vurmaqda çətinlik çəkib."
    if season_diff < -1 else
    "Komanda xG-yə uyğun performans göstərib."
)

top_player_xg = df.groupby("player")["xg"].sum().sort_values(ascending=False)
top_name  = top_player_xg.index[0] if len(top_player_xg) else "—"
top_xg    = top_player_xg.iloc[0]  if len(top_player_xg) else 0
top_goals = int((df[df["player"] == top_name]["shot_type"] == "goal").sum())

s1, s2 = st.columns(2, gap="medium")

with s1:
    st.markdown(f"""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-left:3px solid #38BDF8;border-radius:10px;padding:20px 22px">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#1E3A5F;margin:0 0 12px">Mövsüm xG Balansı</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.85;margin:0">
    <strong style='color:#F1F5F9'>{total_matches} oyun</strong> boyunca
    <strong style='color:#F5C518'>{total_xg:.1f} xG</strong> yaradıldı,
    <strong style='color:#EF5350'>{total_goals} real qol</strong> vuruldu.<br>
    Fərq: <strong style='color:{diff_c}'>{season_diff:+.1f}</strong> —
    {diff_verdict}<br><br>
    Hədəfə düşən zərbə:
    <strong style='color:#F1F5F9'>{sot_pct:.0f}%</strong>
    ({int(sot)} / {total_shots}).
  </p>
</div>
""", unsafe_allow_html=True)

with s2:
    st.markdown(f"""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-left:3px solid #F5C518;border-radius:10px;padding:20px 22px">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#44300A;margin:0 0 12px">Ən Yüksək xG — Oyunçu</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.85;margin:0">
    <strong style='color:#F1F5F9'>{top_name}</strong> mövsümün ən yüksək
    xG-li oyunçusudur.<br>
    Ümumi xG: <strong style='color:#F5C518'>{top_xg:.2f}</strong>
    &nbsp;·&nbsp;
    Real qol: <strong style='color:#EF5350'>{top_goals}</strong><br>
    Konversiya: <strong style='color:#F1F5F9'>
    {(top_goals / top_xg):.2f}x</strong>
    {"— şanslarını effektiv reallaşdırır." if top_goals / top_xg > 1
     else "— şansların bir hissəsi boşa çıxır." if top_xg > 0 else ""}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:0.68rem;color:#1E293B;margin:0'>"
    "Data mənbəyi: Sofascore &nbsp;·&nbsp; Hazırladı: Mikayıl</p>",
    unsafe_allow_html=True,
)