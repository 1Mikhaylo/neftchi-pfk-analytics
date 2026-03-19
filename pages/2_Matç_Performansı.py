import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Match Performance · Neftchi", page_icon="⚽", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
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
[data-testid="metric-container"]:hover {
    border-color: rgba(56,189,248,0.18);
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.60rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #334155 !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 2.0rem !important;
    font-weight: 800 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
}
hr {
    border-color: rgba(255,255,255,0.05) !important;
    margin: 1.5rem 0 !important;
}
section[data-testid="stSidebar"] {
    background: #080C14;
    border-right: 1px solid rgba(255,255,255,0.04);
}
.section-label {
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #334155;
    margin: 0 0 14px 2px;
}
.insight-card {
    background: #0A0F1E;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 20px 22px;
}
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0D1B2A"
PLOT_BG = "#0B1220"
GRID    = "rgba(255,255,255,0.05)"
YELLOW  = "#F5C518"
RED     = "#EF5350"
GREEN   = "#4ADE80"
BLUE    = "#38BDF8"
ORANGE  = "#FB923C"
MUTED   = "#64748B"
TEXT    = "#94A3B8"

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/neftchi_shots.csv")
    df["xgot"] = pd.to_numeric(df["xgot"], errors="coerce")

    def parse_float_date(v):
        parts = str(v).split(".")
        day, month = int(parts[0]), int(parts[1])
        year = 2024 if month >= 8 else 2025
        return pd.Timestamp(year=year, month=month, day=day)

    df["date"] = df["date"].apply(parse_float_date)
    return df

df = load_data()

# ── Match ordering ────────────────────────────────────────────────────────────
match_meta = (
    df.groupby("match")
    .agg(season_order=("season_order", "first"), date=("date", "first"),
         rival=("rival", "first"), era=("era", "first"))
    .reset_index()
    .sort_values("season_order")
)
sorted_matches = match_meta["match"].tolist()

def short_label(m):
    teams = m.split(" - ")
    if "Neftchi" in teams[0] or "Neftçi" in teams[0]:
        opp = teams[1].split(" (")[0].strip()
        loc = "H"
    else:
        opp = teams[0].strip()
        loc = "A"
    abbrev = {
        "Samaxi": "SMX", "Imisli": "IML", "Turan Tovuz": "TRN",
        "Sabah": "SBH", "Sumqayit": "SMQ", "Karvan": "KRV",
        "Kapaz": "KPZ", "Qarabag": "QRB", "Zira": "ZIR",
        "Araz": "ARZ", "Gabala": "GBL",
    }
    opp_a = abbrev.get(opp, opp[:3].upper())
    row = match_meta[match_meta["match"] == m].iloc[0]
    return f"{row['date'].strftime('%d.%m')} {opp_a} ({loc})"

match_labels = [short_label(m) for m in sorted_matches]

# ── Per-match aggregation ─────────────────────────────────────────────────────
grp = (
    df.groupby(["match", "season_order"])
    .agg(
        xg      = ("xg",        "sum"),
        xgot    = ("xgot",      "sum"),
        goals   = ("shot_type", lambda x: (x == "goal").sum()),
        shots   = ("shot_type", "count"),
        on_tgt  = ("shot_type", lambda x: x.isin(["goal", "save"]).sum()),
        penalty = ("situation", lambda x: (x == "penalty").sum()),
        setpce  = ("situation", lambda x: x.isin(["set-piece", "free-kick",
                                                    "throw-in-set-piece"]).sum()),
        open_pl = ("situation", lambda x: (x == "regular").sum()),
    )
    .reset_index()
    .sort_values("season_order")
)
grp["label"]   = grp["match"].map(dict(zip(sorted_matches, match_labels)))
grp["conv"]    = (grp["goals"] / grp["xg"]).replace([np.inf, np.nan], 0)
grp["diff"]    = grp["goals"] - grp["xg"]
grp["sot_pct"] = (grp["on_tgt"] / grp["shots"].replace(0, np.nan) * 100).fillna(0)

grp["xg_cum"]     = grp["xg"].cumsum()
grp["goals_cum"]  = grp["goals"].cumsum()

n_m       = len(grp)
avg_xg    = round(grp["xg"].mean(), 2) if n_m else 0
avg_goals = round(grp["goals"].mean(), 2) if n_m else 0
avg_shots = round(grp["shots"].mean(), 1) if n_m else 0
avg_sot   = round(grp["sot_pct"].mean(), 1) if n_m else 0
conv_rate = round(grp["goals"].sum() / grp["xg"].sum(), 2) if grp["xg"].sum() > 0 else 0
xg_std    = round(grp["xg"].std(), 2) if n_m > 1 else 0

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:28px">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;
            color:#38BDF8;margin:0 0 6px">Neftchi FK &nbsp;·&nbsp; 2024/25 Premyer Liqa</p>
  <h1 style="font-size:2.2rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.03em;
             margin:0;line-height:1.05">Match Performance</h1>
  <p style="font-size:0.80rem;color:#334155;margin:8px 0 0">
    Oyunbaşı hücum effektivliyi — xG · xGOT · Qol dinamikası · Forma xətti
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Oyun",          n_m)
c2.metric("xG / oyun",     avg_xg)
c3.metric("Qol / oyun",    avg_goals)
c4.metric("Zərbə / oyun",  avg_shots)
c5.metric("Hədəfdə %",     f"{avg_sot:.0f}%",   help="Shots on target %")
c6.metric("xG konversiya", f"{conv_rate:.2f}x",  help="Goals ÷ total xG — >1 overperforming")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Match Timeline  (xG bar + xGOT bar + Goals line + Rolling)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">01 — Match Timeline · xG vs xGOT vs Qol</p>',
            unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;gap:24px;margin:-6px 0 16px;flex-wrap:wrap">
  <span style="font-size:0.74rem;color:#475569">
    <span style="color:#F5C518;font-weight:700">xG</span>
    &nbsp;— Gözlənilən qol: zərbənin mövqeyinə və növünə görə hesablanan qol ehtimalı
  </span>
  <span style="font-size:0.74rem;color:#475569">
    <span style="color:#FB923C;font-weight:700">xGOT</span>
    &nbsp;— Hədəfdəki gözlənilən qol: yalnız qapıçıya çatan zərbələr üçün, güc və bucağa görə
  </span>
</div>
""", unsafe_allow_html=True)

fig1 = go.Figure()

fig1.add_trace(go.Bar(
    x=grp["label"], y=grp["xg"].round(2),
    name="xG",
    marker_color=YELLOW, opacity=0.75,
))

fig1.add_trace(go.Bar(
    x=grp["label"], y=grp["xgot"].round(2),
    name="xGOT",
    marker_color=ORANGE, opacity=0.60,
))

fig1.add_trace(go.Scatter(
    x=grp["label"], y=grp["goals"],
    name="Qol",
    mode="markers+lines",
    marker=dict(color=RED, size=9, symbol="circle",
                line=dict(color="#FF8A80", width=1.5)),
    line=dict(color=RED, width=2, dash="dot"),
))

fig1.update_layout(
    paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Arial", color=TEXT, size=11),
    height=420,
    barmode="overlay",
    hovermode="x unified",
    legend=dict(
        orientation="h", x=0.5, xanchor="center", y=1.04, yanchor="bottom",
        font=dict(size=10.5, color=TEXT),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    margin=dict(t=40, b=150, l=44, r=20),
    xaxis=dict(showgrid=False, zeroline=False,
               tickangle=-42, tickfont=dict(size=9.5, color=MUTED)),
    yaxis=dict(
        title="xG / xGOT / Qol", title_font=dict(size=10, color=MUTED),
        showgrid=True, gridcolor=GRID, zeroline=False,
        tickfont=dict(size=9.5, color=MUTED),
    ),
)
st.plotly_chart(fig1, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — xG vs Goals scatter  +  Cumulative momentum
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">02 — Kumulativ Momentum</p>',
            unsafe_allow_html=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=grp["label"], y=grp["xg_cum"].round(2),
    name="Kumulativ xG",
    mode="lines",
    line=dict(color=YELLOW, width=2.2),
    fill="tozeroy",
    fillcolor="rgba(245,197,24,0.06)",
))
fig3.add_trace(go.Scatter(
    x=grp["label"], y=grp["goals_cum"],
    name="Kumulativ Qol",
    mode="lines+markers",
    marker=dict(color=RED, size=7),
    line=dict(color=RED, width=2, dash="dot"),
))
fig3.update_layout(
    paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Arial", color=TEXT, size=11),
    height=370,
    hovermode="x unified",
    legend=dict(
        orientation="h", x=0.5, xanchor="center", y=1.04, yanchor="bottom",
        font=dict(size=10, color=TEXT), bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    margin=dict(t=28, b=140, l=44, r=16),
    xaxis=dict(showgrid=False, zeroline=False,
               tickangle=-40, tickfont=dict(size=9, color=MUTED)),
    yaxis=dict(title="Kumulativ", title_font=dict(size=10, color=MUTED),
               showgrid=True, gridcolor=GRID, zeroline=False,
               tickfont=dict(size=9, color=MUTED)),
)
st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Analyst Notes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<p class="section-label">03 — Analitik Qiymətləndirmə</p>', unsafe_allow_html=True)

if n_m == 0:
    st.warning("Seçilmiş filterlərə uyğun oyun tapılmadı.")
else:
    # ── Compute all insight values ─────────────────────────────────────────────
    best    = grp.loc[grp["xg"].idxmax()]
    worst   = grp.loc[grp["xg"].idxmin()]
    most_e  = grp.loc[grp["diff"].idxmax()]
    least_e = grp.loc[grp["diff"].idxmin()]
    overp   = int((grp["diff"] > 0.3).sum())
    underp  = int((grp["diff"] < -0.3).sum())
    neutral = n_m - overp - underp

    total_xg    = grp["xg"].sum()
    total_goals = int(grp["goals"].sum())
    total_shots = int(grp["shots"].sum())
    total_sot   = int(grp["on_tgt"].sum())
    season_diff = total_goals - total_xg

    open_play_pct = grp["open_pl"].sum() / grp["shots"].sum() * 100 if grp["shots"].sum() > 0 else 0
    setpce_pct    = grp["setpce"].sum() / grp["shots"].sum() * 100 if grp["shots"].sum() > 0 else 0
    penalty_pct   = grp["penalty"].sum() / grp["shots"].sum() * 100 if grp["shots"].sum() > 0 else 0

    # xGOT vs xG ratio — how well shots are placed when on target
    xgot_total = grp["xgot"].sum()
    xgot_ratio = xgot_total / total_xg if total_xg > 0 else 0
    xgot_c     = GREEN if xgot_ratio > 1.05 else (RED if xgot_ratio < 0.85 else MUTED)

    # Trend: last 3 vs rest
    if n_m >= 4:
        last3_xg   = grp["xg"].iloc[-3:].mean()
        last3_goals= grp["goals"].iloc[-3:].mean()
        first_xg   = grp["xg"].iloc[:-3].mean()
        trend_dir  = "↑" if last3_xg > first_xg + 0.1 else ("↓" if last3_xg < first_xg - 0.1 else "→")
        trend_col  = GREEN if trend_dir == "↑" else (RED if trend_dir == "↓" else MUTED)
    else:
        last3_xg = last3_goals = first_xg = None
        trend_dir = "—"; trend_col = MUTED

    # Consistency
    xg_std_v   = grp["xg"].std() if n_m > 1 else 0
    consist_c  = GREEN if xg_std_v < 0.5 else (RED if xg_std_v > 1.0 else MUTED)
    consist_lbl= "Sabit" if xg_std_v < 0.5 else ("Dəyişkən" if xg_std_v > 1.0 else "Orta")

    # Season-level over/under verdict
    season_diff_c = GREEN if season_diff > 0 else (RED if season_diff < -1 else MUTED)
    season_verdict = (
        "Komanda xG-ni keçərək potensialını aşıb." if season_diff > 0
        else ("Komanda mövsüm boyu şanslara baxmayaraq qol vurmaqda çətinlik çəkib." if season_diff < -1
              else "Komanda xG-yə uyğun qol vurmağı bacarıb.")
    )

    # ── ROW 1: 3 cards ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        best_conv  = f"{best['goals'] / best['xg']:.2f}x" if best["xg"] > 0 else "—"
        worst_conv = f"{worst['goals'] / worst['xg']:.2f}x" if worst["xg"] > 0 else "—"
        st.markdown(f"""
<div class="insight-card" style="border-left:3px solid {BLUE}">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#1E3A5F;margin:0 0 14px">Ən Yaxşı · Ən Zəif Oyun</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#1E3A5F">Ən Güclü Hücum</span><br>
    <strong style='color:#F1F5F9'>{best["label"]}</strong><br>
    <strong style='color:{YELLOW}'>{best["xg"]:.2f} xG</strong> ilə mövsümün ən yüksək hücum təzyiqi.
    {int(best["goals"])} qol vuruldu (konversiya: {best_conv}).
    Bu oyunda komanda rəqib qapısını ciddi sıxışdırıb.<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#1E3A5F">Ən Zəif Hücum</span><br>
    <strong style='color:#F1F5F9'>{worst["label"]}</strong><br>
    Cəmi <strong style='color:{RED}'>{worst["xg"]:.2f} xG</strong> — hücum demək olar ki, mövcud olmayıb.
    {int(worst["goals"])} qol. Konversiya: {worst_conv}. Hücum bölməsi fəaliyyətsiz qalıb.
  </p>
</div>
""", unsafe_allow_html=True)

    with col2:
        eff_c   = GREEN if most_e["diff"] > 0 else MUTED
        ineff_c = RED if least_e["diff"] < -0.5 else MUTED
        most_e_conv  = f"{most_e['goals'] / most_e['xg']:.2f}x" if most_e["xg"] > 0 else "—"
        least_e_conv = f"{least_e['goals'] / least_e['xg']:.2f}x" if least_e["xg"] > 0 else "—"
        st.markdown(f"""
<div class="insight-card" style="border-left:3px solid {ORANGE}">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#44300A;margin:0 0 14px">Finişinq Effektivliyi</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#44300A">Ən Effektiv Oyun</span><br>
    <strong style='color:#F1F5F9'>{most_e["label"]}</strong><br>
    {most_e["xg"]:.2f} xG-dən <strong style='color:{eff_c}'>{int(most_e["goals"])} qol</strong>
    — gözləniləndən <span style='color:{eff_c}'>{most_e["diff"]:+.2f}</span> artıq.
    Konversiya {most_e_conv} ilə mövsümün zirvəsi. Hücumçular şansları dəqiqliklə reallaşdırıb.<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#44300A">Ən Çox İsraf Oyun</span><br>
    <strong style='color:#F1F5F9'>{least_e["label"]}</strong><br>
    {least_e["xg"]:.2f} xG yaradılmasına baxmayaraq cəmi
    <strong style='color:{ineff_c}'>{int(least_e["goals"])} qol</strong>
    (<span style='color:{ineff_c}'>{least_e["diff"]:.2f}</span>).
    Konversiya: {least_e_conv}. Şanslar boşa çıxıb — ciddi finişinq problemi.
  </p>
</div>
""", unsafe_allow_html=True)

    with col3:
        trend_body = (
            f"Son 3 oyunun orta xG-si <strong style='color:{trend_col}'>{last3_xg:.2f} {trend_dir}</strong>, "
            f"əvvəlki oyunların ortalaması isə {first_xg:.2f} idi. "
            + ("Hücum getdikcə güclənir — komanda forma tutur." if trend_dir == "↑"
               else ("Son oyunlarda hücum təzyiqi azalır — diqqət tələb olunur." if trend_dir == "↓"
                     else "Hücum gücü sabit qalır — böyük dəyişiklik yoxdur."))
        ) if last3_xg is not None else "Trend hesablamaq üçün kifayət qədər oyun yoxdur."

        st.markdown(f"""
<div class="insight-card" style="border-left:3px solid {GREEN}">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#14532D;margin:0 0 14px">Forma Trendi · Sabitlik</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#14532D">Forma Trendi</span><br>
    {trend_body}<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#14532D">Performans Sabitliyi</span><br>
    xG standart sapması <strong style='color:{consist_c}'>±{xg_std_v:.2f}</strong>
    — <strong style='color:{consist_c}'>{consist_lbl}</strong>.
    {"Komanda hər oyunda oxşar hücum keyfiyyəti nümayiş etdirir." if xg_std_v < 0.5
     else ("Oyundan oyuna böyük fərqlər var — sabitlik problemi mövcuddur." if xg_std_v > 1.0
           else "Performansda müəyyən dəyişkənlik var, lakin həddindən artıq deyil.")}
  </p>
</div>
""", unsafe_allow_html=True)

    # ── ROW 2: 2 wide cards ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    col4, col5 = st.columns(2, gap="medium")

    with col4:
        open_c   = GREEN if open_play_pct > 65 else MUTED
        setpce_c = GREEN if setpce_pct > 20 else MUTED
        st.markdown(f"""
<div class="insight-card" style="border-left:3px solid #A78BFA">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#2D1B6E;margin:0 0 14px">Hücum Mənbəyi Analizi</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    Mövsüm boyu <strong style='color:{open_c}'>{open_play_pct:.0f}%</strong> zərbə
    açıq oyundan, <strong style='color:{setpce_c}'>{setpce_pct:.0f}%</strong> standart
    vəziyyətdən (sərbəst zərbə, künc, kənara atış), <strong style='color:{RED}'>{penalty_pct:.0f}%</strong>
    isə penaltidən gəlib.<br><br>
    {"Komanda əsasən açıq oyun vasitəsilə şans yaradır — dinamik, kombinasiyalı hücum üslubu." if open_play_pct > 65
     else "Açıq oyun zərbəsi azdır — komanda standart vəziyyətlərə həddindən çox güvənir."}<br>
    {"Standart vəziyyətlər ciddi silah — topu saxlayan komanda üçün əhəmiyyətli üstünlük." if setpce_pct > 20 else ""}
  </p>
</div>
""", unsafe_allow_html=True)

    with col5:
        sot_pct_v = total_sot / total_shots * 100 if total_shots > 0 else 0
        sot_c     = GREEN if sot_pct_v > 40 else (RED if sot_pct_v < 28 else MUTED)
        st.markdown(f"""
<div class="insight-card" style="border-left:3px solid #F472B6">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#5B1638;margin:0 0 14px">Mövsüm Ümumiləşdirməsi</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    {n_m} oyunda <strong style='color:#F1F5F9'>{total_shots}</strong> zərbə atılıb,
    bunların <strong style='color:{sot_c}'>{sot_pct_v:.0f}%</strong>-i ({total_sot}) hədəfə düşüb.
    Ümumi xG <strong style='color:{YELLOW}'>{total_xg:.1f}</strong>, real qol
    <strong style='color:{RED}'>{total_goals}</strong>
    (<span style='color:{season_diff_c}'>{season_diff:+.1f}</span>).<br><br>
    {season_verdict}<br>
    xGOT/xG nisbəti <strong style='color:{xgot_c}'>{xgot_ratio:.2f}</strong> —
    {"hədəfə düşən zərbələr zəif bucaqlardan gəlir, qapıçı üçün asan." if xgot_ratio < 0.85
     else ("Zərbələr qapıçını çətin vəziyyətə salır — yüksək keyfiyyətli atışlar." if xgot_ratio > 1.05
           else "Zərbələrin hədəf keyfiyyəti orta səviyyədədir.")}
  </p>
</div>
""", unsafe_allow_html=True)
