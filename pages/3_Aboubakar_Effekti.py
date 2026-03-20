import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Aboubakar Impact · Neftchi", page_icon="⚽", layout="wide")

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
    font-size: 2.0rem !important; font-weight: 800 !important;
    color: #F1F5F9 !important; letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.72rem !important; font-weight: 600 !important; }
hr { border-color: rgba(255,255,255,0.05) !important; margin: 1.5rem 0 !important; }
section[data-testid="stSidebar"] {
    background: #080C14;
    border-right: 1px solid rgba(255,255,255,0.04);
}
.section-label {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #334155; margin: 0 0 14px 2px;
}
.insight-card {
    background: #0A0F1E;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 20px 22px;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0D1B2A"
PLOT_BG  = "#0B1220"
GRID     = "rgba(255,255,255,0.05)"
C_BEFORE = "#38BDF8"
C_WITH   = "#F5C518"
GREEN    = "#4ADE80"
RED      = "#EF5350"
ORANGE   = "#FB923C"
MUTED    = "#64748B"
TEXT     = "#94A3B8"

ERA_LABELS = {"before": "Aboubakar öncəsi", "with": "Aboubakarla"}
ERA_COLORS = {"before": C_BEFORE, "with": C_WITH}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/neftchi_shots.csv")
    def parse_float_date(v):
        p = str(v).split(".")
        d, m = int(p[0]), int(p[1])
        return pd.Timestamp(year=(2024 if m >= 8 else 2025), month=m, day=d)
    df["date"] = df["date"].apply(parse_float_date)
    return df

df = load_data()

# ── Era tagging ───────────────────────────────────────────────────────────────
ABOU_FIRST  = 1026
ABOU_ABSENT = {1410}  # Match where he didn't play (shot data alone can't detect this —
                      # a player can play without taking a shot)

df["era"] = df["season_order"].apply(
    lambda so: None if so in ABOU_ABSENT
    else ("with" if so >= ABOU_FIRST else "before")
)
df_plot = df[df["era"].notna()].copy()

# Absent match name for display
absent_labels = (
    df[df["season_order"].isin(ABOU_ABSENT)]["match"]
    .drop_duplicates()
    .str.replace(r"\s*\(.*\)", "", regex=True)
    .str.strip()
    .tolist()
)  # season_order 1410 → 1 match excluded

# ── Match-level aggregation ───────────────────────────────────────────────────
match_stats = (
    df_plot.groupby(["match", "season_order", "era"])
    .agg(
        xg    = ("xg",        "sum"),
        zerbe = ("xg",        "count"),
        qol   = ("shot_type", lambda x: (x == "goal").sum()),
    )
    .reset_index()
    .sort_values("season_order")
    .reset_index(drop=True)
)

# Preserve date in label to differentiate duplicate opponents
match_stats["label"] = match_stats["match"].str.replace(
    r"^(.*?)\s*\((\d+\.\d+)\)$",
    lambda m: f"{m.group(1).strip()} ({m.group(2)})",
    regex=True,
)
all_labels = match_stats["label"].tolist()

# Aboubakar personal xG per match
abou_per_match = (
    df_plot[df_plot["player"].str.contains("Aboubakar", na=False) & (df_plot["era"] == "with")]
    .groupby("match")
    .agg(abou_xg=("xg", "sum"), abou_shots=("xg", "count"))
    .reset_index()
)
match_stats = match_stats.merge(abou_per_match, on="match", how="left")
match_stats["abou_xg"]    = match_stats["abou_xg"].fillna(0)
match_stats["abou_shots"] = match_stats["abou_shots"].fillna(0)

# ── Era averages ──────────────────────────────────────────────────────────────
def era_avg(era_name):
    g = match_stats[match_stats["era"] == era_name]
    n = len(g)
    if n == 0:
        return {"n": 0, "xg_pm": 0, "shots_pm": 0, "xg_pshot": 0, "goals_pm": 0, "eff": 0}
    return {
        "n"        : n,
        "xg_pm"    : round(g["xg"].mean(), 2),
        "shots_pm" : round(g["zerbe"].mean(), 1),
        "xg_pshot" : round(g["xg"].sum() / g["zerbe"].sum(), 3),
        "goals_pm" : round(g["qol"].mean(), 2),
        "eff"      : round(g["qol"].sum() / g["xg"].sum(), 2) if g["xg"].sum() > 0 else 0,
    }

eras = {k: era_avg(k) for k in ["before", "with"]}

before_games   = eras["before"]["n"]
with_games     = eras["with"]["n"]
xg_change      = eras["with"]["xg_pm"]    - eras["before"]["xg_pm"]
quality_change = eras["with"]["xg_pshot"] - eras["before"]["xg_pshot"]
xg_pct         = (xg_change / eras["before"]["xg_pm"] * 100)      if eras["before"]["xg_pm"]    > 0 else 0
q_pct          = (quality_change / eras["before"]["xg_pshot"] * 100) if eras["before"]["xg_pshot"] > 0 else 0

total_xg_with  = match_stats[match_stats["era"] == "with"]["xg"].sum()
abou_total_xg  = match_stats[match_stats["era"] == "with"]["abou_xg"].sum()
abou_share     = (abou_total_xg / total_xg_with * 100) if total_xg_with > 0 else 0
abou_xg_pm     = abou_total_xg / with_games if with_games > 0 else 0

xg_improvement      = xg_pct
quality_improvement = q_pct

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:24px">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;
            color:#38BDF8;margin:0 0 6px">Neftchi FK &nbsp;·&nbsp; 2025/26 Premyer Liqa</p>
  <h1 style="font-size:2.2rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.03em;
             margin:0;line-height:1.05">Vincent Aboubakar Təsiri</h1>
  <p style="font-size:0.80rem;color:#334155;margin:8px 0 0">
    Kamerunlu hücumçunun komanda hücum performansına statistik təsir analizi
  </p>
</div>
""", unsafe_allow_html=True)

if absent_labels:
    absent_str = ", ".join(f"<strong style='color:#F1F5F9'>{m}</strong>" for m in absent_labels)
    st.markdown(
        f"<p style='font-size:0.72rem;color:#334155;margin:-8px 0 20px'>"
        f"Buraxılan oyun (iştirak etmədi): {absent_str} — analiz hər iki eradan çıxarılıb</p>",
        unsafe_allow_html=True,
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Öncəsi",          f"{before_games} oyun")
c2.metric("Aboubakarla",     f"{with_games} oyun")
c3.metric("xG/oyun dəyişim", f"{xg_change:+.2f}",      delta=f"{xg_pct:+.1f}%")
c4.metric("Zərbə keyfiyyəti",f"{quality_change:+.3f}",  delta=f"{q_pct:+.1f}%")
c5.metric("Aboubakar payı",  f"{abou_share:.0f}%",      help="Onun oyunlarında komanda xG-nin neçə faizi")
c6.metric("Aboubakar xG/m",  f"{abou_xg_pm:.2f}",       help="Oyun başına fərdi xG")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Era comparison bars
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">01 — Era Müqayisəsi · Ortalama Göstəricilər</p>',
            unsafe_allow_html=True)

metrics_cfg = [
    ("xg_pm",    "xG / oyun"),
    ("shots_pm", "Zərbə / oyun"),
    ("xg_pshot", "xG / zərbə"),
]

fig_bars = make_subplots(
    rows=1, cols=3,
    subplot_titles=[m[1] for m in metrics_cfg],
    horizontal_spacing=0.10,
)

for col_idx, (metric, title) in enumerate(metrics_cfg, start=1):
    for era_key in ["before", "with"]:
        val = eras[era_key][metric]
        fig_bars.add_trace(go.Bar(
            x=[ERA_LABELS[era_key]], y=[val],
            name=ERA_LABELS[era_key],
            legendgroup=era_key,
            showlegend=(col_idx == 1),
            marker_color=ERA_COLORS[era_key],
            opacity=0.82,
            text=[f"<b>{val}</b>"],
            textposition="outside",
            textfont=dict(color=TEXT, size=12, family="Arial"),
            cliponaxis=False,
            width=0.42,
            hovertemplate=(
                f"<b>{ERA_LABELS[era_key]}</b><br>"
                f"{title}: <b>%{{y}}</b><br>"
                f"n = {eras[era_key]['n']} oyun<extra></extra>"
            ),
        ), row=1, col=col_idx)

fig_bars.update_layout(
    paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Arial", color=TEXT, size=11),
    height=320,
    margin=dict(t=44, b=16, l=20, r=20),
    legend=dict(
        orientation="h", x=0.5, xanchor="center", y=-0.14,
        font=dict(size=11, color=TEXT),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
)
for i in range(1, 4):
    ax = "" if i == 1 else str(i)
    fig_bars.update_layout(**{
        f"xaxis{ax}": dict(showgrid=False, zeroline=False,
                           tickfont=dict(size=10, color=MUTED)),
        f"yaxis{ax}": dict(showgrid=True, gridcolor=GRID, zeroline=False,
                           tickfont=dict(size=10, color=MUTED)),
    })
fig_bars.update_annotations(font=dict(size=10, color=MUTED, family="Arial"))
st.plotly_chart(fig_bars, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — xG Timeline
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">02 — Oyunbaşı xG Dinamikası · Zaman Xətti</p>',
            unsafe_allow_html=True)

split_idx = match_stats[match_stats["era"] == "with"].index[0]

fig_timeline = go.Figure()

# Aboubakar xG background bars
abou_bar_y = [
    row["abou_xg"] if row["era"] == "with" else None
    for _, row in match_stats.iterrows()
]
fig_timeline.add_trace(go.Bar(
    x=all_labels, y=abou_bar_y,
    name="Aboubakar xG payı",
    marker_color="rgba(245,197,24,0.12)",
    marker_line=dict(color="rgba(245,197,24,0.28)", width=1),
    hovertemplate="<b>Aboubakar xG</b>: %{y:.2f}<extra></extra>",
))

# Two line segments connected at split point
before_idx = list(match_stats[match_stats["era"] == "before"].index) + [split_idx]
with_idx   = [split_idx] + list(match_stats[match_stats["era"] == "with"].index)

for era_key, idx_list in [("before", before_idx), ("with", with_idx)]:
    ys, xs, hovers = [], [], []
    for i, row in match_stats.iterrows():
        if i in idx_list:
            xs.append(row["label"])
            ys.append(row["xg"])
            hovers.append(
                f"<b>{row['label']}</b><br>"
                f"─────────────────────<br>"
                f"xG: <b>{row['xg']:.2f}</b><br>"
                f"Zərbə: {int(row['zerbe'])} &nbsp;·&nbsp; Qol: {int(row['qol'])}<br>"
                f"Aboubakar xG: {row['abou_xg']:.2f}"
            )
    fig_timeline.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+lines",
        name=ERA_LABELS[era_key],
        marker=dict(color=ERA_COLORS[era_key], size=8,
                    line=dict(color="rgba(255,255,255,0.15)", width=1)),
        line=dict(color=ERA_COLORS[era_key], width=2.2),
        text=hovers,
        hovertemplate="%{text}<extra></extra>",
    ))

# Vertical transition line
fig_timeline.add_vline(
    x=split_idx - 0.5,
    line=dict(color="rgba(245,197,24,0.40)", width=1.2, dash="dash"),
    annotation_text="Aboubakar gəldi ▶",
    annotation_position="top",
    annotation_font=dict(color="rgba(245,197,24,0.65)", size=10),
)

# Era average reference lines
for era_key in ["before", "with"]:
    mean_xg = match_stats[match_stats["era"] == era_key]["xg"].mean()
    fig_timeline.add_hline(
        y=mean_xg,
        line=dict(color=ERA_COLORS[era_key], width=0.9, dash="dot"),
        annotation_text=f"ort. {mean_xg:.2f}",
        annotation_font=dict(color=ERA_COLORS[era_key], size=9),
        annotation_position="bottom right" if era_key == "with" else "bottom left",
    )

fig_timeline.update_layout(
    paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Arial", color=TEXT, size=11),
    height=460,
    barmode="overlay",
    hovermode="closest",
    legend=dict(
        orientation="h", x=0.5, xanchor="center", y=1.04, yanchor="bottom",
        font=dict(size=11, color=TEXT),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    margin=dict(t=40, b=130, l=44, r=20),
    xaxis=dict(showgrid=False, zeroline=False, tickangle=-40,
               tickfont=dict(size=8.5, color=MUTED)),
    yaxis=dict(title="xG", title_font=dict(size=10, color=MUTED),
               showgrid=True, gridcolor=GRID, zeroline=False,
               tickfont=dict(size=10, color=MUTED)),
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Analyst Notes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<p class="section-label">03 — Analitik Qiymətləndirmə</p>', unsafe_allow_html=True)

xg_dir_c  = GREEN if xg_change >= 0 else RED
q_dir_c   = GREEN if quality_change >= 0 else RED
dep_c     = RED if abou_share > 40 else (ORANGE if abou_share > 28 else GREEN)
eff_diff  = eras["with"]["eff"] - eras["before"]["eff"]
eff_dir_c = GREEN if eff_diff >= 0 else RED

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown(f"""
<div class="insight-card" style="border-left:3px solid {C_BEFORE}">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#1E3A5F;margin:0 0 14px">Komanda Performansı</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#1E3A5F">xG / Oyun</span><br>
    Öncəsi <strong style='color:#F1F5F9'>{eras["before"]["xg_pm"]}</strong>
    &nbsp;→&nbsp;
    Aboubakarla <strong style='color:{C_WITH}'>{eras["with"]["xg_pm"]}</strong>
    &nbsp;<span style='color:{xg_dir_c}'>({xg_change:+.2f}, {xg_improvement:+.1f}%)</span><br>
    {"Hücum gücü əhəmiyyətli dərəcədə artıb." if xg_change > 0.2
     else ("Müəyyən artım var." if xg_change > 0 else "Aboubakar gəlişinə baxmayaraq xG azalıb.")}<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#1E3A5F">Zərbə Keyfiyyəti (xG/zərbə)</span><br>
    {eras["before"]["xg_pshot"]:.3f} → <strong style='color:{C_WITH}'>{eras["with"]["xg_pshot"]:.3f}</strong>
    &nbsp;<span style='color:{q_dir_c}'>({quality_change:+.3f}, {quality_improvement:+.1f}%)</span><br>
    {"Daha yaxşı mövqelərdən zərbə vurulur — keyfiyyət artıb." if quality_change > 0
     else "Zərbə mövqeyi zəifləyib."}<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#1E3A5F">Finişinq (xG konversiya)</span><br>
    {eras["before"]["eff"]:.2f}x → <strong style='color:{C_WITH}'>{eras["with"]["eff"]:.2f}x</strong>
    &nbsp;<span style='color:{eff_dir_c}'>({eff_diff:+.2f})</span><br>
    {"Şanslar daha effektiv şəkildə qola çevrilir." if eff_diff > 0 else "Finişinq effektivliyi azalıb."}
  </p>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
<div class="insight-card" style="border-left:3px solid {ORANGE}">
  <p style="font-size:0.58rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#44300A;margin:0 0 14px">Aboubakar Asılılığı</p>
  <p style="font-size:0.84rem;color:#64748B;line-height:1.9;margin:0">
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#44300A">xG Payı</span><br>
    İştirak etdiyi <strong style='color:#F1F5F9'>{with_games} oyunda</strong> komanda
    xG-sinin <strong style='color:{dep_c}'>{abou_share:.1f}%</strong>-ini yaradıb.<br>
    {"Yüksək asılılıq — Aboubakar olmadan hücum ciddi zəifləyə bilər." if abou_share > 40
     else ("Mütəvazit asılılıq — komanda digər oyunçulara da güvənir." if abou_share > 28
           else "Balanslaşdırılmış — hücum yalnız ona bağlı deyil.")}<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#44300A">Fərdi Göstərici</span><br>
    Oyun başına <strong style='color:{C_WITH}'>{abou_xg_pm:.2f} xG</strong> —
    komandanın ən məhsuldar hücumçusu.<br>
    Ümumi <strong style='color:{C_WITH}'>{abou_total_xg:.1f} xG</strong>
    {with_games} oyunda.<br><br>
    <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.1em;color:#44300A">Zərbə Aktivliyi</span><br>
    Oyun başına <strong style='color:#F1F5F9'>{eras["with"]["shots_pm"]:.1f}</strong> zərbə
    (öncəsi: {eras["before"]["shots_pm"]:.1f}) —
    {"hücum intensivliyi artıb." if eras["with"]["shots_pm"] > eras["before"]["shots_pm"]
     else "hücum intensivliyi azalıb."}
  </p>
</div>
""", unsafe_allow_html=True)
