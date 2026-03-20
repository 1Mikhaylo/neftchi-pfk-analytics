import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.set_page_config(page_title="Shot Map · Neftchi", page_icon="⚽", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

[data-testid="metric-container"] {
    background: #0F172A;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 20px 24px 16px;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #3D5068 !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
}
hr {
    border-color: rgba(255,255,255,0.05) !important;
    margin: 1.5rem 0 !important;
}
section[data-testid="stSidebar"] {
    background: #080C14;
    border-right: 1px solid rgba(255,255,255,0.04);
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/neftchi_shots.csv")
    df["xgot"] = pd.to_numeric(df["xgot"], errors="coerce")

    def parse_float_date(float_date):
        day_month = str(float_date).split(".")
        day   = int(day_month[0])
        month = int(day_month[1])
        year  = 2024 if month >= 8 else 2025
        return pd.Timestamp(year=year, month=month, day=day)

    df["date"] = df["date"].apply(parse_float_date)
    df['player'] = df['player'].replace({'M. Mammadov': 'M. Məmmədov'})

    return df

def calc_distance(row):
    x = row["pitch_x"]   # Width  (0–68)
    y = row["pitch_y"]   # Depth from goal line (0–105, but shots typically 0–55)
    return np.sqrt((34 - x) ** 2 + y ** 2)

def format_match_label(match_name, match_date):
    teams = match_name.split(" - ")
    if "Neftchi" in teams[0] or "Neftçi" in teams[0]:
        opponent = teams[1].split(" (")[0].replace("_", " ")
        loc = "🏠"
    else:
        opponent = teams[0].replace("_", " ")
        loc = "✈️"
    abbrev = {
        "Samaxi": "SMX", "Imisli": "IML", "Turan Tovuz": "TRN",
        "Sabah": "SBH", "Sumqayit": "SMQ", "Karvan": "KRV",
        "Kapaz": "KPZ", "Qarabag": "QRB", "Zira": "ZIR",
        "Araz": "ARZ", "Gabala": "GBL",
    }
    opp_abbr = abbrev.get(opponent, opponent[:3].upper())
    return f"{match_date.strftime('%d.%m')}  {opp_abbr} ({loc})"

df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(
    "<p style='font-size:0.62rem;font-weight:700;letter-spacing:0.16em;"
    "text-transform:uppercase;color:#334155;margin-bottom:16px'>Filterlər</p>",
    unsafe_allow_html=True,
)

all_players = sorted(df["player"].unique())
selected_players = st.sidebar.multiselect(
    "Oyunçular", options=all_players, default=None, placeholder="Bütün oyunçular"
)

all_situations = sorted(df["situation"].unique())
selected_situations = st.sidebar.multiselect(
    "Vəziyyət", options=all_situations, default=None, placeholder="Bütün vəziyyətlər"
)

match_dates   = df.groupby("match")["date"].first().sort_values()
sorted_matches = match_dates.index.tolist()
match_labels  = [format_match_label(m, match_dates[m]) for m in sorted_matches]

match_range = st.sidebar.select_slider(
    "Oyun aralığı",
    options=range(len(sorted_matches)),
    value=(0, len(sorted_matches) - 1),
    format_func=lambda x: match_labels[x],
)

st.sidebar.markdown(
    "<p style='font-size:0.62rem;font-weight:700;letter-spacing:0.14em;"
    "text-transform:uppercase;color:#334155;margin:16px 0 8px'>Nəticə növü</p>",
    unsafe_allow_html=True,
)
show_goal  = st.sidebar.checkbox("Qol",          value=True)
show_save  = st.sidebar.checkbox("Qapıçı tutdu", value=True)
show_miss  = st.sidebar.checkbox("Qeyri-dəqiq",   value=True)
show_block = st.sidebar.checkbox("Bloklandı",    value=True)
show_post  = st.sidebar.checkbox("Dirək",         value=True)

outcome_filter = []
if show_goal:  outcome_filter.append("goal")
if show_save:  outcome_filter.append("save")
if show_miss:  outcome_filter.append("miss")
if show_block: outcome_filter.append("block")
if show_post:  outcome_filter.append("post")

# ── Apply filters ─────────────────────────────────────────────────────────────
selected_matches = sorted_matches[match_range[0]: match_range[1] + 1]
df_filtered = df[df["match"].isin(selected_matches)].copy()
if selected_players:
    df_filtered = df_filtered[df_filtered["player"].isin(selected_players)]
if selected_situations:
    df_filtered = df_filtered[df_filtered["situation"].isin(selected_situations)]
if outcome_filter:
    df_filtered = df_filtered[df_filtered["shot_type"].isin(outcome_filter)]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<p style='font-size:0.72rem;color:#334155'>"
    f"{len(df_filtered)} zərbə göstərilir &nbsp;·&nbsp; {df['match'].nunique()} oyun</p>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:28px">
  <p style="font-size:0.62rem;font-weight:700;letter-spacing:0.20em;text-transform:uppercase;
            color:#38BDF8;margin:0 0 6px">Neftchi FK &nbsp;·&nbsp; 2024/25 Premyer Liqa</p>
  <h1 style="font-size:2.1rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.03em;
             margin:0;line-height:1.05">Shot Map</h1>
  <p style="font-size:0.82rem;color:#3D5068;margin:8px 0 0">
    Zərbələrin meydandakı nöqtələri &amp; keyfiyyət analizi
  </p>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
n = len(df_filtered)
total_xg     = round(df_filtered["xg"].sum(), 2) if n else 0
total_goals  = int((df_filtered["shot_type"] == "goal").sum()) if n else 0
avg_distance = round(df_filtered.apply(calc_distance, axis=1).mean(), 1) if n else 0
conversion   = round(total_goals / total_xg, 2) if total_xg else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Zərbə",        n)
c2.metric("xG",           total_xg)
c3.metric("Qol",          total_goals)
c4.metric("Konversiya",   f"{conversion:.2f}", help="Qol / xG")
c5.metric("Orta məsafə",  f"{avg_distance}m",  help="Qapıdan orta məsafə")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SHOT MAP  —  half-pitch view (0–52.5 m) + overflow for rare long shots
# ══════════════════════════════════════════════════════════════════════════════
if n == 0:
    st.warning("Seçilmiş filterlərə uyğun zərbə tapılmadı.")
else:
    # Palette
    LC  = "rgba(255,255,255,0.38)"
    LC2 = "rgba(255,255,255,0.12)"
    LW  = 1.3

    STYLES = {
        "goal":  {"face": "#F59E0B", "edge": "#F59E0B", "label": "Qol",          "size_mult": 1.05},
        "save":  {"face": "#38BDF8", "edge": "#38BDF8", "label": "Qapıçı tutdu", "size_mult": 0.72},
        "miss":  {"face": "rgba(0,0,0,0)", "edge": "#A78BFA", "label": "Qeyri-dəqiq", "size_mult": 0.65},
        "block": {"face": "rgba(0,0,0,0)", "edge": "#64748B", "label": "Bloklandı",  "size_mult": 0.65},
        "post":  {"face": "rgba(0,0,0,0)", "edge": "#FB923C", "label": "Dirək",       "size_mult": 0.65},
    }

    def xg_to_size(xg, mult=1.0):
        return max(6, min(22, np.sqrt(xg) * 68)) * mult

    fig = go.Figure()

    # ── Pitch: grass stripes ──────────────────────────────────────────────────
    STRIPE_H = 5.25   # 10 bands across 52.5 m
    for i in range(10):
        if i % 2 == 0:
            fig.add_shape(
                type="rect", x0=0, y0=i * STRIPE_H, x1=68, y1=(i + 1) * STRIPE_H,
                fillcolor="rgba(255,255,255,0.025)", line=dict(width=0), layer="below",
            )

    # ── Pitch line constants ──────────────────────────────────────────────────
    PL = dict(color=LC, width=LW)

    # Outer boundary (half-pitch)
    fig.add_shape(type="rect", x0=0, y0=0, x1=68, y1=52.5,
                  fillcolor="rgba(0,0,0,0)", line=PL)

    # Halfway line (dashed)
    fig.add_shape(type="line", x0=0, y0=52.5, x1=68, y1=52.5,
                  line=dict(color=LC2, width=0.9, dash="dot"))

    # Penalty area  13.84–54.16 m wide, 0–16.5 m deep
    fig.add_shape(type="rect", x0=13.84, y0=0, x1=54.16, y1=16.5,
                  fillcolor="rgba(255,255,255,0.018)", line=PL)

    # Goal area  24.84–43.16 m wide, 0–5.5 m deep
    fig.add_shape(type="rect", x0=24.84, y0=0, x1=43.16, y1=5.5,
                  fillcolor="rgba(0,0,0,0)", line=PL)

    # Goal frame  30.34–37.66 m wide, -2.44–0 m
    fig.add_shape(type="rect", x0=30.34, y0=-2.44, x1=37.66, y1=0,
                  fillcolor="rgba(255,255,255,0.06)",
                  line=dict(color="rgba(255,255,255,0.70)", width=2.2))

    for gx in [30.34, 37.66]:
        fig.add_shape(type="line", x0=gx, y0=-2.44, x1=gx, y1=0,
                      line=dict(color="rgba(255,255,255,0.85)", width=2.5))

    fig.add_shape(type="line", x0=30.34, y0=0, x1=37.66, y1=0,
                  line=dict(color="rgba(255,255,255,0.85)", width=2.5))

    # Penalty spot
    fig.add_trace(go.Scatter(
        x=[34], y=[11.0], mode="markers",
        marker=dict(color=LC, size=4.5, symbol="circle"),
        showlegend=False, hoverinfo="skip",
    ))

    # Penalty arc
    theta = np.linspace(np.pi * 0.27, np.pi * 0.73, 120)
    arc_x = 34 + 9.15 * np.cos(theta)
    arc_y = 11.0 + 9.15 * np.sin(theta)
    mask_arc = arc_y > 16.5
    fig.add_trace(go.Scatter(
        x=arc_x[mask_arc], y=arc_y[mask_arc], mode="lines",
        line=dict(color=LC, width=LW), showlegend=False, hoverinfo="skip",
    ))

    # Corner arcs
    for cx, t0, t1 in [(0, 0, np.pi / 2), (68, np.pi / 2, np.pi)]:
        t = np.linspace(t0, t1, 40)
        fig.add_trace(go.Scatter(
            x=cx + np.cos(t), y=np.sin(t), mode="lines",
            line=dict(color=LC, width=LW), showlegend=False, hoverinfo="skip",
        ))

    # Centre circle arc
    theta_c = np.linspace(0, np.pi, 120)
    cx_c = 34 + 9.15 * np.cos(theta_c)
    cy_c = 52.5 + 9.15 * np.sin(theta_c)
    fig.add_trace(go.Scatter(
        x=cx_c, y=cy_c, mode="lines",
        line=dict(color=LC2, width=0.9), showlegend=False, hoverinfo="skip",
    ))

    # ── KDE heatmap ───────────────────────────────────────────────────────────
    if n >= 5:
        xs = df_filtered["pitch_x"].values
        ys = df_filtered["pitch_y"].values
        ws = df_filtered["xg"].values

        # Clip to half-pitch for heatmap (long-range outliers excluded)
        hm_mask = ys <= 55
        has_spread = (np.std(xs[hm_mask]) > 0.5) and (np.std(ys[hm_mask]) > 0.5)
        if hm_mask.sum() >= 5 and has_spread:
            xi = np.linspace(0, 68, 130)
            yi = np.linspace(0, 55, 115)        # ← extended from 52.5 to 55
            Xi, Yi = np.meshgrid(xi, yi)
            kernel = gaussian_kde(
                np.vstack([xs[hm_mask], ys[hm_mask]]),
                weights=ws[hm_mask], bw_method=0.18,
            )
            Zi = kernel(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            fig.add_trace(go.Heatmap(
                x=xi, y=yi, z=Zi,
                zmin=0, zmax=Zi.max() * 0.72,
                colorscale=[
                    [0.00, "rgba(0,0,0,0)"],
                    [0.25, "rgba(0,60,180,0.0)"],
                    [0.45, "rgba(15,160,90,0.40)"],
                    [0.65, "rgba(250,190,0,0.62)"],
                    [0.82, "rgba(240,70,20,0.80)"],
                    [1.00, "rgba(180,0,0,0.95)"],
                ],
                showscale=False, hoverinfo="skip", opacity=1.0,
                zsmooth="best", name="",
            ))

    # ── Shot dots ─────────────────────────────────────────────────────────────
    seen = set()
    for _, row in df_filtered.iterrows():
        shot_type = row["shot_type"]
        style = STYLES.get(shot_type, STYLES["miss"])
        sz    = xg_to_size(row["xg"], style["size_mult"])

        added    = row["added"]
        min_str  = (f"{int(row['minute'])}+{int(added)}′"
                    if pd.notna(added) and added > 0
                    else f"{int(row['minute'])}′")
        xgot_val = row["xgot"]
        xgot_str = f"{xgot_val:.3f}" if pd.notna(xgot_val) else "—"

        hover = (
            f"<b style='color:#F1F5F9'>{row['player']}</b>"
            f"  <span style='color:#475569'>#{row['jersey']} · {row['position']}</span><br>"
            f"<span style='color:#334155'>────────────────────</span><br>"
            f"<b style='color:{style['edge']}'>{style['label'].upper()}</b>"
            f"  <span style='color:#64748B'>{min_str}</span><br>"
            f"<span style='color:#475569'>{row['match']}</span><br>"
            f"<span style='color:#334155'>────────────────────</span><br>"
            f"xG <b style='color:#F1F5F9'>{row['xg']:.4f}</b>"
            f"  &nbsp; xGoT <b style='color:#F1F5F9'>{xgot_str}</b><br>"
            f"<span style='color:#475569'>{row['situation']} · {row['body_part']}</span>"
        )

        first_of_type = style["label"] not in seen
        if first_of_type:
            seen.add(style["label"])

        fig.add_trace(go.Scatter(
            x=[row["pitch_x"]], y=[row["pitch_y"]], mode="markers",
            name=style["label"], legendgroup=style["label"], showlegend=first_of_type,
            marker=dict(
                symbol="circle", size=sz,
                color=style["face"], opacity=0.88,
                line=dict(color=style["edge"], width=1.8),
            ),
            text=[hover], hovertemplate="%{text}<extra></extra>",
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    # Max visible y = half-pitch + a little overflow for rare long shots
    y_max = min(df_filtered["pitch_y"].max() + 4, 62) if n else 56
    y_max = max(y_max, 56)    # never smaller than half-pitch view

    fig.update_layout(
        paper_bgcolor="#080C14",
        plot_bgcolor="#091A0D",
        font=dict(family="'Inter', 'Arial', sans-serif", color="#64748B", size=11),
        height=700,                                         # ← taller for more depth
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#0F172A", bordercolor="#1E293B",
            font=dict(size=11, color="#94A3B8", family="'Inter','Arial',sans-serif"),
            align="left",
        ),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.05,
            font=dict(size=11, color="#64748B"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(t=16, b=56, l=12, r=12),
        xaxis=dict(range=[-2, 70], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-4, y_max], showgrid=False, zeroline=False,
                   showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYST NOTES
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()

    # Central corridor: width 20–48m, depth ≤ 30m (was ≤20 with old scale)
    central = df_filtered[
        (df_filtered["pitch_x"] >= 20) &
        (df_filtered["pitch_x"] <= 48) &
        (df_filtered["pitch_y"] <= 30)
    ]
    central_pct = len(central) / n * 100

    wide = df_filtered[(df_filtered["pitch_x"] < 20) | (df_filtered["pitch_x"] > 48)]
    wide_avg_xg = wide["xg"].mean() if len(wide) > 0 else 0

    high_q    = df_filtered[df_filtered["xg"] > 0.15]
    hq_pct    = len(high_q) / n * 100
    hq_goals  = int((high_q["shot_type"] == "goal").sum())
    hq_conv   = hq_goals / len(high_q) if len(high_q) > 0 else 0

    box_shots = df_filtered[df_filtered["pitch_y"] <= 16.5]   # penalty area
    box_pct   = len(box_shots) / n * 100

    loc_lines = [
        f"Zərbələrin <strong style='color:#F1F5F9'>{central_pct:.0f}%</strong>-i"
        f" mərkəzi koridordandır (30m daxili). "
    ]
    if central_pct > 60:
        loc_lines.append("Komanda optimal yerlərdən hücuma keçir.")
    elif central_pct < 40:
        loc_lines.append("Çox kənar zərbə — mərkəzə daha çox daxil olmaq lazımdır.")
    if len(wide) > 0 and wide_avg_xg < 0.08:
        loc_lines.append(
            f"Kənar zərbələrin orta xG-si "
            f"<strong style='color:#F87171'>{wide_avg_xg:.3f}</strong> — keyfiyyətsizdir."
        )

    q_color = "#F87171" if hq_conv < 0.35 else ("#4ADE80" if hq_conv > 0.50 else "#94A3B8")
    q_lines = [
        f"Yüksək keyfiyyətli zərbə (xG&gt;0.15): "
        f"<strong style='color:#F1F5F9'>{hq_pct:.0f}%</strong> — "
        f"{hq_goals} qol "
        f"(<strong style='color:{q_color}'>{hq_conv*100:.0f}% konversiya</strong>)."
    ]
    if hq_conv < 0.35:
        q_lines.append("Yaxşı şanslar reallaşmır — bitiricilik problemi mövcuddür.")
    elif hq_conv > 0.50:
        q_lines.append("Optimal mövqelər effektiv qola çevrilir.")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(f"""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-left:3px solid #38BDF8;border-radius:8px;padding:20px 22px">
  <p style="font-size:0.60rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#1E3A5F;margin:0 0 12px">
     Zərbə Yerləşməsi
  </p>
  <p style="font-size:0.88rem;color:#64748B;line-height:1.75;margin:0">
    {"<br>".join(loc_lines)}
  </p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-left:3px solid #F59E0B;border-radius:8px;padding:20px 22px">
  <p style="font-size:0.60rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#44300A;margin:0 0 12px">
     Keyfiyyət &amp; Konversiya
  </p>
  <p style="font-size:0.88rem;color:#64748B;line-height:1.75;margin:0">
    {"<br>".join(q_lines)}
  </p>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SHOT DISTANCE DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<p style='font-size:0.58rem;font-weight:700;letter-spacing:0.18em;"
        "text-transform:uppercase;color:#334155;margin:0 0 4px'>"
        "Zərbə Məsafəsi Paylanması</p>",
        unsafe_allow_html=True,
    )

    df_filtered["distance"] = df_filtered.apply(calc_distance, axis=1)

    bins = [0, 8, 16, 24, 32, 40, 60]
    labels_dist = ["0–8m", "8–16m", "16–24m", "24–32m", "32–40m", "40m+"]
    df_filtered["dist_bin"] = pd.cut(
        df_filtered["distance"], bins=bins, labels=labels_dist, right=False,
    )

    dist_grp = (
        df_filtered.groupby("dist_bin", observed=True)
        .agg(
            shots=("xg", "count"),
            avg_xg=("xg", "mean"),
            goals=("shot_type", lambda x: (x == "goal").sum()),
        )
        .reindex(labels_dist)
        .fillna(0)
    )

    fig_dist = go.Figure()

    # Shot count bars
    fig_dist.add_trace(go.Bar(
        x=dist_grp.index,
        y=dist_grp["shots"],
        name="Zərbə sayı",
        marker_color="#38BDF8",
        opacity=0.75,
        text=[int(v) for v in dist_grp["shots"]],
        textposition="outside",
        textfont=dict(color="#64748B", size=10),
        yaxis="y",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Zərbə: %{y}<br>"
            "<extra></extra>"
        ),
    ))

    # Goal count bars (stacked appearance)
    fig_dist.add_trace(go.Bar(
        x=dist_grp.index,
        y=dist_grp["goals"],
        name="Qol",
        marker_color="#F59E0B",
        opacity=0.90,
        yaxis="y",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Qol: %{y}<br>"
            "<extra></extra>"
        ),
    ))

    # Average xG line on secondary axis
    fig_dist.add_trace(go.Scatter(
        x=dist_grp.index,
        y=dist_grp["avg_xg"],
        name="Orta xG",
        mode="markers+lines",
        marker=dict(color="#F87171", size=8, line=dict(color="white", width=1)),
        line=dict(color="#F87171", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Orta xG: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig_dist.update_layout(
        paper_bgcolor="#080C14",
        plot_bgcolor="#0B1220",
        font=dict(family="'Inter','Arial',sans-serif", color="#64748B", size=11),
        height=320,
        barmode="overlay",
        hovermode="x unified",
        margin=dict(t=16, b=40, l=44, r=44),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.06,
            font=dict(size=10, color="#64748B"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color="#64748B"),
        ),
        yaxis=dict(
            title="Zərbə", title_font=dict(size=9, color="#334155"),
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, tickfont=dict(size=10, color="#64748B"),
        ),
        yaxis2=dict(
            title="Orta xG", title_font=dict(size=9, color="#F87171"),
            overlaying="y", side="right",
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color="#F87171"),
            rangemode="tozero",
        ),
    )

    st.plotly_chart(fig_dist, use_container_width=True)