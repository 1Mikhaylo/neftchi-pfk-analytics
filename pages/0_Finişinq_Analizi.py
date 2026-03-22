import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Finishing Analysis · Neftchi", page_icon="⚽", layout="wide")

# ── Global CSS (matching Shot Map) ────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/neftchi_shots.csv")
    
    # FIX: Standardize player names
    df['player'] = df['player'].replace({
        'M. Mammadov': 'M. Məmmədov',
        'B. Almeyda': 'B. Almeida',
    })
    
    return df

df = load_data()

BG = "#0D1B2A"
GRID = "rgba(255,255,255,0.07)"
BLUE = "#4FC3F7"
RED = "#EF5350"
MUTED = "#7A8FA6"

total_xg = round(df["xg"].sum(), 2)
total_goals = int((df["shot_type"] == "goal").sum())
conversion = round(total_goals / total_xg, 3) if total_xg else 0.0
diff = round(total_xg - total_goals, 2)

sit = df.groupby("situation").agg(
    xg=("xg", "sum"),
    goals=("shot_type", lambda x: (x == "goal").sum())
).reset_index().sort_values("xg", ascending=True)

player = df.groupby("player").agg(
    shots=("xg", "count"),
    xg=("xg", "sum"),
    goals=("shot_type", lambda x: (x == "goal").sum())
).reset_index()
player["diff"] = (player["xg"] - player["goals"]).round(2)
player = player[player["shots"] >= 5].sort_values("diff", ascending=False)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:28px">
  <p style="font-size:0.62rem;font-weight:700;letter-spacing:0.20em;text-transform:uppercase;
            color:#38BDF8;margin:0 0 6px">Neftchi FK &nbsp;·&nbsp; 2025/26 Premyer Liqa</p>
  <h1 style="font-size:2.1rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.03em;
             margin:0;line-height:1.05">Finishing Analysis</h1>
  <p style="font-size:0.82rem;color:#3D5068;margin:8px 0 0">
    Yaradılan şansların qola çevrilmə effektivliyi
  </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Qol", total_goals)
c2.metric("xG", total_xg)
c3.metric("Konversiya", f"{conversion:.2f}", help="Qol / xG — 1.0 normadır")
c4.metric("xG fərqi", f"{-diff:+.1f}", help="Müsbət = xG-dən artıq qol, mənfi = israf")

st.divider()

# ── Chart ────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.42, 0.58],
    horizontal_spacing=0.10,
    subplot_titles=["Situasiya üzrə xG vs Qol", "Oyunçu üzrə xG fərqi (≥5 zərbə)"]
)

for label, col, color in [("xG", "xg", MUTED), ("Qol", "goals", BLUE)]:
    fig.add_trace(go.Bar(
        y=sit["situation"], x=sit[col].round(1),
        name=label, orientation="h",
        marker_color=color,
        text=sit[col].round(1), textposition="outside",
        textfont=dict(size=10, color=MUTED),
        cliponaxis=False,
    ), row=1, col=1)

for _, p in player.iterrows():
    c = RED if p["diff"] > 0 else BLUE
    fig.add_trace(go.Scatter(
        x=[0, p["diff"]], y=[p["player"], p["player"]],
        mode="lines", line=dict(color=c, width=1.5),
        showlegend=False, hoverinfo="skip"
    ), row=1, col=2)

fig.add_trace(go.Scatter(
    x=player["diff"], y=player["player"],
    mode="markers",
    marker=dict(color=[RED if v > 0 else BLUE for v in player["diff"]], size=9),
    showlegend=False,
    customdata=player[["xg", "goals", "shots"]].round(2),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "xG: %{customdata[0]}  |  Qol: %{customdata[1]}<br>"
        "Zərbə: %{customdata[2]}<extra></extra>"
    )
), row=1, col=2)

fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), row=1, col=2)

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="'Inter', 'Arial', sans-serif", color=MUTED, size=11),
    height=400,
    barmode="group",
    margin=dict(t=40, b=20, l=10, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=False, zeroline=False)

st.plotly_chart(fig, use_container_width=True)

# ── Analyst notes ────────────────────────────────────────────────────────────
st.divider()

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-left:3px solid #38BDF8;border-radius:8px;padding:20px 22px">
  <p style="font-size:0.60rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#1E3A5F;margin:0 0 12px">
     Komanda finişi
  </p>
""", unsafe_allow_html=True)
    
    if conversion < 0.85:
        st.markdown(
            f"<p style='font-size:0.88rem;color:#64748B;line-height:1.75;margin:0'>"
            f"Neftçi bu dövrdə <strong style='color:#F1F5F9'>{total_xg} xG</strong> yaratmasına baxmayaraq, "
            f"cəmi <strong style='color:#F1F5F9'>{total_goals}</strong> qol vura bilib — konversiya "
            f"<strong style='color:#F87171'>{conversion:.2f}</strong>, "
            f"yəni hər yaradılan şansın yalnız {conversion*100:.0f}%-i nəticəyə çevrilib. "
            f"Bu, hücum xəttində ciddi effektivlik probleminə işarədir.</p>",
            unsafe_allow_html=True
        )
    elif conversion < 1.0:
        st.markdown(
            f"<p style='font-size:0.88rem;color:#64748B;line-height:1.75;margin:0'>"
            f"Neftçi <strong style='color:#F1F5F9'>{total_xg} xG</strong>-dən "
            f"<strong style='color:#F1F5F9'>{total_goals}</strong> qol vurub — "
            f"konversiya <strong style='color:#94A3B8'>{conversion:.2f}</strong> ilə normaya yaxın, lakin hələ də "
            f"<strong style='color:#F87171'>{diff:.1f} xG</strong> "
            f"israf olunub. Əsas problem kəskin anların reallaşdırılmasındadır.</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='font-size:0.88rem;color:#64748B;line-height:1.75;margin:0'>"
            f"Neftçi <strong style='color:#F1F5F9'>{total_xg} xG</strong>-dən "
            f"<strong style='color:#F1F5F9'>{total_goals}</strong> qol vuraraq gözləntiləri aşıb — "
            f"konversiya <strong style='color:#4ADE80'>{conversion:.2f}</strong>. Bu göstərici ya yüksək keyfiyyətli finalın, "
            f"ya da şanslı epizodların nəticəsidir.</p>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div style="background:#0A0F1E;border:1px solid rgba(255,255,255,0.05);
     border-left:3px solid #F59E0B;border-radius:8px;padding:20px 22px">
  <p style="font-size:0.60rem;font-weight:700;letter-spacing:0.15em;
     text-transform:uppercase;color:#44300A;margin:0 0 12px">
     Ən qeyri-effektiv / Ən effektiv
  </p>
""", unsafe_allow_html=True)
    
    if player.empty:
        st.markdown(
            "<p style='font-size:0.88rem;color:#64748B;line-height:1.75;margin:0'>"
            "Minimum 5 zərbəyə sahib oyunçu yoxdur.</p>",
            unsafe_allow_html=True
        )
    else:
        worst = player.iloc[0]
        best = player.loc[player["diff"].idxmin()]
        
        st.markdown(
            f"<p style='font-size:0.88rem;color:#64748B;line-height:1.75;margin:0'>"
            f"<strong style='color:#F1F5F9'>{worst['player']}</strong> — "
            f"{worst['shots']:.0f} zərbədən {worst['xg']:.1f} xG yaradıb, lakin cəmi "
            f"<strong style='color:#F1F5F9'>{int(worst['goals'])}</strong> qol vurub "
            f"(<strong style='color:#F87171'>{worst['diff']:.1f} xG boşa gedib</strong>). "
            f"Bu həcmdə şans yaradan oyunçudan daha yüksək reallaşma gözlənilir.",
            unsafe_allow_html=True
        )
        
        if best["player"] != worst["player"]:
            st.markdown(
                f"<br><br><strong style='color:#F1F5F9'>{best['player']}</strong> isə "
                f"{best['xg']:.1f} xG qarşısında "
                f"<strong style='color:#F1F5F9'>{int(best['goals'])}</strong> qol vuraraq "
                f"(<strong style='color:#4ADE80'>{-best['diff']:+.1f}</strong>) "
                f"komandanın ən effektiv bitiricisi olub.</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)