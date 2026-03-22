"""
visualizations.py — All chart/plot helpers for MammoCAD dashboards
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Netflix-dark color palette
DARK_BG    = "#141414"
CARD_BG    = "#1f1f1f"
RED        = "#E50914"
RED_LIGHT  = "#FF4F57"
WHITE      = "#FFFFFF"
GRAY       = "#808080"
GRAY_LIGHT = "#b3b3b3"
GREEN      = "#46d369"

_LAYOUT_BASE = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=WHITE, family="'Helvetica Neue', Arial, sans-serif"),
    margin=dict(l=30, r=30, t=50, b=30),
)


# ══════════════════════════════════════════════════════════════
# 1. Probability gauge
# ══════════════════════════════════════════════════════════════

def probability_gauge(benign_prob: float, malignant_prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(malignant_prob * 100, 1),
        title={"text": "Malignancy Probability (%)", "font": {"color": WHITE, "size": 16}},
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": RED}, "decreasing": {"color": GREEN}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": GRAY, "tickfont": {"color": GRAY_LIGHT}},
            "bar": {"color": RED if malignant_prob >= 0.5 else GREEN, "thickness": 0.3},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20],  "color": "#1a3a1a"},
                {"range": [20, 50], "color": "#2a2a1a"},
                {"range": [50, 80], "color": "#3a1a1a"},
                {"range": [80, 100],"color": "#4a0a0a"},
            ],
            "threshold": {
                "line": {"color": WHITE, "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
        number={"font": {"color": WHITE, "size": 40}, "suffix": "%"},
    ))
    fig.update_layout(**_LAYOUT_BASE, height=280)
    return fig


# ══════════════════════════════════════════════════════════════
# 2. Benign vs Malignant bar
# ══════════════════════════════════════════════════════════════

def probability_bar(benign_prob: float, malignant_prob: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Benign", "Malignant"],
        y=[benign_prob * 100, malignant_prob * 100],
        marker_color=[GREEN, RED],
        text=[f"{benign_prob*100:.1f}%", f"{malignant_prob*100:.1f}%"],
        textposition="outside",
        textfont=dict(color=WHITE, size=16),
        width=0.4,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Prediction Confidence",
        yaxis=dict(range=[0, 120], showgrid=False, visible=False),
        xaxis=dict(showgrid=False),
        height=300,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 3. BI-RADS distribution bar (population context)
# ══════════════════════════════════════════════════════════════

def birads_distribution_chart(birads_category: str) -> go.Figure:
    cats   = ["BI-RADS 0", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5", "BI-RADS 6"]
    probs  = [5, 25, 40, 20, 7, 3]        # Approximate distribution
    colors = [
        GRAY if c != birads_category else RED for c in cats
    ]
    fig = go.Figure(go.Bar(
        x=cats,
        y=probs,
        marker_color=colors,
        text=[f"{p}%" for p in probs],
        textposition="outside",
        textfont=dict(color=WHITE, size=12),
        width=0.5,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="BI-RADS Category Distribution (Population Reference)",
        yaxis=dict(visible=False, showgrid=False),
        xaxis=dict(showgrid=False),
        height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 4. Radar / Spider — Cell characteristics
# ══════════════════════════════════════════════════════════════

def radar_chart(features: dict, patient_name: str = "Patient") -> go.Figure:
    mean_keys = [k for k in features if k.endswith("_mean")]
    if not mean_keys:
        return go.Figure()

    labels = [k.replace("_mean", "").replace("_", " ").title() for k in mean_keys]
    values = [features[k] for k in mean_keys]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    # Worst features for outer ring
    worst_keys   = [k.replace("_mean", "_worst") for k in mean_keys]
    worst_values = [features.get(k, 0) for k in worst_keys]
    worst_closed = worst_values + [worst_values[0]]

    fig = go.Figure()

    # Outer ring — worst case
    fig.add_trace(go.Scatterpolar(
        r=worst_closed,
        theta=labels_closed,
        fill="toself",
        name="Worst",
        line=dict(color=RED, width=1, dash="dot"),
        fillcolor=f"rgba(229,9,20,0.08)",
        hovertemplate="%{theta}: %{r:.3f}<extra>Worst</extra>",
    ))

    # Mean ring
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name="Mean",
        line=dict(color=RED_LIGHT, width=2),
        fillcolor=f"rgba(255,79,87,0.20)",
        hovertemplate="%{theta}: %{r:.3f}<extra>Mean</extra>",
    ))

    # Dot markers
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        mode="markers",
        name="Values",
        marker=dict(color=WHITE, size=7, symbol="circle"),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Cell Feature Profile — {patient_name}",
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color=GRAY_LIGHT, size=9),
                gridcolor="#333",
                linecolor="#444",
            ),
            angularaxis=dict(
                tickfont=dict(color=WHITE, size=11),
                gridcolor="#333",
                linecolor="#444",
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color=WHITE), bgcolor=CARD_BG),
        height=480,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 5. Feature comparison bar — mean vs worst
# ══════════════════════════════════════════════════════════════

def feature_bar_chart(features: dict) -> go.Figure:
    mean_keys  = sorted([k for k in features if k.endswith("_mean")])
    worst_keys = [k.replace("_mean", "_worst") for k in mean_keys]
    labels     = [k.replace("_mean", "").replace("_", " ").title() for k in mean_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Mean",
        x=labels,
        y=[features[k] for k in mean_keys],
        marker_color=RED_LIGHT,
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Worst",
        x=labels,
        y=[features.get(k, 0) for k in worst_keys],
        marker_color=RED,
        opacity=0.7,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Feature Comparison: Mean vs Worst",
        barmode="group",
        xaxis_tickangle=-35,
        yaxis=dict(range=[0, 1.15], gridcolor="#333"),
        height=380,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 6. Population statistics (doctor dashboard)
# ══════════════════════════════════════════════════════════════

def population_pie(benign_count: int, malignant_count: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Benign", "Malignant"],
        values=[benign_count, malignant_count],
        hole=0.55,
        marker=dict(colors=[GREEN, RED], line=dict(color=DARK_BG, width=3)),
        textinfo="label+percent",
        textfont=dict(color=WHITE, size=13),
        hovertemplate="%{label}: %{value} cases (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Case Distribution",
        showlegend=False,
        height=300,
        annotations=[dict(
            text=f"<b>{benign_count + malignant_count}</b><br>Total",
            x=0.5, y=0.5, font_size=18,
            font_color=WHITE, showarrow=False
        )],
    )
    return fig


def analyses_over_time(analyses: list) -> go.Figure:
    if not analyses:
        return go.Figure()

    df = pd.DataFrame(analyses)
    df["date"] = pd.to_datetime(df["analysed_at"]).dt.date
    daily = df.groupby(["date", "prediction"]).size().reset_index(name="count")

    fig = go.Figure()
    for pred, color in [("Benign", GREEN), ("Malignant", RED)]:
        d = daily[daily["prediction"] == pred]
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["count"],
            mode="lines+markers",
            name=pred,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Analyses Over Time",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2a2a"),
        height=300,
    )
    return fig


def birads_histogram(analyses: list) -> go.Figure:
    if not analyses:
        return go.Figure()

    df   = pd.DataFrame(analyses)
    cats = df["birads_category"].value_counts().reset_index()
    cats.columns = ["category", "count"]

    fig = go.Figure(go.Bar(
        x=cats["category"],
        y=cats["count"],
        marker_color=[RED if "5" in c or "6" in c else RED_LIGHT if "4" in c else GRAY
                      for c in cats["category"]],
        text=cats["count"],
        textposition="outside",
        textfont=dict(color=WHITE),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="BI-RADS Category Distribution (Your Cases)",
        yaxis=dict(visible=False, showgrid=False),
        height=300,
    )
    return fig


def feature_scatter_matrix(analyses: list) -> go.Figure:
    """Scatter matrix of key features across all analyses."""
    records = []
    for a in analyses:
        f = a.get("features", {})
        if f:
            records.append({
                "Radius":     f.get("radius_mean", 0),
                "Texture":    f.get("texture_mean", 0),
                "Compactness":f.get("compactness_mean", 0),
                "Concavity":  f.get("concavity_mean", 0),
                "Diagnosis":  a.get("prediction", "Unknown"),
            })
    if not records:
        return go.Figure()

    df = pd.DataFrame(records)
    color_map = {"Benign": GREEN, "Malignant": RED}
    fig = px.scatter_matrix(
        df,
        dimensions=["Radius", "Texture", "Compactness", "Concavity"],
        color="Diagnosis",
        color_discrete_map=color_map,
        title="Feature Scatter Matrix",
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=500,
        legend=dict(font=dict(color=WHITE)),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))