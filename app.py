"""
PRISM — Proof-driven Ranking & Intelligent Selection Model
Streamlit frontend for the 15-layer evidence-backed hiring pipeline.

Design concept: a prism splits one beam of light into a spectrum. PRISM splits
one candidate score into six rubric dimensions. That single idea — the
"spectral signature" — is the recurring visual device used everywhere a score
needs to be read: the ranking table, the roster cards, and the candidate
dossier.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="PRISM",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────────────────────────────
# One score, six wavelengths. Each rubric dimension owns a fixed hue across
# the whole app — the same law that maps confidence and claim status onto
# the same three-color scale (green / amber / red) rather than inventing a
# new palette per widget.
DIMENSIONS = [
    ("skill_fit", "Skill Fit", "var(--spec-skill)", "#D7263D"),
    ("experience_depth", "Experience Depth", "var(--spec-experience)", "#F0631F"),
    ("seniority_match", "Seniority Match", "var(--spec-seniority)", "#D9A400"),
    ("domain_match", "Domain Match", "var(--spec-domain)", "#1F9E5B"),
    ("career_growth", "Career Growth", "var(--spec-growth)", "#1768D1"),
    ("proof_strength", "Proof Strength", "var(--spec-proof)", "#5B3DF2"),
]
WEIGHT_MAP = {
    "Skill Fit": "0.30", "Experience Depth": "0.20", "Seniority Match": "0.15",
    "Domain Match": "0.15", "Career Growth": "0.10", "Proof Strength": "0.10",
}
TIER_LEVELS = {"exceptional": 5, "strong": 4, "moderate": 3, "weak": 2, "poor": 1}
TIER_COLOR = {
    "exceptional": "var(--spec-domain)", "strong": "var(--spec-domain)",
    "moderate": "var(--spec-seniority)", "weak": "var(--spec-skill)", "poor": "var(--spec-skill)",
}
CONF_COLOR = {
    "high": "var(--spec-domain)", "medium": "var(--spec-seniority)",
    "low": "var(--spec-skill)", "unstable": "var(--spec-skill)",
}
CONF_ICON = {"high": "—", "medium": "∼", "low": "≈", "unstable": "≈"}
STATUS_COLOR = {
    "verified": "var(--spec-domain)", "weak": "var(--spec-seniority)",
    "inferred": "var(--spec-growth)", "unsupported": "var(--spec-skill)",
    "contradicted": "var(--spec-skill)", "pending": "var(--ink-muted)",
}

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#050816;
  --bg2:#0B1023;

  --surface:rgba(255,255,255,.06);
  --surface-hover:rgba(255,255,255,.10);

  --line:rgba(255,255,255,.08);

  --ink:#F8FAFF;
  --ink-muted:#AAB2D5;

  --spec-skill:#FF4D6D;
  --spec-experience:#FF8A3D;
  --spec-seniority:#FFD93D;
  --spec-domain:#38D39F;
  --spec-growth:#4DA8FF;
  --spec-proof:#8B5CF6;
}

html, body, [data-testid="stApp"]{
    background:
      radial-gradient(circle at top left, rgba(139,92,246,.25), transparent 30%),
      radial-gradient(circle at top right, rgba(77,168,255,.18), transparent 30%),
      radial-gradient(circle at bottom left, rgba(255,77,109,.18), transparent 35%),
      linear-gradient(135deg,var(--bg),var(--bg2));

    color:var(--ink);
    font-family:'Inter',sans-serif;
}

/* Hide Streamlit branding */
#MainMenu,
footer,
header{
    visibility:hidden;
}

/* Sidebar */

[data-testid="stSidebar"]{
    background:rgba(10,14,30,.95);
    backdrop-filter:blur(24px);
    border-right:1px solid var(--line);
}

[data-testid="stSidebar"] *{
    color:var(--ink) !important;
}

/* Typography */

h1,h2,h3,h4,h5,h6{
    font-family:'Space Grotesk',sans-serif !important;
    color:var(--ink);
}

p,span,label,div{
    color:var(--ink);
}

/* Glass Cards */

.prism-card,
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stExpander"]{
    background:var(--surface) !important;

    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);

    border:1px solid var(--line) !important;

    border-radius:20px !important;

    box-shadow:
      0 8px 40px rgba(0,0,0,.35),
      inset 0 1px 0 rgba(255,255,255,.08);

    transition:.3s ease;
}

.prism-card:hover{
    transform:translateY(-4px);

    box-shadow:
      0 16px 45px rgba(139,92,246,.25);z
}

/* Buttons */

.stButton button,
.stDownloadButton button{

    background:linear-gradient(
        135deg,
        #8B5CF6,
        #4DA8FF);

    color:white !important;

    border:none !important;

    border-radius:14px !important;

    font-family:'IBM Plex Mono',monospace;

    font-weight:600;

    padding:.7rem 1rem;

    box-shadow:
      0 0 25px rgba(139,92,246,.4);

    transition:.25s ease;
}

.stButton button:hover,
.stDownloadButton button:hover{

    transform:translateY(-2px) scale(1.02);

    box-shadow:
      0 0 40px rgba(139,92,246,.7);
}

/* Inputs */

.stTextArea textarea,
.stTextInput input{

    background:rgba(255,255,255,.05) !important;

    color:white !important;

    border:1px solid var(--line) !important;

    border-radius:14px !important;
}

/* Tabs */

button[data-baseweb="tab"]{

    color:var(--ink-muted) !important;

    font-family:'IBM Plex Mono',monospace;

    text-transform:uppercase;
}

button[data-baseweb="tab"][aria-selected="true"]{

    color:#8B5CF6 !important;
}

div[data-baseweb="tab-highlight"]{
    background:#8B5CF6 !important;
}

/* Scrollbar */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#8B5CF6;
    border-radius:50px;
}

/* Hero */

.hero{

    padding:2rem;

    border-radius:24px;

    background:
      linear-gradient(
      135deg,
      rgba(139,92,246,.25),
      rgba(77,168,255,.12),
      rgba(255,77,109,.10));

    backdrop-filter:blur(25px);

    border:1px solid rgba(255,255,255,.08);

    margin-bottom:1.5rem;
}

.hero-title{

    font-size:4rem;
    font-weight:800;

    background:linear-gradient(
      90deg,
      #FF4D6D,
      #FF8A3D,
      #FFD93D,
      #38D39F,
      #4DA8FF,
      #8B5CF6);

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-subtitle{

    color:var(--ink-muted);
    margin-top:.4rem;
    font-size:1rem;
}

</style>
""", unsafe_allow_html=True)

# ── Signature graphics ───────────────────────────────────────────────────────

def prism_logo_svg(size: int = 40) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="2" y1="32" x2="22" y2="32" stroke="#14181F" stroke-width="2.5"/>'
        f'<polygon points="22,46 22,18 46,32" fill="none" stroke="#14181F" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<line x1="46" y1="32" x2="62" y2="14" stroke="#D7263D" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="21" stroke="#F0631F" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="27" stroke="#D9A400" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="33" stroke="#1F9E5B" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="39" stroke="#1768D1" stroke-width="2"/>'
        f'<line x1="46" y1="32" x2="62" y2="46" stroke="#5B3DF2" stroke-width="2"/>'
        f'</svg>'
    )


def spectrum_rule(thick: int = 6) -> None:
    st.markdown(
        f"""
        <div style="
            height:{thick}px;
            border-radius:100px;

            background: linear-gradient(
                90deg,
                #FF4D6D 0%,
                #FF8A3D 20%,
                #FFD93D 40%,
                #38D39F 60%,
                #4DA8FF 80%,
                #8B5CF6 100%
            );

            box-shadow:
                0 0 8px rgba(255,77,109,0.8),
                0 0 16px rgba(77,168,255,0.7),
                0 0 24px rgba(139,92,246,0.8);

            margin: 1rem 0 1.5rem;

            position: relative;
            overflow: hidden;
        ">
            <div style="
                position:absolute;
                top:0;
                left:-100%;
                width:100%;
                height:100%;
                background:linear-gradient(
                    90deg,
                    transparent,
                    rgba(255,255,255,0.5),
                    transparent
                );
                animation: shimmer 3s infinite;
            "></div>
        </div>

        <style>
        @keyframes shimmer {{
            0% {{
                left: -100%;
            }}
            100% {{
                left: 100%;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def spectral_signature(card: dict, height: int = 28, bar_w: int = 10, gap: int = 3) -> str:
    """The signature device: one bar per rubric dimension, colored by its
    fixed wavelength, height by score. A candidate's spectral fingerprint."""
    x = 0
    bars = []
    for key, _, _, hexcolor in DIMENSIONS:
        v = max(0.0, min(1.0, float(card.get(key, 0) or 0)))
        h = max(2, round(v * height))
        y = height - h
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="2" fill="{hexcolor}"/>')
        x += bar_w + gap
    total_w = x - gap
    return (
        f'<svg width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle">' + "".join(bars) + "</svg>"
    )


# ── Text/markup helpers ───────────────────────────────────────────────────────

def section_header(text: str) -> None:
    st.markdown(f"<div class='section-hdr'>{text}</div>", unsafe_allow_html=True)


def id_tag(cid: str) -> str:
    return f"<span class='id-tag'>{cid}</span>"


def rank_mono(rank, leader: bool = False) -> str:
    color = "var(--spec-proof)" if leader else "var(--ink-muted)"
    weight = 700 if leader else 600
    try:
        txt = f"№{int(rank):02d}"
    except (TypeError, ValueError):
        txt = "№??"
    return (f"<span style='font-family:IBM Plex Mono,monospace;font-size:1rem;"
            f"color:{color};font-weight:{weight}'>{txt}</span>")


def tier_meter(label: str) -> str:
    label = (label or "moderate").lower()
    n = TIER_LEVELS.get(label, 3)
    color = TIER_COLOR.get(label, "var(--ink-muted)")
    bars = ""
    for i in range(5):
        h = 5 + i * 3
        fill = color if i < n else "var(--line)"
        bars += (f"<span style='display:inline-block;width:5px;height:{h}px;background:{fill};"
                 f"margin-right:2px;border-radius:1px;vertical-align:bottom'></span>")
    return (f"<span style='display:inline-flex;align-items:flex-end;margin-right:.5rem'>{bars}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;letter-spacing:.07em;"
            f"color:{color};text-transform:uppercase;font-weight:700'>{label}</span>")


def conf_tag(conf: str) -> str:
    conf = (conf or "medium").lower()
    color = CONF_COLOR.get(conf, "var(--ink-muted)")
    icon = CONF_ICON.get(conf, "—")
    return (f"<span style='font-family:IBM Plex Mono,monospace;font-size:.7rem;letter-spacing:.05em;"
            f"color:{color};font-weight:700'>{icon} {conf.upper()}</span>")


def dim_row(label: str, value: float, color: str) -> None:
    pct = int(max(0.0, min(1.0, float(value or 0))) * 100)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.7rem;margin:.3rem 0'>"
        f"<span style='width:150px;font-family:IBM Plex Mono,monospace;font-size:.72rem;"
        f"letter-spacing:.03em;color:var(--ink-muted);flex-shrink:0;text-transform:uppercase'>{label}</span>"
        f"<div style='flex:1;background:var(--line);border-radius:3px;height:8px;overflow:hidden'>"
        f"<div style='width:{pct}%;background:{color};height:100%'></div></div>"
        f"<span style='width:42px;text-align:right;font-family:IBM Plex Mono,monospace;"
        f"font-size:.78rem;font-weight:700;color:var(--ink)'>{float(value or 0):.2f}</span></div>",
        unsafe_allow_html=True,
    )


def render_weights() -> None:
    section_header("Weights applied")
    row = " &nbsp;&nbsp; ".join(
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:var(--ink-muted)'>"
        f"{lbl} <b style='color:var(--ink)'>{w}</b></span>"
        for lbl, w in WEIGHT_MAP.items()
    )
    st.markdown(row, unsafe_allow_html=True)


def skill_chips(items: list[str], variant: str) -> str:
    if not items:
        return "<span style='color:var(--ink-muted);font-size:.8rem;font-family:IBM Plex Mono,monospace'>none</span>"
    return " ".join(f'<span class="chip {variant}">{s}</span>' for s in items)


def readout_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"<div class='readout-cell'><div class='v'>{val}</div><div class='l'>{lbl}</div></div>"
        for lbl, val in items
    )
    st.markdown(f"<div class='readout-strip'>{cells}</div>", unsafe_allow_html=True)


def render_ledger_rows(claims: list[dict]) -> None:
    if not claims:
        st.markdown(
            "<span style='color:var(--ink-muted);font-family:IBM Plex Mono,monospace;font-size:.85rem'>"
            "No evidence ledger entries for this candidate.</span>",
            unsafe_allow_html=True,
        )
        return
    for claim in claims:
        status = (claim.get("verification_status") or "pending").lower()
        color = STATUS_COLOR.get(status, "var(--ink-muted)")
        conf = claim.get("confidence", 0.0) or 0.0
        source = claim.get("source", "resume")
        snippet = claim.get("notes") or claim.get("claim_text", "")
        url = claim.get("evidence_url")
        recency = claim.get("recency_years")
        st.markdown(
            f"<div class='ledger-row' style='border-left:3px solid {color}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
            f"<span style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink)'>"
            f"{claim.get('skill', '')}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;letter-spacing:.06em;"
            f"color:{color};font-weight:700;text-transform:uppercase'>{status}</span></div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:var(--ink-muted);margin:.3rem 0'>"
            f"confidence {conf:.2f} · source {source}"
            + (f" · {recency:.1f}y ago" if recency is not None else "")
            + "</div>"
            + (f"<div style='font-family:Source Serif 4,serif;font-size:.86rem;color:var(--ink);"
               f"margin-top:.3rem;font-style:italic'>{snippet}</div>" if snippet else "")
            + (f"<div style='margin-top:.3rem'><a href='{url}' style='color:var(--spec-proof);"
               f"font-family:IBM Plex Mono,monospace;font-size:.72rem'>{url}</a></div>" if url else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def render_interview_questions(questions: list[dict]) -> None:
    if not questions:
        st.markdown(
            "<span style='color:var(--ink-muted);font-size:.85rem'>No interview questions generated.</span>",
            unsafe_allow_html=True,
        )
        return
    for q in questions:
        priority = q.get("priority", "medium")
        color = ("var(--spec-skill)" if priority == "high" else
                  "var(--spec-seniority)" if priority == "medium" else "var(--spec-domain)")
        rationale = q.get("rationale", "")
        st.markdown(
            f"<div style='border-left:3px solid {color};padding:.5rem .9rem;margin:.4rem 0'>"
            f"<span class='chip skill'>{q.get('skill', '')}</span><br>"
            f"<span style='font-family:Source Serif 4,serif;font-size:.92rem;color:var(--ink);"
            f"font-weight:600'>{q.get('question', '')}</span>"
            + (f"<div style='font-size:.78rem;color:var(--ink-muted);font-style:italic;margin-top:.2rem'>"
               f"{rationale}</div>" if rationale else "")
            + "</div>",
            unsafe_allow_html=True,
        )


# ── Roster card ───────────────────────────────────────────────────────────────

def render_roster_card(card: dict, expanded: bool = False) -> None:
    rank = card.get("rank", 0)
    name = card.get("candidate_name") or card.get("candidate_id", "Unknown")
    score = card.get("final_score", 0.0)
    label = card.get("score_label", "moderate")
    conf = card.get("rank_confidence", "medium")
    cid = card.get("candidate_id", "")
    leader = str(rank) == "1"

    with st.container(border=True):
        cols = st.columns([0.08, 0.46, 0.24, 0.22])
        with cols[0]:
            st.markdown(rank_mono(rank, leader=leader), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(
                f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.02rem;"
                f"font-weight:700;color:var(--ink)'>{name}</span> {id_tag(cid)}<br>{conf_tag(conf)}",
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(spectral_signature(card), unsafe_allow_html=True)
        with cols[3]:
            st.markdown(
                f"<div style='text-align:right'>{tier_meter(label)}<br>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:1.15rem;"
                f"font-weight:700;color:var(--ink)'>{score:.1f}</span></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Open full report", expanded=expanded):
            tab_summary, tab_scoring, tab_evidence, tab_rank, tab_interview = st.tabs(
                ["Summary", "Scoring", "Evidence", "Rank rationale", "Interview prep"]
            )
            verified = card.get("verified_claims", 0)
            unverified = card.get("unverified_claims", 0)
            total_c = verified + unverified
            proof_pct = int(verified / total_c * 100) if total_c else 0

            with tab_summary:
                readout_strip([
                    ("Score", f"{score:.0f}"),
                    ("Verified", str(verified)),
                    ("Proof ratio", f"{proof_pct}%"),
                    ("Tier", label.capitalize()),
                ])
                if card.get("summary"):
                    st.markdown(
                        f"<div style='font-family:Source Serif 4,serif;font-size:.92rem;color:var(--ink);"
                        f"line-height:1.7;background:var(--paper);border-radius:8px;"
                        f"padding:.9rem 1.1rem;margin:.6rem 0'>{card['summary']}</div>",
                        unsafe_allow_html=True,
                    )
                c1, c2 = st.columns(2)
                with c1:
                    section_header("Strengths")
                    for s in card.get("strengths", []):
                        st.markdown(
                            f"<div style='font-size:.86rem;color:var(--ink);margin:.2rem 0'>+ {s}</div>",
                            unsafe_allow_html=True,
                        )
                with c2:
                    section_header("Risks & gaps")
                    risks = card.get("risks", [])
                    if risks:
                        for r in risks:
                            st.markdown(
                                f"<div style='font-size:.86rem;color:var(--ink);margin:.2rem 0'>− {r}</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "<span style='color:var(--spec-domain);font-size:.85rem'>No critical risks.</span>",
                            unsafe_allow_html=True,
                        )

            with tab_scoring:
                for key, lbl, var_color, _ in DIMENSIONS:
                    dim_row(lbl, card.get(key, 0), var_color)
                st.markdown("<br>", unsafe_allow_html=True)
                render_weights()

            with tab_evidence:
                e1, e2 = st.columns(2)
                with e1:
                    section_header(f"Verified ({verified})")
                    st.markdown(skill_chips(card.get("verified_skills", [])[:10], "verified"), unsafe_allow_html=True)
                with e2:
                    section_header(f"Unverified ({unverified})")
                    st.markdown(skill_chips(card.get("unverified_skills", [])[:8], "unverified"), unsafe_allow_html=True)

            with tab_rank:
                gap = card.get("score_gap_to_next")
                why = card.get("why_above_next", "")
                if gap is not None:
                    st.markdown(
                        f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;"
                        f"font-weight:800;color:var(--spec-proof)'>+{gap:.1f}</span> "
                        f"<span style='font-family:IBM Plex Mono,monospace;font-size:.78rem;"
                        f"color:var(--ink-muted)'>points above next candidate</span>",
                        unsafe_allow_html=True,
                    )
                if why:
                    st.markdown(
                        f"<div style='border-left:3px solid var(--spec-proof);padding:.7rem 1rem;"
                        f"margin-top:.6rem;font-family:Source Serif 4,serif;font-size:.88rem;"
                        f"color:var(--ink);background:var(--paper)'>{why}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:var(--ink-muted);font-size:.85rem'>No pairwise justification available.</span>",
                        unsafe_allow_html=True,
                    )

            with tab_interview:
                render_interview_questions(card.get("interview_questions", []))


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:.4rem 0 1rem'>"
            f"{prism_logo_svg(46)}"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:800;"
            f"color:var(--ink);margin-top:.4rem'>PRISM</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;letter-spacing:.08em;"
            f"color:var(--spec-proof);text-transform:uppercase;margin-top:.15rem;line-height:1.4'>"
            f"Proof-driven Ranking &amp;<br>Intelligent Selection Model</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        spectrum_rule(3)

        mode = st.radio("Input mode", ["Sample data", "Upload resumes"], label_visibility="collapsed")

        jd_text = ""
        candidate_files = []

        if mode == "Sample data":
            st.markdown(
                "<div style='font-family:Source Serif 4,serif;font-size:.82rem;color:var(--ink-muted);"
                "margin:.4rem 0 .8rem'>Runs the bundled job description against four synthetic candidates.</div>",
                unsafe_allow_html=True,
            )
        else:
            jd_text = st.text_area(
                "Job description",
                placeholder="Paste the full job description here…",
                height=180,
            )
            candidate_files = st.file_uploader(
                "Candidate resumes",
                type=["txt", "md", "pdf"],
                accept_multiple_files=True,
                help="One file per candidate. PDF, TXT, or MD.",
            )

        st.divider()
        section_header("Pipeline options")
        force_fallback = not st.toggle(
            "Use LLM (requires API key)",
            value=False,
            help="When off, the pipeline runs fully offline with rule-based fallbacks.",
        )
        shortlist_k = st.slider("Retrieval shortlist size", 5, 50, 25, 5)
        stab_runs = st.slider("Stability test runs", 3, 10, 5, 1)

        st.divider()
        run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    return mode, jd_text, candidate_files, force_fallback, shortlist_k, stab_runs, run_btn, (mode == "Sample data")


# ── Pipeline runner ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_v2_pipeline(
    jd_text: str,
    candidate_texts: list[tuple[str, str]],
    force_fallback: bool,
    k: int,
    stability_runs: int,
) -> dict:
    """Run the V2 pipeline and return all artefacts for the dashboard."""
    from ai_hiring_ranker.evaluation import run_pipeline_from_texts

    result = run_pipeline_from_texts(
        jd_text=jd_text,
        candidate_texts=candidate_texts,
        output_dir=str(ROOT / "outputs"),
        force_fallback=force_fallback,
        k=k,
        stability_runs=stability_runs,
    )

    payload = result.to_manifest_dict()
    payload["ranked_output"] = result.ranked_output

    files = result.manifest.to_dict()
    payload["recruiter_report"] = load_report_json(files.get("report_json"))
    payload["audit_report"] = load_report_json(files.get("audit_json"))
    payload["evidence_ledger"] = load_report_json(files.get("ledger_json"))
    payload["fairness_risk"] = (payload.get("audit_report", {}) or {}).get("fairness_risk_level", "info")

    return payload


def load_report_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ── Tab: Overview ────────────────────────────────────────────────────────────

def render_ranking_row(row: dict) -> None:
    rank = row.get("rank", 0)
    name = row.get("candidate_name") or row.get("candidate_id", "")
    score = row.get("final_score", 0.0)
    label = row.get("score_label", "moderate")
    cid = row.get("candidate_id", "")
    leader = str(rank) == "1"

    cols = st.columns([0.07, 0.27, 0.14, 0.22, 0.3])
    with cols[0]:
        st.markdown(rank_mono(rank, leader=leader), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            f"<span style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink)'>{name}</span>"
            f"<br>{id_tag(cid)}",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;font-size:1.05rem;"
            f"color:var(--ink)'>{score:.1f}</span>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(spectral_signature(row), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(tier_meter(label), unsafe_allow_html=True)
    st.markdown("<div style='border-bottom:1px solid var(--line);margin:.3rem 0 .6rem'></div>", unsafe_allow_html=True)


def render_overview_tab(manifest: dict, ranked: list[dict]) -> None:
    n = len(ranked)
    top_score = ranked[0]["final_score"] if ranked else 0
    verified_avg = sum(r.get("verified_claims", 0) for r in ranked) / n if n else 0

    st.markdown("<br>", unsafe_allow_html=True)
    readout_strip([
        ("Candidates", str(n)),
        ("Top score", f"{top_score:.0f}"),
        ("Avg verified claims", f"{verified_avg:.1f}"),
        ("Fairness", manifest.get("fairness_risk", "info").upper()),
        ("Run time", f"{manifest.get('duration_s', 0):.1f}s"),
    ])

    if manifest.get("warnings"):
        with st.expander(f"{len(manifest['warnings'])} pipeline warning(s)"):
            for w in manifest["warnings"]:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;"
                    f"color:var(--spec-seniority)'>· {w}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Ranking")
    spectrum_rule(2)
    for row in ranked[:15]:
        render_ranking_row(row)


# ── Tab: Roster ──────────────────────────────────────────────────────────────

def _card_map(report: Optional[dict]) -> dict[str, dict]:
    if not report:
        return {}
    return {c["candidate_id"]: c for c in report.get("candidates", [])}


def _ledger_map(ledger: Optional[dict]) -> dict[str, dict]:
    if not ledger:
        return {}
    return {c["candidate_id"]: c for c in ledger.get("candidates", [])}


def _stability_map(audit: Optional[dict]) -> dict[str, dict]:
    if not audit:
        return {}
    return {row["candidate_id"]: row for row in audit.get("candidate_stability", [])}


def _ranked_map(ranked: list[dict]) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in ranked}


def render_roster_tab(ranked: list[dict], report: Optional[dict]) -> None:
    cards = _card_map(report)
    st.markdown(
        "<div style='font-family:Source Serif 4,serif;font-size:.86rem;color:var(--ink-muted);"
        "margin-bottom:.8rem'>Expand any candidate for the full report, or use the "
        "<b>Candidate</b> tab for a single-candidate deep dive.</div>",
        unsafe_allow_html=True,
    )
    for row in ranked:
        cid = row["candidate_id"]
        card = cards.get(cid, row)
        render_roster_card(card, expanded=False)


# ── Tab: Candidate ───────────────────────────────────────────────────────────

def render_candidate_deep_dive(
    candidate_id: str,
    card: dict,
    ledger_entry: Optional[dict],
    stability: Optional[dict],
    ranked_row: Optional[dict],
    audit: Optional[dict],
) -> None:
    name = card.get("candidate_name") or candidate_id
    rank = card.get("rank", ranked_row.get("rank") if ranked_row else "?")
    score = card.get("final_score", 0.0)
    label = card.get("score_label", "moderate")
    conf = card.get("rank_confidence", "medium")
    leader = str(rank) == "1"

    st.markdown(
        f"<div class='prism-card{' leader' if leader else ''}' style='display:flex;align-items:center;gap:1.1rem'>"
        f"{rank_mono(rank, leader=leader)}"
        f"<div style='flex:1'>"
        f"<div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:800;"
        f"color:var(--ink)'>{name}</div>"
        f"<div style='margin-top:.2rem'>{id_tag(candidate_id)} &nbsp; {conf_tag(conf)}</div>"
        f"</div>"
        f"<div style='text-align:right'>{tier_meter(label)}<div style='font-family:IBM Plex Mono,monospace;"
        f"font-size:1.5rem;font-weight:800;color:var(--ink);margin-top:.2rem'>{score:.1f}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    tab_summary, tab_ledger, tab_scoring, tab_rank, tab_interview, tab_audit = st.tabs(
        ["Summary", "Evidence ledger", "Scoring", "Rank rationale", "Interview prep", "Audit flags"]
    )

    with tab_summary:
        if card.get("summary"):
            st.markdown(
                f"<div style='font-family:Source Serif 4,serif;font-size:.92rem;color:var(--ink);"
                f"line-height:1.8;background:var(--paper);border-radius:8px;padding:1rem 1.2rem'>"
                f"{card['summary']}</div>",
                unsafe_allow_html=True,
            )
        c1, c2 = st.columns(2)
        with c1:
            section_header("Strengths")
            for s in card.get("strengths", ranked_row.get("strengths", []) if ranked_row else []):
                st.markdown(f"<div style='font-size:.88rem;color:var(--ink);margin:.25rem 0'>+ {s}</div>", unsafe_allow_html=True)
        with c2:
            section_header("Risks & gaps")
            risks = card.get("risks", ranked_row.get("risks", []) if ranked_row else [])
            if risks:
                for r in risks:
                    st.markdown(f"<div style='font-size:.88rem;color:var(--ink);margin:.25rem 0'>− {r}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:var(--spec-domain);font-size:.86rem'>No critical risks flagged.</span>", unsafe_allow_html=True)

        section_header("Verified vs unverified skills")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(skill_chips(card.get("verified_skills", []), "verified"), unsafe_allow_html=True)
        with s2:
            st.markdown(skill_chips(card.get("unverified_skills", []), "unverified"), unsafe_allow_html=True)

    with tab_ledger:
        section_header("Claim verification")
        claims = ledger_entry.get("claims", []) if ledger_entry else []
        verified = sum(1 for c in claims if (c.get("verification_status") or "").lower() == "verified")
        readout_strip([
            ("Total claims", str(len(claims))),
            ("Verified", str(verified)),
            ("Proof strength", f"{card.get('proof_strength', 0):.2f}"),
        ])
        render_ledger_rows(claims)

    with tab_scoring:
        section_header("Rubric dimensions")
        for key, lbl, var_color, _ in DIMENSIONS:
            dim_row(lbl, card.get(key, 0), var_color)
        st.markdown("<br>", unsafe_allow_html=True)
        render_weights()
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Score evidence")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.85rem;color:var(--ink);line-height:1.9'>"
            f"verified claims — {card.get('verified_claims', 0)}<br>"
            f"unverified claims — {card.get('unverified_claims', 0)}<br>"
            f"final score — {score:.1f}/100 ({label})</div>",
            unsafe_allow_html=True,
        )

    with tab_rank:
        gap = card.get("score_gap_to_next")
        if gap is not None:
            st.markdown(
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:.78rem;color:var(--ink-muted)'>"
                f"SCORE GAP TO NEXT</span><br>"
                f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:800;"
                f"color:var(--spec-proof)'>+{gap:.1f} pts</span>",
                unsafe_allow_html=True,
            )
        why = card.get("why_above_next", "")
        if why:
            st.markdown(
                f"<div style='border-left:3px solid var(--spec-proof);padding:1rem 1.2rem;margin-top:.8rem;"
                f"background:var(--paper);font-family:Source Serif 4,serif;font-size:.9rem;color:var(--ink)'>{why}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span style='color:var(--ink-muted);font-size:.85rem'>No pairwise rank justification available.</span>", unsafe_allow_html=True)

        if stability:
            stable = stability.get("is_stable", True)
            color = "var(--spec-domain)" if stable else "var(--spec-skill)"
            st.markdown(
                f"<div style='margin-top:.9rem;font-family:IBM Plex Mono,monospace;font-size:.82rem;color:var(--ink)'>"
                f"<span style='color:{color};font-weight:700'>{'STABLE' if stable else 'UNSTABLE'}</span> &nbsp; "
                f"observed range {stability.get('min_rank_observed', '?')}–{stability.get('max_rank_observed', '?')}, "
                f"σ={stability.get('score_std', 0):.3f}</div>",
                unsafe_allow_html=True,
            )

    with tab_interview:
        render_interview_questions(card.get("interview_questions", []))

    with tab_audit:
        flags = []
        if audit:
            for flag in audit.get("fairness_flags", []):
                if candidate_id in flag.get("affected_ids", []):
                    flags.append(flag)
            if candidate_id in audit.get("unstable_candidates", []):
                flags.append({
                    "proxy_field": "rank_stability",
                    "description": "Rank is unstable across perturbation runs.",
                    "severity": "warning",
                })
        if flags:
            for flag in flags:
                sev = flag.get("severity", "info")
                color = "var(--spec-skill)" if sev == "high" else "var(--spec-seniority)" if sev == "warning" else "var(--ink-muted)"
                st.markdown(
                    f"<div style='border-left:3px solid {color};padding:.6rem .9rem;margin:.4rem 0'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;font-size:.78rem'>"
                    f"{flag.get('proxy_field', 'flag').upper()}</span>"
                    f"<div style='font-family:Source Serif 4,serif;font-size:.86rem;color:var(--ink);"
                    f"margin-top:.2rem'>{flag.get('description', '')}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<span style='color:var(--spec-domain);font-size:.85rem'>No audit flags for this candidate.</span>", unsafe_allow_html=True)


def render_candidate_tab(
    ranked: list[dict],
    report: Optional[dict],
    ledger: Optional[dict],
    audit: Optional[dict],
) -> None:
    if not ranked:
        st.markdown("<span style='color:var(--ink-muted)'>Run the pipeline first to generate candidate reports.</span>", unsafe_allow_html=True)
        return

    cards = _card_map(report)
    ledgers = _ledger_map(ledger)
    stability = _stability_map(audit)
    ranked_by_id = _ranked_map(ranked)

    options = [
        f"№{int(row.get('rank', 0)):02d} — {row.get('candidate_name') or row['candidate_id']} "
        f"({row.get('final_score', 0):.0f} pts)"
        for row in ranked
    ]
    selected = st.selectbox("Select candidate", options, label_visibility="collapsed")
    idx = options.index(selected)
    cid = ranked[idx]["candidate_id"]

    card = cards.get(cid, ranked[idx])
    render_candidate_deep_dive(
        candidate_id=cid,
        card=card,
        ledger_entry=ledgers.get(cid),
        stability=stability.get(cid),
        ranked_row=ranked_by_id.get(cid),
        audit=audit,
    )


# ── Tab: Fairness & stability ────────────────────────────────────────────────

def render_fairness_tab(audit: Optional[dict], manifest: dict) -> None:
    if not audit:
        st.markdown("<span style='font-family:IBM Plex Mono,monospace;color:var(--ink-muted)'>No audit report available for this run.</span>", unsafe_allow_html=True)
        return

    st.markdown("<br>", unsafe_allow_html=True)
    readout_strip([
        ("Fairness risk", audit.get("fairness_risk_level", "info").upper()),
        ("Stability ratio", f"{audit.get('stability_ratio', 0):.0%}"),
        ("Unstable", str(len(audit.get("unstable_candidates", [])))),
    ])

    section_header("Fairness flags")
    flags = audit.get("fairness_flags", [])
    if flags:
        for flag in flags:
            sev = flag.get("severity", "info")
            color = "var(--spec-skill)" if sev == "high" else "var(--spec-seniority)" if sev == "warning" else "var(--ink-muted)"
            st.markdown(
                f"<div style='border-left:3px solid {color};padding:.6rem .9rem;margin:.4rem 0;"
                f"background:var(--surface);border-radius:0 6px 6px 0'>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;font-size:.78rem;"
                f"letter-spacing:.04em;color:var(--ink)'>{flag.get('proxy_field', '').upper()}</span>"
                f"<div style='font-family:Source Serif 4,serif;font-size:.86rem;color:var(--ink);"
                f"margin-top:.25rem'>{flag.get('description', '')}</div>"
                + (f"<div style='font-family:Source Serif 4,serif;font-size:.8rem;color:var(--ink-muted);"
                   f"margin-top:.2rem;font-style:italic'>{flag.get('recommendation', '')}</div>"
                   if flag.get("recommendation") else "")
                + "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<span style='color:var(--spec-domain);font-family:IBM Plex Mono,monospace;font-size:.85rem'>No fairness proxy flags detected.</span>", unsafe_allow_html=True)

    section_header("Top-k impact ratios")
    for ratio in audit.get("top_k_ratios", []):
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.82rem;color:var(--ink);margin:.25rem 0'>"
            f"{ratio.get('attribute')} = {ratio.get('value')} — {ratio.get('top_k_ratio', 0):.0%} in top-{ratio.get('k')} "
            f"vs {ratio.get('baseline_ratio', 0):.0%} baseline (×{ratio.get('impact_ratio', 1):.1f})</div>",
            unsafe_allow_html=True,
        )

    section_header("Rank stability by candidate")
    for row in audit.get("candidate_stability", []):
        stable = row.get("is_stable", True)
        color = "var(--spec-domain)" if stable else "var(--spec-skill)"
        mark = "STABLE" if stable else "UNSTABLE"
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:var(--ink);margin:.2rem 0'>"
            f"<span style='color:{color};font-weight:700'>{mark}</span> &nbsp; {row.get('candidate_id')} — "
            f"base rank {row.get('base_rank')} (range {row.get('rank_range', 0)}, σ={row.get('score_std', 0):.3f})</div>",
            unsafe_allow_html=True,
        )


# ── Tab: Pipeline ────────────────────────────────────────────────────────────

def render_pipeline_tab(manifest: dict) -> None:
    section_header("Layer execution log")
    for layer in manifest.get("layer_records", []):
        status = (layer.get("status") or "pending").lower()
        color = ("var(--spec-domain)" if status == "complete" else
                  "var(--spec-seniority)" if status == "skipped" else
                  "var(--spec-skill)" if status == "error" else "var(--ink-muted)")
        try:
            layer_no = int(layer.get("layer", 0))
        except (TypeError, ValueError):
            layer_no = 0
        st.markdown(
            f"<div style='display:flex;gap:.8rem;align-items:baseline;padding:.5rem 0;"
            f"border-bottom:1px solid var(--line)'>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-weight:700;color:var(--ink-muted);"
            f"width:34px'>{layer_no:02d}</span>"
            f"<span style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink);flex:1'>"
            f"{layer.get('name')}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{color};"
            f"font-weight:700;letter-spacing:.05em'>{status.upper()}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:var(--ink-muted);"
            f"width:55px;text-align:right'>{layer.get('duration', 0):.2f}s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if layer.get("notes"):
            st.markdown(
                f"<div style='font-family:Source Serif 4,serif;font-size:.82rem;color:var(--ink-muted);"
                f"margin:.15rem 0 .4rem 44px'>{layer.get('notes')}</div>",
                unsafe_allow_html=True,
            )
        if layer.get("error"):
            st.markdown(
                f"<div style='color:var(--spec-skill);font-family:IBM Plex Mono,monospace;"
                f"font-size:.78rem;margin-left:44px'>{layer['error']}</div>",
                unsafe_allow_html=True,
            )


# ── Tab: Export ──────────────────────────────────────────────────────────────

def render_export_tab(manifest: dict) -> None:
    files = manifest.get("output_files", {})
    labels = {
        "ranked_json": "Ranked output",
        "report_json": "Recruiter report (JSON)",
        "report_md": "Recruiter report (Markdown)",
        "audit_json": "Audit report",
        "ledger_json": "Evidence ledger",
        "manifest_json": "Run manifest",
    }
    section_header("Export run artefacts")
    cols = st.columns(2)
    i = 0
    for key, label in labels.items():
        path = files.get(key)
        if not path or not Path(path).exists():
            continue
        data = Path(path).read_bytes()
        with cols[i % 2]:
            st.markdown(
                f"<div class='prism-card' style='padding:.8rem 1rem;margin-bottom:.6rem'>"
                f"<div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:var(--ink)'>{label}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:var(--ink-muted);"
                f"margin:.2rem 0 0'>{Path(path).name}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download",
                data=data,
                file_name=Path(path).name,
                mime="application/json" if path.endswith(".json") else "text/markdown",
                use_container_width=True,
                key=f"dl_{key}",
            )
        i += 1


# ── Input loading ────────────────────────────────────────────────────────────

def _load_sample_inputs() -> tuple[str, list[tuple[str, str]]]:
    from ai_hiring_ranker.ingestion.loader import ingest

    sample_jd = ROOT / "data" / "sample" / "jd.txt"
    sample_dir = ROOT / "data" / "sample" / "candidates"
    result = ingest(jd_path=sample_jd, candidates_dir=sample_dir)
    texts = [(c.candidate_id, c.raw_text) for c in result.candidates]
    return result.jd.raw_text, texts


def _load_upload_inputs(jd_text: str, uploaded_files: list) -> tuple[str, list[tuple[str, str]]]:
    from ai_hiring_ranker.ingestion.parsers import extract_text

    candidate_texts: list[tuple[str, str]] = []
    for i, uploaded in enumerate(uploaded_files, start=1):
        suffix = Path(uploaded.name).suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        text = extract_text(tmp_path)
        cid = Path(uploaded.name).stem or f"C{i:03d}"
        candidate_texts.append((cid, text))
        tmp_path.unlink(missing_ok=True)
    return jd_text, candidate_texts


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:

    st.markdown(
        f"""
        <div class="hero">

            <div style="display:flex;align-items:center;gap:1.5rem;">

                <div>
                    {prism_logo_svg(65)}
                </div>

                <div>

                    <div class="hero-title">
                        PRISM
                    </div>

                    <div class="hero-subtitle">
                        Proof-driven Ranking &amp; Intelligent Selection Model
                    </div>

                    <div style="
                        margin-top:.6rem;
                        font-family:'IBM Plex Mono', monospace;
                        font-size:.85rem;
                        color:var(--ink-muted);
                        letter-spacing:.05em;
                    ">
                        ✦ Evidence-backed Candidate Ranking <br>
                        ✦ 15-Layer Intelligent Hiring Pipeline
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    spectrum_rule(6)

    (mode, jd_text, candidate_files, force_fallback, shortlist_k, stab_runs, run_btn, use_sample) = render_sidebar()

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    if run_btn:
        try:
            with st.spinner("Running the 15-layer pipeline…"):
                if use_sample:
                    jd_text, candidate_texts = _load_sample_inputs()
                else:
                    if not jd_text.strip():
                        st.error("Paste a job description first.")
                        st.stop()
                    if not candidate_files:
                        st.error("Upload at least one candidate resume.")
                        st.stop()
                    jd_text, candidate_texts = _load_upload_inputs(jd_text, candidate_files)

                st.session_state.pipeline_result = run_v2_pipeline(
                    jd_text=jd_text,
                    candidate_texts=candidate_texts,
                    force_fallback=force_fallback,
                    k=shortlist_k,
                    stability_runs=stab_runs,
                )
            st.success("Pipeline complete.")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    result = st.session_state.pipeline_result
    if result is None:
        st.markdown(
            "<div class='prism-card' style='text-align:center;padding:2.6rem 1.5rem'>"
            "<div style='font-family:Space Grotesk,sans-serif;font-size:1.15rem;font-weight:700;"
            "color:var(--ink)'>Nothing to refract yet</div>"
            "<div style='font-family:Source Serif 4,serif;font-size:.88rem;color:var(--ink-muted);"
            "margin-top:.5rem'>Choose sample data or upload resumes, then click "
            "<b>Run pipeline</b> in the sidebar.</div></div>",
            unsafe_allow_html=True,
        )
        return

    manifest = result
    ranked = result.get("ranked_output", [])
    report = result.get("recruiter_report")
    audit = result.get("audit_report")
    ledger = result.get("evidence_ledger")

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.78rem;color:var(--ink-muted)'>"
        f"run {manifest.get('run_id', '')} · {manifest.get('job_title', 'Job')} · "
        f"{manifest.get('candidate_count', len(ranked))} candidates · "
        f"{manifest.get('duration_s', 0):.1f}s</div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_roster, tab_candidate, tab_fairness, tab_pipeline, tab_export = st.tabs(
        ["Overview", "Roster", "Candidate", "Fairness & stability", "Pipeline", "Export"]
    )

    with tab_overview:
        render_overview_tab(manifest, ranked)
    with tab_roster:
        render_roster_tab(ranked, report)
    with tab_candidate:
        render_candidate_tab(ranked, report, ledger, audit)
    with tab_fairness:
        render_fairness_tab(audit, manifest)
    with tab_pipeline:
        render_pipeline_tab(manifest)
    with tab_export:
        render_export_tab(manifest)


if __name__ == "__main__":
    main()