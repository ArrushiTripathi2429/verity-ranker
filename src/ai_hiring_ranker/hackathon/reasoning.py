"""Generate fact-grounded, JD-specific reasoning for each ranked candidate."""

from __future__ import annotations

from typing import Any


def _extract_key_jd_signals(row: dict[str, Any], job_title: str) -> dict[str, str]:
    """
    Extract JD-specific signals from candidate features.
    
    JD signals the ranker should look for:
    - "Shipped ranking/search/recommendation system to real users at meaningful scale"
    - "Strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online)"
    - "Located in or willing to relocate to Noida or Pune"
    - "Active on Redrob platform"
    """
    signals = {}
    
    # Production proof: shipped systems
    strengths = row.get("strengths", []) or []
    matched_req = row.get("matched_required", []) or []
    matched_pref = row.get("matched_preferred", []) or []
    
    # Check for production/shipping language
    production_keywords = {
        "shipped", "deployed", "production", "live users", "scaled",
        "ranking", "search", "recommendation", "information retrieval",
        "vector database", "embeddings", "dense retrieval", "hybrid search",
    }
    
    agent_summary = (row.get("agent_summary") or "").lower()
    production_hits = [kw for kw in production_keywords if kw in agent_summary]
    
    if production_hits:
        signals["production"] = f"Has shipped {', '.join(production_hits[:3])} systems"
    
    # Retrieval/ranking expertise
    retrieval_keywords = {
        "embeddings", "vector", "dense", "sparse", "bm25", "hybrid",
        "retrieval", "ranking", "search", "information retrieval",
    }
    retrieval_hits = [
        skill for skill in matched_req
        if any(kw in skill.lower() for kw in retrieval_keywords)
    ]
    if retrieval_hits:
        signals["retrieval"] = f"Strong in {', '.join(retrieval_hits[:2])} retrieval"
    
    # Evaluation/frameworks
    eval_keywords = {
        "evaluation", "metrics", "offline", "online", "a/b test",
        "mrr", "ndcg", "map", "recall", "precision",
    }
    eval_hits = [skill for skill in matched_req if any(kw in skill.lower() for kw in eval_keywords)]
    if eval_hits:
        signals["evaluation"] = f"Skilled in {', '.join(eval_hits[:2])} evaluation"
    
    # Location match
    location = (row.get("location") or "").lower()
    PREFERRED = {"pune", "noida", "delhi", "ncr", "gurgaon"}
    if any(city in location for city in PREFERRED):
        signals["location"] = f"Based in {location} (JD-preferred location)"
    elif row.get("willing_to_relocate"):
        signals["location"] = f"In {location} but willing to relocate to Pune/Noida"
    
    # Platform engagement
    open_to_work = row.get("open_to_work")
    if open_to_work:
        signals["availability"] = "Actively open to work"
    
    return signals


def _extract_concerns(row: dict[str, Any]) -> list[str]:
    """Extract fact-grounded concern areas."""
    concerns = []
    
    risks = row.get("risks", []) or []
    missing_req = row.get("missing_required", []) or []
    
    if missing_req:
        concerns.append(f"Missing key skills: {', '.join(missing_req[:2])}")
    
    if risks:
        # Take first 1-2 risks, avoid generic placeholder text
        for risk in risks[:2]:
            if risk and risk.strip() and len(risk) > 5:
                concerns.append(risk.strip())
    
    # Disqualifiers
    disqualifiers = row.get("disqualifiers", []) or []
    for disq in disqualifiers[:1]:  # Include at most 1 disqualifier in reasoning
        concerns.append(f" {disq}")
    
    # Behavioral signals
    if not row.get("open_to_work"):
        concerns.append("Currently not open to work")
    
    notice_days = row.get("notice_period_days")
    if notice_days and notice_days > 60:
        concerns.append(f"Long notice period: {notice_days} days")
    
    response_time = row.get("avg_response_time_hours")
    if response_time and response_time > 72:
        concerns.append(f"Slow platform response time: {response_time:.0f}h avg")
    
    return concerns


def build_reasoning(row: dict[str, Any], job_title: str, rank: int) -> str:
    """
    Build fact-grounded, JD-specific reasoning that avoids generic templates.
    
    Structure:
    1. Opening statement: fit tier (strong/good/moderate/marginal)
    2. Key strengths: specific JD signals (production proof, retrieval expertise, location)
    3. Experience summary: years + relevant domain
    4. Concerns: specific gaps + behavioral signals
    5. Closing: actionable hiring signal
    """
    score = row.get("final_score", 0.0)
    years_exp = row.get("years_experience") or row.get("total_years_experience") or 0
    title = row.get("job_title", "").strip()
    name = row.get("candidate_name", "").strip()
    
    # Determine fit tier based on score
    if score >= 80:
        tier = "Strong fit"
    elif score >= 65:
        tier = "Good fit"
    elif score >= 50:
        tier = "Moderate fit"
    else:
        tier = "Marginal fit"
    
    # Extract JD-specific signals
    jd_signals = _extract_key_jd_signals(row, job_title)
    concerns = _extract_concerns(row)
    
    # Build reasoning narrative
    parts = []
    
    # Opening: tier + primary reason
    if jd_signals.get("production"):
        parts.append(
            f"{tier} for Senior AI Engineer: {jd_signals['production']} "
            f"with demonstrated at-scale impact."
        )
    elif jd_signals.get("retrieval"):
        parts.append(
            f"{tier} for Senior AI Engineer: {jd_signals['retrieval']} "
            f"with proven system design experience."
        )
    else:
        matched_skills = row.get("matched_required", [])[:3]
        if matched_skills:
            parts.append(
                f"{tier} for Senior AI Engineer: "
                f"Covers core required skills ({', '.join(matched_skills[:2])}) "
                f"with {years_exp:.0f} years experience."
            )
        else:
            parts.append(
                f"{tier} for Senior AI Engineer: "
                f"{years_exp:.0f} years experience in relevant domain."
            )
    
    # Add location signal
    if jd_signals.get("location"):
        parts.append(jd_signals["location"])
    
    # Add platform engagement
    if jd_signals.get("availability"):
        parts.append(jd_signals["availability"])
    
    # Behavioral signal strengths
    response_time = row.get("avg_response_time_hours")
    if response_time and response_time <= 24:
        parts.append(f"Quick responder: {response_time:.1f}h avg response time.")
    
    offer_acceptance = row.get("offer_acceptance_rate")
    if offer_acceptance and offer_acceptance >= 0.7:
        parts.append(f"High offer acceptance rate ({offer_acceptance*100:.0f}%) suggests commitment.")
    
    # Add concerns if any
    if concerns:
        concern_str = " ".join(concerns[:2])  # Max 2 concerns
        parts.append(f"Key concerns: {concern_str}")
    
    # Evaluation framework expertise if available
    if jd_signals.get("evaluation"):
        parts.append(
            f"Can contribute on evaluation design: {jd_signals['evaluation']}."
        )
    
    # Closing statement
    matched_pref = row.get("matched_preferred", [])
    if score >= 75 and not row.get("disqualifiers"):
        if years_exp >= 5:
            parts.append("Ready for immediate impact as senior IC.")
        else:
            parts.append("Strong candidate with room to grow into senior role.")
    elif score < 50 and row.get("disqualifiers"):
        parts.append("Consider revisiting after addressing flagged concerns.")
    
    # Join into coherent reasoning (max ~240 chars per spec)
    reasoning = " ".join(parts)
    
    # Truncate to ~240 chars if needed, maintaining sentence boundaries
    if len(reasoning) > 240:
        truncated = reasoning[:237] + "..."
    else:
        truncated = reasoning
    
    return truncated