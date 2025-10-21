# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re

from core import merged, rank_hybrid_filtered, candidate_total

st.set_page_config(page_title="Indonesian Recipe Gacha", page_icon="🍳", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _split_to_items(text: str):
    if not isinstance(text, str):
        return []
    parts = re.split(r'[\n,;•\-]+', text)
    return [p.strip() for p in parts if p.strip()]

def _to_bullets(items):
    return "\n".join(f"- {x}" for x in items) if items else "- —"

def _to_steps(items):
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, start=1)) if items else "1. —"

def find_one(category, a, b, strategy, alpha, k_pref, boost, boost_w, require_both, ptr):
    """Return (row or None, pool_size:int, total:int) using your real ranker."""
    idxs, picked = rank_hybrid_filtered(
        category, a, b, ptr=ptr,
        alpha=float(alpha), k_prefilter=int(k_pref),
        strategy=strategy,
        boost_exact=bool(boost), boost_weight=float(boost_w),
        require_both=bool(require_both),
    )
    pool_size = 0 if idxs is None else len(idxs)
    total = candidate_total(category, strategy, int(k_pref), a, b, require_both)
    if not picked:
        return None, pool_size, max(0, total)
    return merged.iloc[picked[0]], pool_size, max(1, total)

# ──────────────────────────────────────────────────────────────────────────────
# Header + Tutorial (collapsible) + optional sidebar “open help” button
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")

# allow sidebar button to open the tutorial dropdown
if "show_help" not in st.session_state:
    st.session_state.show_help = False

with st.sidebar:
    if st.button("❓ Help / Tutorial", use_container_width=True):
        st.session_state.show_help = True

with st.expander("📘 How to use (full tutorial)", expanded=st.session_state.show_help):
    st.session_state.show_help = False  # reset the flag after opening once
    st.markdown(
        """
**What this app does**  
Find Indonesian recipes by combining **semantic search** (BERT) and **keyword search** (TF-IDF).  
You can ask for 1–2 ingredients, tweak matching behavior, then **Find** and **Reroll** to browse candidates.

---

### 1) Basic flow
1. Pick a **Dish type** (main / side / snack).  
2. Enter **Main ingredient 1/2** (optional). e.g., `ayam`, `nasi`, `kecap`, `cabe`, `cokelat`.  
3. Leave **Retrieval strategy = Auto** (smart) unless you want manual control.  
4. (Optional) Turn on **Boost exact word matches** or **Require both ingredients (AND)**.  
5. Click **Find** → click **Reroll** to cycle through other strong matches.

---

### 2) Retrieval strategy
- **Auto (recommended)** – automatically mixes BERT + TF-IDF, and switches to two-stage when the set is large.  
  *Rare words → more TF-IDF; Common words → more BERT; Large pools → TF-IDF prefilter then BERT.*
- **Hybrid** – fixed formula: `score = α·BERT + (1 − α)·TF-IDF`.  
  Control α with **BERT weight**.
- **Two-stage** – TF-IDF selects **top-K** → BERT re-ranks those K.  
  Faster on big sets; adjust **K** with the *Two-stage: top-K from TF-IDF* slider.

**Tips**
- If results feel too “literal”, raise **BERT weight** (more semantic).  
- If results feel too “fuzzy”, lower **BERT weight** (more exact keywords).  
- For speed on large categories, prefer **Two-stage** or keep **Auto**.

---

### 3) Matching options
- **Boost exact word matches** – give extra score to recipes that literally contain your ingredients
  in Title/Ingredients.  
  **Exact-match weight** controls the strength (0.25 = gentle, 1.0 = strong).
- **Require both ingredients (AND)** – shows only recipes that contain **both** ingredient 1 **and** 2  
  (exact token presence). Turn this off if you get “no results”.

---

### 4) Examples
- *Find a soy-sauce chicken*:  
  - Dish: **main dish**, Ingredients: `ayam`, `kecap`.  
  - Strategy: **Auto**, Boost: **on**, Exact-match weight: **0.25–0.6**.
- *Crispy tofu snacks*:  
  - Dish: **snack**, Ingredients: `tahu`, *(leave second blank)*.  
  - Strategy: **Hybrid**, α = **0.6**.
- *Strictly “ayam + cabe” together*:  
  - Dish: **main dish**, Ingredients: `ayam`, `cabe`.  
  - **Require both ingredients (AND)** ✅.

---

### 5) Troubleshooting
- **“No recipe matched”** – try turning off **AND**, reduce **Exact-match weight**, or tweak words
  (`cabai` vs `cabe`, `mie` vs `mi`).  
- **Too slow** – switch to **Two-stage** and reduce **top-K** (e.g., 200).  
- **Weird matches** – lower **BERT weight** (more literal), or turn on **Boost exact** a little.

Enjoy exploring! 🍜
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Controls (with tooltips)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search")
    category = st.selectbox(
        "Dish type",
        ["main dish","side dish","snack"],
        help="Which recipe bucket to search in."
    )
    ing1 = st.text_input("Main ingredient 1", "", help="e.g., ayam / nasi / lemon")
    ing2 = st.text_input("Main ingredient 2", "", help="e.g., cabe / kecap / cokelat")

    st.markdown("### Retrieval strategy")
    strategy = st.radio(
        " ",
        ["auto","hybrid","two-stage"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Auto: smart blend + two-stage when needed.\n"
            "Hybrid: fixed α·BERT + (1−α)·TF-IDF.\n"
            "Two-stage: TF-IDF prefilter (top-K) then BERT re-rank."
        )
    )

    alpha = st.slider(
        "BERT weight (hybrid)",
        0.0, 1.0, 0.7, 0.05,
        help="Higher = more semantic. Lower = more literal keywords. Used by Hybrid (and used internally by Auto)."
    )
    k_pref = st.slider(
        "Two-stage: top-K from TF-IDF",
        50, 2000, 300, 50,
        help="How many TF-IDF candidates get re-ranked by BERT in two-stage."
    )

    st.markdown("### Matching options")
    boost_exact = st.checkbox(
        "Boost exact word matches", value=False,
        help="Give a score bonus when your exact tokens appear in Title/Ingredients."
    )
    boost_weight = st.slider(
        "Exact-match weight", 0.0, 1.5, 0.25, 0.05,
        help="Strength of that exact-token bonus."
    )
    require_both = st.checkbox(
        "Require both ingredients (AND)", value=False,
        help="Only keep recipes that contain BOTH ingredient 1 AND ingredient 2 (exact tokens)."
    )

    find = st.button("🔎 Find", use_container_width=True)
    reroll = st.button("🎲 Reroll", use_container_width=True)
    reset = st.button("↩️ Reset", use_container_width=True)

# keep position across rerolls
if "ptr" not in st.session_state:
    st.session_state.ptr = 0

# button behavior
if reset:
    st.session_state.ptr = 0
if find:
    st.session_state.ptr = 0
if reroll:
    st.session_state.ptr += 1

# ──────────────────────────────────────────────────────────────────────────────
# Main result pane
# ──────────────────────────────────────────────────────────────────────────────
row, pool_size, total = find_one(
    category, ing1, ing2, strategy, alpha, k_pref,
    boost_exact, boost_weight, require_both, st.session_state.ptr
)

left, right = st.columns([1.1, 1.9])

with left:
    st.markdown("#### Candidate")
    if total > 0:
        st.progress(min(1.0, ((st.session_state.ptr % total) + 1) / float(total)))
        st.caption(f"Candidate # {(st.session_state.ptr % total) + 1} of {total}  •  Pool size: {pool_size}")

with right:
    cont = st.container()
    with cont:
        if row is None:
            st.subheader("—")
            msg = "No recipe matched your settings."
            if require_both and (ing1.strip() and ing2.strip()):
                msg += " Try turning off **AND** or adjusting keywords."
            st.write(msg)
        else:
            st.subheader(str(row.get("Title","—")))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ingredients**")
                st.text(_to_bullets(_split_to_items(row.get("Ingredients",""))))
            with c2:
                st.markdown("**Steps**")
                st.text(_to_steps(_split_to_items(row.get("Steps",""))))
            url = str(row.get("URL","")).strip()
            if url:
                st.markdown(f"**Source**: {url}")
            st.caption(
                f"Category: {row.get('Category','?')}  •  Loves: {row.get('Loves','-')}  •  Source file: {row.get('Source','-')}"
            )
