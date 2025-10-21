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
# Session defaults
# ──────────────────────────────────────────────────────────────────────────────
st.session_state.setdefault("ptr", 0)           # reroll pointer
st.session_state.setdefault("show_result", False)  # start with no result shown
st.session_state.setdefault("help_open", False)    # tutorial expander state

# ──────────────────────────────────────────────────────────────────────────────
# Header + Tutorial with toggle button
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")

with st.sidebar:
    # Toggle the expander every click (open if closed; close if open)
    if st.button("❓ Help / Tutorial", use_container_width=True):
        st.session_state.help_open = not st.session_state.help_open

with st.expander("📘 How to use (full tutorial)", expanded=st.session_state.help_open):
    st.markdown(
        """
**What this app does**  
Find Indonesian recipes by combining **semantic search** (BERT) and **keyword search** (TF-IDF).  
Enter 1–2 ingredients, tweak options, hit **Find**, then **Reroll** to cycle candidates.

---

### 1) Basic flow
1. Pick a **Dish type** (main / side / snack).  
2. Type **Main ingredient 1/2** (optional): e.g., `ayam`, `nasi`, `kecap`, `cabe`, `cokelat`.  
3. Leave **Retrieval strategy = Auto** unless you want manual control.  
4. (Optional) **Boost exact word matches** or **Require both ingredients (AND)**.  
5. Click **Find** → **Reroll** to browse more.

---

### 2) Retrieval strategy
- **Auto (recommended)** – smartly balances BERT/TF-IDF; switches to two-stage on big sets.  
- **Hybrid** – fixed blend `score = α·BERT + (1−α)·TF-IDF`. Control **BERT weight**.  
- **Two-stage** – TF-IDF picks **top-K** → BERT re-ranks those K (faster on large sets).

**Tips**: Higher **BERT weight** = more semantic; lower = more literal keywords.

---

### 3) Matching options
- **Boost exact word matches** – bonus when your tokens appear in Title/Ingredients.  
  **Exact-match weight** controls strength (0.25 gentle → 1.0 strong).  
- **Require both ingredients (AND)** – only recipes that contain **both** ingredient 1 **and** 2 (exact tokens).

---

### 4) Examples
- Main dish with soy-sauce chicken: `ayam` + `kecap`, **Auto**, Boost on (0.25–0.6).  
- Snack ideas with tofu: Dish = snack, Ingredient 1 = `tahu`.  
- Strict “ayam + cabe”: enable **AND**.

---

### 5) Troubleshooting
- **No results** → turn off **AND**, lower **Exact-match weight**, or try synonyms (`cabai`/`cabe`, `mie`/`mi`).  
- **Slow** → use **Two-stage** and reduce **top-K** (e.g., 200).  
- **Weird matches** → lower **BERT weight** or enable a small exact-match boost.
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search")
    category = st.selectbox("Dish type", ["main dish","side dish","snack"])
    ing1 = st.text_input("Main ingredient 1", "")
    ing2 = st.text_input("Main ingredient 2", "")

    st.markdown("### Retrieval strategy")
    strategy = st.radio(" ", ["auto","hybrid","two-stage"], index=0, label_visibility="collapsed")
    alpha = st.slider("BERT weight (hybrid)", 0.0, 1.0, 0.7, 0.05)
    k_pref = st.slider("Two-stage: top-K from TF-IDF", 50, 2000, 300, 50)

    st.markdown("### Matching options")
    boost_exact = st.checkbox("Boost exact word matches", value=False)
    boost_weight = st.slider("Exact-match weight", 0.0, 1.5, 0.25, 0.05)
    require_both = st.checkbox("Require both ingredients (AND)", value=False)

    btn_find  = st.button("🔎 Find",  use_container_width=True)
    btn_reroll= st.button("🎲 Reroll",use_container_width=True)
    btn_reset = st.button("↩️ Reset", use_container_width=True)

# Button logic (controls both pointer and whether to show results)
if btn_reset:
    st.session_state.ptr = 0
    st.session_state.show_result = False

if btn_find:
    st.session_state.ptr = 0
    st.session_state.show_result = True

if btn_reroll:
    st.session_state.ptr += 1
    st.session_state.show_result = True

# ──────────────────────────────────────────────────────────────────────────────
# Main result pane (blank by default until Find/Reroll)
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1.9])

if st.session_state.show_result:
    # only run the expensive search when we actually want a result
    row, pool_size, total = find_one(
        category, ing1, ing2, strategy, alpha, k_pref,
        boost_exact, boost_weight, require_both, st.session_state.ptr
    )
else:
    row, pool_size, total = None, 0, 0  # blank state

with left:
    st.markdown("#### Candidate")
    if st.session_state.show_result and total > 0:
        st.progress(min(1.0, ((st.session_state.ptr % total) + 1) / float(total)))
        st.caption(f"Candidate # {(st.session_state.ptr % total) + 1} of {total}  •  Pool size: {pool_size}")
    else:
        st.progress(0.0)
        st.caption("No candidate yet — enter ingredients and click **Find**.")

with right:
    box = st.container()
    with box:
        if not st.session_state.show_result:
            st.subheader("—")
            st.write("Start by entering an ingredient and pressing **Find**.")
        elif row is None:
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
