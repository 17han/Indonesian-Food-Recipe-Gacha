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
defaults = {
    "ptr": 0,
    "show_result": False,
    "help_open": False,
    "category": "main dish",
    "ing1": "",
    "ing2": "",
    "strategy": "auto",
    "alpha": 0.7,
    "k_pref": 300,
    "boost_exact": False,
    "boost_weight": 0.25,
    "require_both": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ──────────────────────────────────────────────────────────────────────────────
# Header + Tutorial with toggle
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")

with st.sidebar:
    if st.button("❓ Help / Tutorial", use_container_width=True):
        st.session_state.help_open = not st.session_state.help_open

with st.expander("📘 How to use (full tutorial)", expanded=st.session_state.help_open):
    st.markdown(
        """
**What this app does**  
Find Indonesian recipes using **semantic (BERT)** + **keyword (TF-IDF)** hybrid search.

---

### 🧭 Quick Steps
1. Pick **Dish type** (main / side / snack)  
2. Fill **Main ingredient 1/2** (optional)  
3. Leave **Auto** strategy (recommended)  
4. Adjust sliders or options if you like  
5. Click **Find** → **Reroll** to browse

---

### ⚙️ Strategy Guide
- **Auto** – smart blend of BERT + TF-IDF (adjusts automatically)  
- **Hybrid** – fixed blend `score = α·BERT + (1−α)·TF-IDF`  
- **Two-stage** – TF-IDF selects top-K → BERT re-ranks (faster for large sets)

**Tips:**  
- ↑ BERT weight → more *semantic* matches  
- ↓ BERT weight → more *literal* matches  

---

### 🧩 Matching Options
- **Boost exact word matches** → bonus for recipes literally containing your ingredient  
- **Exact-match weight** → how strong the boost is  
- **Require both ingredients (AND)** → only recipes with both ingredients

---

### 🧠 Examples
| Goal | Settings |
|------|-----------|
| Soy-sauce chicken | main dish, ayam + kecap, Boost on |
| Tofu snacks | snack, tahu |
| Spicy chicken only | main dish, ayam + cabe, AND on |

---

### 🆘 Troubleshooting
- “No recipe found” → turn off AND, lower exact-match weight, or change words  
- “Slow” → use Two-stage, lower top-K  
- “Weird matches” → lower BERT weight or enable Boost
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search")

    st.session_state.category = st.selectbox("Dish type", ["main dish", "side dish", "snack"], index=["main dish","side dish","snack"].index(st.session_state.category))
    st.session_state.ing1 = st.text_input("Main ingredient 1", st.session_state.ing1)
    st.session_state.ing2 = st.text_input("Main ingredient 2", st.session_state.ing2)

    st.markdown("### Retrieval strategy")
    st.session_state.strategy = st.radio(" ", ["auto","hybrid","two-stage"],
                                         index=["auto","hybrid","two-stage"].index(st.session_state.strategy),
                                         label_visibility="collapsed")

    # ─── Sliders with reset buttons ───
    def resettable_slider(label, key, min_value, max_value, value, step, help_text=""):
        c1, c2 = st.columns([5,1])
        with c1:
            st.session_state[key] = st.slider(label, min_value=min_value, max_value=max_value,
                                              value=st.session_state.get(key, value), step=step, help=help_text)
        with c2:
            if st.button("↩️", key=f"reset_{key}", help=f"Reset {label}"):
                st.session_state[key] = value

    resettable_slider("BERT weight (hybrid)", "alpha", 0.0, 1.0, 0.7, 0.05,
                      "Higher = more semantic, Lower = more literal keywords.")
    resettable_slider("Two-stage: top-K from TF-IDF", "k_pref", 50, 2000, 300, 50,
                      "How many TF-IDF candidates to pass to BERT.")

    st.markdown("### Matching options")
    st.session_state.boost_exact = st.checkbox("Boost exact word matches", value=st.session_state.boost_exact)
    st.session_state.boost_weight = st.slider("Exact-match weight", 0.0, 1.5, st.session_state.boost_weight, 0.05)
    st.session_state.require_both = st.checkbox("Require both ingredients (AND)", value=st.session_state.require_both)

    # ─── Main buttons ───
    btn_find = st.button("🔎 Find", use_container_width=True)
    btn_reroll = st.button("🎲 Reroll", use_container_width=True)
    btn_reset = st.button("🧹 Reset All", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Button logic
# ──────────────────────────────────────────────────────────────────────────────
if btn_reset:
    for k, v in defaults.items():
        st.session_state[k] = v

if btn_find:
    st.session_state.ptr = 0
    st.session_state.show_result = True

if btn_reroll:
    st.session_state.ptr += 1
    st.session_state.show_result = True

# ──────────────────────────────────────────────────────────────────────────────
# Main results
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1.9])

if st.session_state.show_result:
    row, pool_size, total = find_one(
        st.session_state.category, st.session_state.ing1, st.session_state.ing2,
        st.session_state.strategy, st.session_state.alpha, st.session_state.k_pref,
        st.session_state.boost_exact, st.session_state.boost_weight,
        st.session_state.require_both, st.session_state.ptr
    )
else:
    row, pool_size, total = None, 0, 0

with left:
    st.markdown("#### Candidate")
    if st.session_state.show_result and total > 0:
        st.progress(min(1.0, ((st.session_state.ptr % total) + 1) / float(total)))
        st.caption(f"Candidate # {(st.session_state.ptr % total) + 1} of {total}  •  Pool size: {pool_size}")
    else:
        st.progress(0.0)
        st.caption("No candidate yet — enter ingredients and click **Find**.")

with right:
    cont = st.container()
    with cont:
        if not st.session_state.show_result:
            st.subheader("—")
            st.write("Start by entering an ingredient and pressing **Find**.")
        elif row is None:
            st.subheader("—")
            msg = "No recipe matched your settings."
            if st.session_state.require_both and (st.session_state.ing1.strip() and st.session_state.ing2.strip()):
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
            st.caption(f"Category: {row.get('Category','?')}  •  Loves: {row.get('Loves','-')}  •  Source file: {row.get('Source','-')}")
