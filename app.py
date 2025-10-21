# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
from pathlib import Path

# <-- uses your models/data/ranker built in core.py
from core import merged, rank_hybrid_filtered, candidate_total

st.set_page_config(page_title="Indonesian Recipe Gacha", page_icon="🍳", layout="wide")
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")

# ---------- small format helpers ----------
def _split_to_items(text: str):
    if not isinstance(text, str):
        return []
    parts = re.split(r'[\n,;•\-]+', text)
    return [p.strip() for p in parts if p.strip()]

def _to_bullets(items):
    return "\n".join(f"- {x}" for x in items) if items else "- —"

def _to_steps(items):
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, start=1)) if items else "1. —"

# ---------- the function you were missing ----------
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
        return None, pool_size, total
    return merged.iloc[picked[0]], pool_size, max(1, total)

# ---------- UI ----------
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

    find = st.button("🔎 Find", use_container_width=True)
    reroll = st.button("🎲 Reroll", use_container_width=True)
    reset = st.button("↩️ Reset", use_container_width=True)

# keep position across rerolls
if "ptr" not in st.session_state: st.session_state.ptr = 0

# actions
if reset:
    st.session_state.ptr = 0

if find:
    st.session_state.ptr = 0

if reroll:
    st.session_state.ptr += 1

row, pool_size, total = find_one(
    category, ing1, ing2, strategy, alpha, k_pref,
    boost_exact, boost_weight, require_both, st.session_state.ptr
)

# ---------- Right side: results ----------
colL, colR = st.columns([1.1, 1.9])

with colL:
    st.markdown("#### Candidate")
    if total > 0:
        # 1-based for display
        st.progress(min(1.0, ( (st.session_state.ptr % total) + 1) / float(total)))
        st.caption(f"Candidate # {(st.session_state.ptr % total) + 1} of {total}  •  Pool size: {pool_size}")

with colR:
    with st.container():
        if row is None:
            st.subheader("—")
            st.write("No recipe matched your settings. Try relaxing filters or turning off **AND**.")
        else:
            st.subheader(str(row.get("Title","—")))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ingredients**")
                st.text(_to_bullets(_split_to_items(row.get("Ingredients",""))))
            with c2:
                st.markdown("**Steps**")
                st.text(_to_steps(_split_to_items(row.get("Steps",""))))
            if str(row.get("URL","")).strip():
                st.markdown(f"**Source**: {row['URL']}")
            st.caption(f"Category: {row.get('Category','?')}  •  Loves: {row.get('Loves','-')}  •  Source file: {row.get('Source','-')}")

st.divider()
st.markdown("##### Debug view (top 30 table for your current dish type)")
sub = merged.loc[merged["Category"].str.lower() == category.lower(), ["Title","Category","Loves","URL","Source"]].head(30)
st.dataframe(sub, use_container_width=True)
