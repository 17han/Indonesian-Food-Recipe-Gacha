# app.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# use your data / ranker
from core import merged, rank_hybrid_filtered, candidate_total

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Indonesian Recipe Gacha", page_icon="🍳", layout="wide")
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")
st.caption("Semantic (BERT) + Keyword (TF-IDF) with Auto / Hybrid / Two-stage, "
           "Exact-match boost, and optional AND filter for two ingredients.")

# in app.py, after the header:
counts = merged["Category"].str.lower().value_counts().to_dict()
st.caption(f"Dataset: main dish {counts.get('main dish',0)}, "
           f"side dish {counts.get('side dish',0)}, snack {counts.get('snack',0)}")

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _split_to_items(text: str):
    if not isinstance(text, str):
        return []
    parts = re.split(r"[\n,;•\-]+", text)
    return [p.strip() for p in parts if p.strip()]

def _to_bullets(items):
    return "\n".join(f"- {x}" for x in items) if items else "- —"

def _to_steps(items):
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, start=1)) if items else "1. —"


st.sidebar.write(f"Loaded recipes: {len(merged)}")
st.sidebar.write(f"Columns: {list(merged.columns)}")
st.sidebar.write(merged.head())

# ---------------------------------------------------------------------
# Resettable widgets (each gets a small ↩︎ button beside it)
# ---------------------------------------------------------------------
def _ensure_default(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def _reset_button(suffix):
    return st.button("↩︎", key=f"{suffix}__reset", help="Reset to default")

def resettable_slider(label, key, *, min_value, max_value, value, step=1, help=None):
    _ensure_default(key, value)
    c1, c2 = st.columns([8, 1])
    with c1:
        v = st.slider(label, min_value=min_value, max_value=max_value,
                      value=st.session_state[key], step=step, key=key, help=help)
    with c2:
        if _reset_button(key):
            st.session_state.pop(key, None)
            st.rerun()
    return v

def resettable_radio(label, key, options, index=0, help=None, horizontal=False):
    default = options[index]
    _ensure_default(key, default)
    c1, c2 = st.columns([8, 1])
    with c1:
        v = st.radio(label, options, index=options.index(st.session_state[key]),
                     key=key, help=help, horizontal=horizontal)
    with c2:
        if _reset_button(key):
            st.session_state.pop(key, None)
            st.rerun()
    return v

def resettable_checkbox(label, key, value=False, help=None):
    _ensure_default(key, bool(value))
    c1, c2 = st.columns([8, 1])
    with c1:
        v = st.checkbox(label, value=st.session_state[key], key=key, help=help)
    with c2:
        if _reset_button(key):
            st.session_state.pop(key, None)
            st.rerun()
    return v

def resettable_text(label, key, value="", help=None, placeholder=None):
    _ensure_default(key, value)
    c1, c2 = st.columns([8, 1])
    with c1:
        v = st.text_input(label, value=st.session_state[key], key=key,
                          help=help, placeholder=placeholder)
    with c2:
        if _reset_button(key):
            st.session_state.pop(key, None)
            st.rerun()
    return v

# ---------------------------------------------------------------------
# Search/ranking hook
# ---------------------------------------------------------------------
def find_one(category, a, b, strategy, alpha, k_pref, boost, boost_w, require_both, ptr):
    """
    Return (row or None, pool_size:int, total:int) using your real ranker.
    """
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

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Search")

    category = resettable_radio(
        "Dish type", "cat", ["main dish", "side dish", "snack"], index=0
    )
    ing1 = resettable_text("Main ingredient 1", "ing1", "", placeholder="e.g., ayam / nasi / lemon")
    ing2 = resettable_text("Main ingredient 2", "ing2", "", placeholder="e.g., cabe / kecap / cokelat")

    st.markdown("### Retrieval strategy")
    strategy = resettable_radio(" ", "strategy", ["auto", "hybrid", "two-stage"], index=0, horizontal=True)

    alpha = resettable_slider(
        "BERT weight (hybrid)", "alpha", min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )
    k_pref = resettable_slider(
        "Two-stage: top-K from TF-IDF", "k_pref", min_value=50, max_value=2000, value=300, step=50
    )

    st.markdown("### Matching options")
    boost_exact = resettable_checkbox("Boost exact word matches", "boost", value=False)
    boost_weight = resettable_slider(
        "Exact-match weight", "boost_w", min_value=0.0, max_value=1.5, value=0.25, step=0.05
    )
    require_both = resettable_checkbox("Require both ingredients (AND)", "and", value=False)

    st.markdown("---")
    find_btn   = st.button("🔎 Find", use_container_width=True, key="btn_find")
    reroll_btn = st.button("🎲 Reroll", use_container_width=True, key="btn_reroll")
    reset_all  = st.button("↩️ Reset all", use_container_width=True, key="btn_reset_all")

# pointer across rerolls
if "ptr" not in st.session_state:
    st.session_state.ptr = 0

# actions
if reset_all:
    for k in ["cat", "ing1", "ing2", "strategy", "alpha", "k_pref", "boost", "boost_w", "and", "ptr"]:
        st.session_state.pop(k, None)
    st.rerun()

if find_btn:
    st.session_state.ptr = 0

if reroll_btn:
    st.session_state.ptr += 1

# ---------------------------------------------------------------------
# Run a search
# ---------------------------------------------------------------------
row, pool_size, total = find_one(
    category, ing1, ing2, strategy, alpha, k_pref,
    boost_exact, boost_weight, require_both, st.session_state.ptr
)

# ---------------------------------------------------------------------
# Right panel: results
# ---------------------------------------------------------------------
colL, colR = st.columns([1.1, 1.9])

with colL:
    st.markdown("#### Candidate")
    if total > 0:
        current = (st.session_state.ptr % total) + 1
        st.progress(min(1.0, current / float(total)))
        st.caption(f"Candidate # {current} of {total}  •  Pool size: {pool_size}")
    else:
        st.caption("No candidates in the current pool. Try different terms or relax the AND filter.")

with colR:
    with st.container():
        if row is None:
            st.subheader("—")
            st.write("No recipe matched your settings. Try relaxing filters, unchecking **AND**, "
                     "or using more common ingredients.")
        else:
            st.subheader(str(row.get("Title", "—")))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ingredients**")
                st.text(_to_bullets(_split_to_items(row.get("Ingredients", ""))))
            with c2:
                st.markdown("**Steps**")
                st.text(_to_steps(_split_to_items(row.get("Steps", ""))))
            url = str(row.get("URL", "")).strip()
            if url:
                st.markdown(f"**Source**: {url}")
            st.caption(
                f"Category: {row.get('Category','?')}  •  Loves: {row.get('Loves','-')}  •  Source file: {row.get('Source','-')}"
            )

st.divider()
st.markdown("##### Debug view (first 30 rows of the selected dish type)")
sub = merged.loc[merged["Category"].str.lower() == category.lower(), ["Title", "Category", "Loves", "URL", "Source"]].head(30)
st.dataframe(sub, use_container_width=True)
