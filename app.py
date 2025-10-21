# app.py
import streamlit as st, re
from core import merged, rank_hybrid_filtered, candidate_total

st.set_page_config(page_title="Indonesian Recipe Gacha", page_icon="🍳", layout="wide")
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")

# ----------------------------------------------------
# Quick data check
# ----------------------------------------------------
st.caption(f"Loaded **{len(merged):,}** recipes")
if len(merged) == 0:
    st.error("No recipe data found. Add CSVs in /data or make your Drive folder public.")
    st.stop()

# ----------------------------------------------------
# Small format helpers
# ----------------------------------------------------
def _split_to_items(text):
    if not isinstance(text, str): return []
    parts = re.split(r'[\n,;•\-]+', text)
    return [p.strip() for p in parts if p.strip()]

def _to_bullets(items): return "\n".join(f"- {x}" for x in items) if items else "- —"
def _to_steps(items):   return "\n".join(f"{i}. {x}" for i,x in enumerate(items,1)) if items else "1. —"

# ----------------------------------------------------
# Resettable slider helper
# ----------------------------------------------------
def resettable_slider(label, key, *, min_value, max_value, value, step):
    c1, c2 = st.columns([6,1])
    with c1:
        v = st.slider(label, min_value=min_value, max_value=max_value,
                      value=value, step=step, key=key)
    with c2:
        if st.button("↺", key=f"{key}__reset"):
            st.session_state[key] = value
    return st.session_state[key]

# ----------------------------------------------------
# Finder function
# ----------------------------------------------------
def find_one(category, a, b, strategy, alpha, k_pref, boost, boost_w, require_both, ptr):
    idxs, picked = rank_hybrid_filtered(
        category, a, b, ptr=ptr,
        alpha=float(alpha), k_prefilter=int(k_pref),
        strategy=strategy,
        boost_exact=bool(boost), boost_weight=float(boost_w),
        require_both=bool(require_both),
    )
    pool_size = 0 if idxs is None else len(idxs)
    total = candidate_total(category, strategy, int(k_pref), a, b, require_both)
    if not picked: return None, pool_size, total
    return merged.iloc[picked[0]], pool_size, max(1, total)

# ----------------------------------------------------
# Sidebar UI
# ----------------------------------------------------
with st.sidebar:
    st.header("Search")
    category = st.selectbox("Dish type", ["main dish","side dish","snack"])
    ing1 = st.text_input("Main ingredient 1", "")
    ing2 = st.text_input("Main ingredient 2", "")

    st.markdown("### Retrieval strategy")
    strategy = st.radio(" ", ["auto","hybrid","two-stage"], index=0, label_visibility="collapsed")

    alpha = resettable_slider("BERT weight (hybrid)", "alpha", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    k_pref = resettable_slider("Two-stage: top-K from TF-IDF", "k_pref", min_value=50, max_value=2000, value=300, step=50)

    st.markdown("### Matching options")
    boost_exact = st.checkbox("Boost exact word matches", value=False, key="boost_exact")
    boost_weight = resettable_slider("Exact-match weight", "boost_weight", min_value=0.0, max_value=1.5, value=0.25, step=0.05)
    require_both = st.checkbox("Require both ingredients (AND)", value=False, key="require_both")

    find = st.button("🔎 Find", use_container_width=True)
    reroll = st.button("🎲 Reroll", use_container_width=True)
    reset = st.button("↩️ Reset", use_container_width=True)

# ----------------------------------------------------
# Button logic
# ----------------------------------------------------
if "ptr" not in st.session_state: st.session_state.ptr = 0
if reset or find: st.session_state.ptr = 0
if reroll: st.session_state.ptr += 1

row, pool_size, total = find_one(
    category, ing1, ing2, strategy, alpha, k_pref,
    boost_exact, boost_weight, require_both, st.session_state.ptr
)

# ----------------------------------------------------
# Results
# ----------------------------------------------------
colL, colR = st.columns([1.1, 1.9])

with colL:
    st.markdown("#### Candidate")
    if total and total > 0:
        st.progress(min(1.0, (((st.session_state.ptr % total) + 1) / float(total))))
        st.caption(f"Candidate # {(st.session_state.ptr % total) + 1} of {total}  •  Pool size: {pool_size}")

with colR:
    if row is None:
        st.subheader("—")
        st.write("No recipe matched your settings. Try other ingredients or disable **AND**.")
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
        st.caption(f"Category: {row.get('Category','?')}  •  Loves: {row.get('Loves','-')}  •  Source: {row.get('Source','-')}")

st.divider()
st.markdown("##### Debug view (top 30 table for your current dish type)")
sub = merged.loc[merged["Category"].str.lower() == category.lower(), ["Title","Category","Loves","URL","Source"]].head(30)
st.dataframe(sub, use_container_width=True)
