import streamlit as st
import numpy as np
import pandas as pd
import re
from pathlib import Path

st.set_page_config(page_title="Indonesian Recipe Gacha", page_icon="🍳", layout="wide")
st.markdown("## 🤖🍽️ Hybrid Deep Recipe Finder")

# ========= DATA & MODELS (from Drive folder) =========
import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/186OfK2ekqedpEEoNhJM2crhDvWE2uxPp?usp=sharing"
DATA_DIR = Path("recipes_data")

@st.cache_data(show_spinner=True)
def fetch_and_merge(folder_url: str) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    # download all files in the Drive folder (must be public)
    gdown.download_folder(folder_url, output=str(DATA_DIR), quiet=True, use_cookies=False)

    csv_files = sorted([p for p in DATA_DIR.glob("*.csv")])
    if not csv_files:
        raise RuntimeError("No CSV files found in 'recipes_data'. Make sure the Drive folder is public and has CSVs.")

    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).lower()).strip()

    MAIN_TITLE_KWS = {
        "nasi","mi","mie","bihun","kwetiau","kwetiaw","spaghetti","pasta",
        "soto","sop","sup","rawon","rendang","gudeg","tongseng","gulai",
        "kari","kare","opor","sate","bakso","pecel","lontong","ketoprak",
        "tahu campur","gado gado","pempek","ikan bakar","ayam bakar","ikan kuah",
        "woku","rica","balado","semur","tumis sayur","sayur asem","sayur lodeh"
    }
    SIDE_TITLE_KWS = {
        "orek","oseng","tumis","balado","bacem","semur","terik","acar",
        "sambal","telur dadar","orak arik","fuyunghai","perkedel","pepese","pepes",
        "tahu goreng","tempe goreng","telur ceplok","telur balado"
    }
    SNACK_TITLE_KWS = {
        "martabak","risol","risoles","lumpia","pastel","donat","bolu","brownies",
        "cookies","kukis","pukis","kue","pisang goreng","cireng","cilok",
        "sosis gulung","popcorn","bakso tahu goreng","pangsit goreng","cemilan","snack"
    }
    MAIN_ING_KWS = {
        "nasi","beras","mie","bihun","spaghetti","kwetiau","kentang","ubi",
        "santan","kaldu","beras ketan","lontong","ketupat"
    }
    PROTEIN_KWS = {"ayam","sapi","kambing","ikan","udang","telur","daging"}
    SIDE_ING_KWS = {"tahu","tempe","telur","sambal","sayur"}
    SNACK_ING_KWS = {"cokelat","keju","tepung","gula","mentega","margarin","tepung terigu"}

    def classify_row(title: str, ingredients: str) -> str:
        from collections import Counter
        t = normalize(title)
        ing = normalize(ingredients)
        score = Counter({"main dish":0, "side dish":0, "snack":0})
        # Title cues
        if any(kw in t for kw in SNACK_TITLE_KWS): score["snack"] += 3
        if any(kw in t for kw in SIDE_TITLE_KWS):  score["side dish"] += 2
        if any(kw in t for kw in MAIN_TITLE_KWS):  score["main dish"] += 2
        # Ingredient cues
        tokens = set(re.split(r"[^a-z0-9]+", ing))
        if tokens & MAIN_ING_KWS: score["main dish"] += 2
        if tokens & PROTEIN_KWS:  score["main dish"] += 1
        if tokens & SIDE_ING_KWS: score["side dish"] += 1
        if tokens & SNACK_ING_KWS and not (tokens & MAIN_ING_KWS): score["snack"] += 1
        # Heuristics
        if "goreng" in t and (("tahu" in t) or ("tempe" in t) or ({"tahu","tempe"} & tokens)):
            if not (tokens & MAIN_ING_KWS): score["side dish"] += 1
        if ("soto" in t or "sup" in t or "sop" in t) and "bakso tahu goreng" not in t:
            score["main dish"] += 2
        if any(k in t for k in ["kue","donat","bolu","brownies","cookies"]) and not (tokens & MAIN_ING_KWS):
            score["snack"] += 2
        # Decide + tie-breakers
        best = max(score.values())
        winners = [k for k,v in score.items() if v==best]
        if len(winners)==1: return winners[0]
        if "goreng" in t and (("tahu" in t) or ("tempe" in t)): return "side dish"
        if tokens & MAIN_ING_KWS: return "main dish"
        for c in ["main dish","side dish","snack"]:
            if c in winners: return c

    frames = []
    for p in csv_files:
        df = pd.read_csv(p)
        for col in ["Title","Ingredients","Steps","Loves","URL"]:
            if col not in df.columns: df[col] = ""
        df["Source"] = p.name
        df["Category"] = [classify_row(t, i) for t, i in zip(df["Title"], df["Ingredients"])]
        frames.append(df[["Title","Ingredients","Steps","Loves","URL","Source","Category"]])

    merged_df = pd.concat(frames, ignore_index=True)
    return merged_df

merged = fetch_and_merge(FOLDER_URL)

def build_cat_to_idx(df: pd.DataFrame):
    out = {}
    for cat in ["main dish","side dish","snack"]:
        out[cat] = np.where(df["Category"] == cat)[0]
    return out

cat_to_idx = build_cat_to_idx(merged)

# Optional: TF-IDF + BERT (cache)
@st.cache_resource(show_spinner=True)
def build_tfidf(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = (df["Title"].fillna("") + " " + df["Ingredients"].fillna("")).str.lower()
    vec = TfidfVectorizer(min_df=2, max_features=50000, ngram_range=(1,2))
    X = vec.fit_transform(text)
    return vec, X

try:
    vec, X = build_tfidf(merged)
except Exception as e:
    vec, X = None, None
    st.warning(f"TF-IDF not built: {e}")

@st.cache_resource(show_spinner=True)
def build_bert(df: pd.DataFrame):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text = (df["Title"].fillna("") + " " + df["Ingredients"].fillna("")).tolist()
    emb = model.encode(text, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return model, np.asarray(emb, dtype=np.float32)

try:
    model, embeddings = build_bert(merged)
except Exception as e:
    model, embeddings = None, None
    st.warning(f"BERT embeddings not built: {e}")

# ========= UI =========
st.sidebar.header("Search")
cat = st.sidebar.selectbox("Dish type", ["main dish","side dish","snack"])
a = st.sidebar.text_input("Main ingredient 1", "")
b = st.sidebar.text_input("Main ingredient 2", "")
boost = st.sidebar.checkbox("Boost exact word matches", value=False)
boost_w = st.sidebar.slider("Exact-match weight", 0.0, 1.5, 0.25, 0.05)
require_both = st.sidebar.checkbox("Require both ingredients (AND)", value=False)

# Minimal demo “search” (replace with your full ranker later)
def simple_filter(df: pd.DataFrame, cat: str, a: str, b: str, require_both: bool):
    idxs = cat_to_idx[cat]
    sub = df.iloc[idxs].copy()
    terms = [t for t in [a.strip(), b.strip()] if t]
    if not terms:
        return sub.head(30)
    if require_both and len(terms) == 2:
        mask = sub["Ingredients"].str.contains(terms[0], case=False, na=False) & \
               sub["Ingredients"].str.contains(terms[1], case=False, na=False)
        return sub[mask].head(30)
    # otherwise OR filter
    m = False
    for t in terms:
        m = m | sub["Ingredients"].str.contains(t, case=False, na=False) | \
                sub["Title"].str.contains(t, case=False, na=False)
    return sub[m].head(30)

results = simple_filter(merged, cat, a, b, require_both)
st.write(f"Found {len(results)} results (showing up to 30):")
st.dataframe(results[["Title","Category","Loves","URL","Source"]])
