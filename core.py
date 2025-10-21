# core.py
import re, glob, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Heuristic classifier (same as Colab) ----------
def _normalize(s: str) -> str:
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

def _classify_row(title: str, ingredients: str) -> str:
    from collections import Counter
    t = _normalize(title)
    ing = _normalize(ingredients)
    score = Counter({"main dish":0, "side dish":0, "snack":0})
    if any(kw in t for kw in SNACK_TITLE_KWS): score["snack"] += 3
    if any(kw in t for kw in SIDE_TITLE_KWS):  score["side dish"] += 2
    if any(kw in t for kw in MAIN_TITLE_KWS):  score["main dish"] += 2
    tokens = set(re.split(r"[^a-z0-9]+", ing))
    if tokens & MAIN_ING_KWS: score["main dish"] += 2
    if tokens & PROTEIN_KWS:  score["main dish"] += 1
    if tokens & SIDE_ING_KWS: score["side dish"] += 1
    if tokens & SNACK_ING_KWS and not (tokens & MAIN_ING_KWS): score["snack"] += 1
    if "goreng" in t and (("tahu" in t) or ("tempe" in t) or ({"tahu","tempe"} & tokens)):
        if not (tokens & MAIN_ING_KWS): score["side dish"] += 1
    if ("soto" in t or "sup" in t or "sop" in t) and "bakso tahu goreng" not in t:
        score["main dish"] += 2
    if any(k in t for k in ["kue","donat","bolu","brownies","cookies"]) and not (tokens & MAIN_ING_KWS):
        score["snack"] += 2
    best = max(score.values())
    winners = [k for k,v in score.items() if v==best]
    if len(winners)==1: return winners[0]
    if "goreng" in t and (("tahu" in t) or ("tempe" in t)): return "side dish"
    if tokens & MAIN_ING_KWS: return "main dish"
    for c in ["main dish","side dish","snack"]:
        if c in winners: return c

# ---------- Load & merge (auto-add Category if missing) ----------
def _load_merged():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    merged_csv = data_dir / "merged.csv"
    files = [str(merged_csv)] if merged_csv.exists() else sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        return pd.DataFrame(columns=["Title","Ingredients","Steps","Loves","URL","Source","Category"])
    frames = []
    for f in files:
        df = pd.read_csv(f)
        for col in ["Title","Ingredients","Steps","Loves","URL"]:
            if col not in df.columns: df[col] = ""
        if "Source" not in df.columns: df["Source"] = Path(f).name
        if "Category" not in df.columns:
            df["Category"] = [_classify_row(t, i) for t, i in zip(df["Title"], df["Ingredients"])]
        else:
            cat = df["Category"].fillna("").astype(str).str.strip().str.lower()
            need = (cat == "")
            if need.any():
                df.loc[need, "Category"] = [
                    _classify_row(t, i) for t, i in zip(df.loc[need, "Title"], df.loc[need, "Ingredients"])
                ]
            df["Category"] = df["Category"].str.strip().str.lower()
        frames.append(df[["Title","Ingredients","Steps","Loves","URL","Source","Category"]])
    return pd.concat(frames, ignore_index=True)

merged: pd.DataFrame = _load_merged()

cat_to_idx = {
    c: merged.index[merged["Category"].str.lower() == c].to_numpy()
    for c in ["main dish","side dish","snack"]
}

# ---------- BERT ----------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)
texts = (merged["Title"].fillna("") + " | " + merged["Ingredients"].fillna("")).tolist()
embeddings = (
    np.asarray(
        model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True),
        dtype="float32"
    ) if len(texts) else np.zeros((0,384), dtype="float32")
)

# ---------- TF-IDF ----------
tfidf_texts = merged["Ingredients"].fillna("").astype(str).str.lower().tolist()
if len(tfidf_texts):
    vec = TfidfVectorizer(token_pattern=r"[a-zA-ZÀ-ÿ0-9_\-]+")
    X = vec.fit_transform(tfidf_texts)
    idf_map = dict(zip(vec.get_feature_names_out(), vec.idf_))
    avg_idf = float(np.mean(list(idf_map.values()))) if idf_map else 0.0
else:
    from scipy.sparse import csr_matrix
    vec = TfidfVectorizer()
    X = csr_matrix((0,0))
    idf_map, avg_idf = {}, 0.0

# ---------- Helpers ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()

def _tokens(q: str):
    return re.findall(r"[a-z0-9_\-]+", _norm(q))

def _rarity_score(query: str) -> float:
    toks = _tokens(query)
    if not toks: return 0.0
    vals = [idf_map.get(t, avg_idf) for t in toks]
    return float(np.clip(np.mean(vals) - avg_idf, 0.0, 1.5))

def sims_tfidf_idx(q_vec, idxs): return cosine_similarity(q_vec, X[idxs]).ravel()
def sims_bert_idx(q_emb, idxs):  return (embeddings[idxs] @ q_emb).astype("float32")

def exact_match_bonus(query: str, idxs, weight: float):
    if weight <= 0: return np.zeros(len(idxs), dtype="float32")
    qtok = set(_tokens(query))
    if not qtok:   return np.zeros(len(idxs), dtype="float32")
    texts = (merged.loc[idxs, "Title"].fillna("").astype(str).str.lower() + " | " +
             merged.loc[idxs, "Ingredients"].fillna("").astype(str).str.lower()).tolist()
    hits = np.zeros(len(idxs), dtype="float32")
    for i, t in enumerate(texts):
        toks = set(re.findall(r"[a-z0-9_\-]+", t))
        hits[i] = float(len(qtok & toks))
    return hits * float(weight)

def _contains_all_terms(row_text: str, terms: list[str]) -> bool:
    toks = set(re.findall(r"[a-z0-9_\-]+", row_text.lower()))
    return all(t in toks for t in terms)

def _filtered_idxs_for_terms(category: str, ing1: str, ing2: str, require_both: bool):
    idxs = cat_to_idx.get((category or "").lower(), np.array([], dtype=int))
    if not require_both: return idxs
    terms = [t.strip().lower() for t in (ing1, ing2) if t and t.strip()]
    if len(terms) < 2:   return idxs
    blobs = (merged.loc[idxs, "Title"].astype(str) + " | " +
             merged.loc[idxs, "Ingredients"].astype(str)).tolist()
    keep = [i for i, txt in zip(idxs, blobs) if _contains_all_terms(txt, terms)]
    return np.array(keep, dtype=int)

# ---------- Ranker ----------
def rank_hybrid_filtered(category: str, ing1: str, ing2: str, ptr: int,
                         alpha: float = 0.7, k_prefilter: int = 300,
                         strategy: str = "auto",     # "auto" | "hybrid" | "two-stage"
                         boost_exact: bool = False,
                         boost_weight: float = 0.25,
                         require_both: bool = False):
    idxs = _filtered_idxs_for_terms(category, ing1, ing2, require_both)
    if idxs is None or len(idxs) == 0:
        return idxs, []
    terms = [t.strip() for t in [ing1, ing2] if t and t.strip()]
    if not terms:
        return idxs, [ idxs[ptr % len(idxs)] ]
    query = " ".join(terms)
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    q_vec = vec.transform([_norm(query)])
    if strategy == "auto":
        r = _rarity_score(query)
        alpha_eff = float(np.clip(0.85 - 0.2 * (r / 1.5), 0.55, 0.85))
        use_two_stage = len(idxs) > int(k_prefilter)
    else:
        alpha_eff = alpha
        use_two_stage = (strategy == "two-stage")
    if use_two_stage:
        sims_t1 = sims_tfidf_idx(q_vec, idxs)
        topk = np.argsort(-sims_t1)[:min(int(k_prefilter), len(idxs))]
        idxs_k = idxs[topk]
        sims_b = sims_bert_idx(q_emb, idxs_k)
        sims_t = sims_t1[topk]
        final_sims = alpha_eff * sims_b + (1 - alpha_eff) * sims_t
        if boost_exact: final_sims += exact_match_bonus(query, idxs_k, boost_weight)
        ranked = idxs_k[np.argsort(-final_sims)]
    else:
        sims_b = sims_bert_idx(q_emb, idxs)
        sims_t = sims_tfidf_idx(q_vec, idxs)
        final_sims = alpha_eff * sims_b + (1 - alpha_eff) * sims_t
        if boost_exact: final_sims += exact_match_bonus(query, idxs, boost_weight)
        ranked = idxs[np.argsort(-final_sims)]
    if len(ranked) == 0:
        return idxs, []
    return idxs, [ ranked[ptr % len(ranked)] ]

def candidate_total(cat: str, strat: str, k_pref: int, ing1: str = "", ing2: str = "", require_both: bool = False):
    idxs = _filtered_idxs_for_terms(cat, ing1, ing2, require_both)
    use_two = (strat == "two-stage") or (strat == "auto" and len(idxs) > int(k_pref))
    return int(min(int(k_pref), len(idxs))) if use_two else int(len(idxs))
