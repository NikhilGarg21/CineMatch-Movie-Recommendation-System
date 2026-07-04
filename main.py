import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_BASE_URL = "https://imdb.iamidiotareyoutoo.com"

app = FastAPI(title="CineMatch API", version="5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

class MovieCard(BaseModel):
    imdb_id: str
    title: str
    poster_url: Optional[str] = None
    year: Optional[str] = None
    actors: Optional[str] = None

class MovieDetails(BaseModel):
    imdb_id: str
    title: str
    plot: Optional[str] = None
    year: Optional[str] = None
    poster_url: Optional[str] = None
    actors: List[str] = []
    rating: Optional[str] = None
    votes: Optional[str] = None

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    meta: Optional[MovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[TFIDFRecItem]

def _norm_title(t: str) -> str:
    return str(t).strip().lower()

async def fetch_public_data(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{API_BASE_URL}{endpoint}", params=params)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network Error: {type(e).__name__}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Engine Error {r.status_code}")
    return r.json()

def find_column_ignore_case(dataframe: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    cols_lower = {c.lower().strip(): c for c in dataframe.columns}
    for name in possible_names:
        clean_name = name.lower().strip()
        if clean_name in cols_lower:
            return cols_lower[clean_name]
    return None

async def extract_cards_from_search(query: str, limit: int = 12) -> List[MovieCard]:
    global df
    out: List[MovieCard] = []
    seen_titles = set()
    seen_ids = set()

    clean_q = query.strip()
    if not clean_q:
        return []

    if df is not None and not df.empty:
        title_col = find_column_ignore_case(df, ['title', 'movie_title', 'name'])
        if title_col:
            try:
                local_matches = df[df[title_col].str.contains(clean_q, case=False, na=False)]
                for idx, row in local_matches.iterrows():
                    t_name = str(row[title_col]).strip()
                    t_clean = t_name.lower()
                    
                    poster = None
                    p_col = find_column_ignore_case(df, ['poster_url', 'poster', 'image_url', 'img', 'images'])
                    if p_col and pd.notna(row[p_col]):
                        poster = str(row[p_col]).strip()
                    
                    if poster and "http" in poster and t_clean not in seen_titles:
                        seen_titles.add(t_clean)
                        
                        y_col = find_column_ignore_case(df, ['year', 'release_year', 'date'])
                        year_val = str(row[y_col]) if y_col and pd.notna(row[y_col]) else ""

                        id_col = find_column_ignore_case(df, ['imdb_id', 'id', 'movie_id'])
                        id_val = str(row[id_col]) if id_col and pd.notna(row[id_col]) else str(idx)
                        seen_ids.add(id_val)

                        out.append(MovieCard(imdb_id=id_val, title=t_name, poster_url=poster, year=year_val))
                        
                    if len(out) >= limit:
                        return out
            except Exception:
                pass

    search_queries = [clean_q, f"The {clean_q}", f"{clean_q} Begins", f"{clean_q} 2"]
    for q_var in search_queries:
        if len(out) >= limit:
            break
        try:
            data = await fetch_public_data("/search", {"q": q_var})
            raw_items = data.get("description", [])
            for item in raw_items:
                imdb_id = item.get("#IMDB_ID")
                title = item.get("#TITLE", "").strip()
                title_clean = title.lower()
                poster_url = item.get("#IMG_POSTER")
                
                if poster_url and "http" in poster_url and title_clean not in seen_titles and imdb_id not in seen_ids:
                    seen_titles.add(title_clean)
                    seen_ids.add(imdb_id)
                    out.append(MovieCard(imdb_id=imdb_id, title=title, poster_url=poster_url, year=str(item.get("#YEAR", ""))))
                if len(out) >= limit:
                    break
        except Exception:
            continue

    return out[:limit]

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("Matrix indices asset mismatch.")

def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX, df
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="Matrix missing.")
    
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
        
    if df is not None and not df.empty:
        title_col = find_column_ignore_case(df, ['title', 'movie_title', 'name'])
        if title_col:
            try:
                matched_rows = df[df[title_col].str.lower().str.contains(key, case=False, na=False)]
                if not matched_rows.empty:
                    first_match_title = _norm_title(matched_rows.iloc[0][title_col])
                    if first_match_title in TITLE_TO_IDX:
                        return int(TITLE_TO_IDX[first_match_title])
            except Exception:
                pass

    raise HTTPException(status_code=404, detail=f"Missing context index for item: '{title}'")

def tfidf_recommend_titles(query_title: str, top_n: int = 12) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    title_col = find_column_ignore_case(df, ['title', 'movie_title', 'name']) or 'title'
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)][title_col])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out

async def look_up_card_by_title(title: str) -> Optional[MovieCard]:
    cards = await extract_cards_from_search(title, limit=1)
    return cards[0] if cards else None

@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

@app.get("/home", response_model=List[MovieCard])
async def home(limit: int = 12):
    out: List[MovieCard] = []
    seen_titles = set()
    
    premium_movie_list = [
        "Inception", "The Dark Knight", "Interstellar", "Avatar", 
        "The Avengers", "Gladiator", "Titanic", "The Matrix", 
        "Joker", "Spider-Man", "Iron Man", "The Prestige"
    ]
    
    for title in premium_movie_list:
        if len(out) >= limit:
            break
        try:
            cards = await extract_cards_from_search(title, limit=1)
            if cards:
                target_card = cards[0]
                t_clean = target_card.title.lower().strip()
                if t_clean not in seen_titles and target_card.poster_url:
                    seen_titles.add(t_clean)
                    out.append(target_card)
        except Exception:
            continue
                
    return out[:limit]

@app.get("/search", response_model=List[MovieCard])
async def search_engine(query: str = Query(..., min_length=1), limit: int = Query(12, ge=1, le=50)):
    return await extract_cards_from_search(query, limit=limit)

@app.get("/movie/bundle", response_model=SearchBundleResponse)
async def search_bundle(query: str = Query(..., min_length=1), tfidf_top_n: int = Query(12, ge=1, le=30)):
    global df
    cards = await extract_cards_from_search(query, limit=1)
    if not cards:
        raise HTTPException(status_code=404, detail=f"No profiles found for: {query}")

    target = cards[0]
    actor_list = [a.strip() for a in target.actors.split(",")] if target.actors else []
    
    plot_summary = f"Profile record for {target.title} released in {target.year}."
    rating = "N/A"
    votes = "N/A"

    if df is not None and not df.empty:
        title_col = find_column_ignore_case(df, ['title', 'movie_title', 'name'])
        if title_col:
            matched_rows = df[df[title_col].str.contains(query, case=False, na=False)]
            if not matched_rows.empty:
                row = matched_rows.iloc[0]
                
                plot_col = find_column_ignore_case(df, ['plot', 'overview', 'description', 'summary'])
                if plot_col and pd.notna(row[plot_col]):
                    plot_summary = str(row[plot_col]).strip()
                    
                rating_col = find_column_ignore_case(df, ['rating', 'score', 'vote_average', 'avg_vote', 'imdb_rating'])
                if rating_col and pd.notna(row[rating_col]):
                    rating = str(row[rating_col]).strip()
                    
                votes_col = find_column_ignore_case(df, ['votes', 'vote_count', 'total_votes', 'reviews'])
                if votes_col and pd.notna(row[votes_col]):
                    votes = str(row[votes_col]).strip()

    details = MovieDetails(
        imdb_id=target.imdb_id, title=target.title, plot=plot_summary,
        year=target.year, poster_url=target.poster_url, actors=actor_list,
        rating=rating, votes=votes
    )

    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    for title_variant in [details.title, query, query.split(":")[0]]:
        try:
            recs = tfidf_recommend_titles(title_variant, top_n=tfidf_top_n)
            if recs: break
        except Exception: continue

    for title, score in recs:
        meta_card = await look_up_card_by_title(title)
        if not meta_card:
            meta_card = MovieCard(imdb_id="placeholder", title=title, poster_url="https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=500")
        tfidf_items.append(TFIDFRecItem(title=title, score=score, meta=meta_card))

    return SearchBundleResponse(query=query, movie_details=details, tfidf_recommendations=tfidf_items[:tfidf_top_n])