# Hotel Matching Project: Data Integration and Enrichment via API

## Analysis Preview

**Full and Detailed Analysis**:  
Access the full project page [here] (https://ferreiragabrielw.github.io/portfolio-gabriel/projetos/DataScience/3MatchingHoteis/MatchingHoteis.html)(interactive HTML with code, visuals, and video demo).

## About the Project

This Data Science project focuses on entity resolution (record linkage) for hotel datasets, addressing the challenge of merging two sources (`hotels_A.csv` and `hotels_B.csv`) without shared keys. The datasets contain 436 and 425 hotel listings, respectively, with attributes like names, addresses, cities, countries, geolocations (latitude/longitude), ratings, and review counts. A labeled training set (`train.csv`) with 332 pairs enables validation.

The main objectives are:

- Develop a scalable matching algorithm to identify duplicate hotels (e.g., same property listed differently across platforms like Booking and Expedia), using textual similarities (Levenshtein distance) and geographic proximity (Haversine formula).
- Enrich matched records via a simulated RESTful API (`mock_api.py`), adding attributes like category stars (3-5), review scores (7.0-9.7), and amenities (e.g., "wifi, pool, gym") to simulate real-world integrations (e.g., Google Places API).

This pipeline simulates hospitality data unification, enabling duplicate detection, enhanced search, and personalized recommendations—reducing manual efforts by 90% and improving platform accuracy.

## Technologies and Process

**Tools**: Python (Pandas, NumPy, Levenshtein for similarity, FastAPI/Uvicorn for mock API, Matplotlib/Seaborn for visuals), Quarto (for rendered reports), Jupyter Notebook (exploratory analysis).

**End-to-End Analysis Pipeline (E2E)**:

1. **Data Loading & EDA**: Inspect schemas, quality checks (nulls/duplicates), and frequency analysis to identify noise (e.g., generics like "hotel", "rua").
2. **Preprocessing**: Normalize text (lowercase, punctuation removal, generic filtering) to create cleaned columns for robust similarity computation.
3. **Candidate Generation (Blocking)**: Hierarchical pruning—textual (city-country keys) reduces pairs by 90%; geographic (Haversine ≤1km) further cuts 70-80%, avoiding O(n²) (185k+ pairs).
4. **Similarity Computation**: Levenshtein ratio on cleaned names/addresses for geo-blocked candidates (~5k pairs).
5. **Threshold Optimization**: Grid search on 80/20 train/validation split maximizes F1-score (precision/recall harmonic); flexible rules handle edge cases.
6. **Entity Matching**: Apply optimized rules (strict: name≥0.75, address≥0.50, geo≤0.9km; flexible: name≥0.95, geo≤0.5km) for binary predictions (`predicted_match` 0/1).
7. **API Enrichment**: POST unique matches to mock `/enrich` endpoint; merge stars, scores, amenities into final CSV.
8. **Evaluation**: Perfect F1=1.0 (TP=full, FP/FN=0) on unseen validation; runtime <10s.

## Key Insights (Business Perspective)

The pipeline achieves **perfect accuracy (F1=1.0)** on validation, with ~150-200 matches from 18k candidates—97% reduction via blocking, enabling real-time use on larger datasets.

- **Efficiency**: Textual blocking prunes 90% (city-country focus captures location-bound duplicates); geographic adds precision, filtering distant pairs (e.g., same city but >1km unlikely matches).
- **Accuracy Drivers**: Levenshtein tolerates noise (e.g., "Paulista" vs. "Pça. Paulista"); flexible rules catch 10-15% geo-dominant cases (e.g., exact names, minor address typos). No overfitting (val F1 matches train).
- **Enrichment Value**: Adds actionable data (e.g., 70% matches have "wifi/pool"); simulates real APIs for features like amenity-based filtering ("spa hotels near me").

## Repository Content

- **data/**: Raw datasets (`hotels_A.csv`, `hotels_B.csv`, `train.csv`).
- **quarto/**: Rendered report (`MatchingHoteis.qmd`, `MatchingHoteis.html`), assets (`MatchingHoteis_files/`), and demo video (`pipeline_execution.mp4` showing API enrichment).
- **scripts/**: Production pipeline (`hotel_matching_pipeline_gabrielferreira.py` for matching + enrichment + CSV export) and mock API (`mock_api.py`).
- **notebook/**: Exploratory Jupyter Notebook (`HotelsMatchingNotebook_gf.ipynb`) with EDA, prototyping, and visuals.
- **requirements.txt**: Python dependencies (e.g., pandas, numpy, python-Levenshtein, fastapi, uvicorn).
- **README.md**: This document.
- **LICENSE**: Project license (MIT License).

## How to View the Full Analysis

- **Online (HTML)**: Open [here] (https://ferreiragabrielw.github.io/portfolio-gabriel/projetos/DataScience/3MatchingHoteis/MatchingHoteis.html) in your browser for interactive code, plots, and embedded video.
- **Jupyter Notebook**: View [HotelsMatchingNotebook_gf.ipynb](notebook/HotelsMatchingNotebook_gf.ipynb) on GitHub or run locally (`jupyter notebook`).
- **Locally (Quarto)**:
  1. Install Quarto and Python (with libraries from `requirements.txt`: `pip install -r requirements.txt`).
  2. Download `MatchingHoteis.qmd` from `quarto/`.
  3. Open in VS Code (with Quarto extension) or RStudio; render via `quarto render MatchingHoteis.qmd`.
- **Run Pipeline**:
  1. Install deps: `pip install -r requirements.txt`.
  2. Start mock API: `uvicorn scripts.mock_api:app --host 0.0.0.0 --port 8000` (new terminal).
  3. Execute: `python scripts/hotel_matching_pipeline_gabrielferreira.py`—generates `output.csv` and `output_final_enriquecido.csv`.

## License

This project is licensed under the MIT License.