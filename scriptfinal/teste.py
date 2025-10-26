# ================================================================
# Projeto Matching de Hotéis - Befly - Gabriel Ferreira
# Script Final
# ================================================================

import pandas as pd
import numpy as np
import re
from Levenshtein import ratio as levenshtein_ratio
import itertools
import time

# ================================================================
# 1. Leitura dos Dados
# ================================================================
hotels_A = pd.read_csv('hotels_A.csv')
hotels_B = pd.read_csv('hotels_B.csv')
train_truth_df = pd.read_csv('train.csv')

# ================================================================
# 2. Funções Auxiliares
# ================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\sÀ-ÿ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in common_terms_to_remove_set]
    return " ".join(filtered_words)

def haversine_distance(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def apply_matching_rules(df, name_sim_thresh, addr_sim_thresh, final_geo_dist_thresh,
                         flexible_name_sim_thresh, flexible_geo_dist_thresh):
    df = df.copy()
    df['predicted_match'] = 0
    df.loc[
        (df['name_similarity'] >= name_sim_thresh) &
        (df['address_similarity'] >= addr_sim_thresh) &
        (df['geo_distance_km'] <= final_geo_dist_thresh),
        'predicted_match'
    ] = 1
    df.loc[
        (df['name_similarity'] >= flexible_name_sim_thresh) &
        (df['geo_distance_km'] <= flexible_geo_dist_thresh),
        'predicted_match'
    ] = 1
    return df

def calculate_metrics(pred_df, truth_df):
    truth_pos = truth_df[truth_df['is_match'] == 1][['id_A', 'id_B']].drop_duplicates()
    all_pairs = truth_df[['id_A', 'id_B']].drop_duplicates()
    preds = pred_df[pred_df['predicted_match'] == 1][['id_A', 'id_B']].drop_duplicates()

    pred_in_truth = pd.merge(preds, all_pairs, on=['id_A', 'id_B'], how='inner')
    tp = len(pd.merge(pred_in_truth, truth_pos, on=['id_A', 'id_B'], how='inner'))
    fp = len(pred_in_truth) - tp
    fn = len(truth_pos) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }

# ================================================================
# 3. Pré-processamento
# ================================================================
common_terms_to_remove = [
    'da','dos','das','hotel','inn','pousada','grand','residency','palace',
    'suites','boutique','spa','saint','avenue','av','street','r','rua',
    'praça','pça','avenida','ave','ii','i','2','st'
]
common_terms_to_remove_set = set(common_terms_to_remove)

for df in [hotels_A, hotels_B]:
    df['cleaned_hotel_name'] = df['hotel_name'].apply(clean_text)
    df['cleaned_address'] = df['address'].apply(clean_text)
    df['cleaned_city'] = df['city'].apply(clean_text)
    df['cleaned_country'] = df['country'].apply(clean_text)

# ================================================================
# 4. Blocking por Cidade e País
# ================================================================
hotels_A['block_key'] = hotels_A['cleaned_city'] + '_' + hotels_A['cleaned_country']
hotels_B['block_key'] = hotels_B['cleaned_city'] + '_' + hotels_B['cleaned_country']
blocks_B = hotels_B.groupby('block_key')

potential_matches = []
for key in set(hotels_A['block_key']).intersection(hotels_B['block_key']):
    blockA = hotels_A[hotels_A['block_key'] == key]
    blockB = blocks_B.get_group(key)
    for _, a in blockA.iterrows():
        for _, b in blockB.iterrows():
            potential_matches.append({
                'id_A': a['id_A'], 'id_B': b['id_B'],
                'cleaned_hotel_name_A': a['cleaned_hotel_name'],
                'cleaned_hotel_name_B': b['cleaned_hotel_name'],
                'cleaned_address_A': a['cleaned_address'],
                'cleaned_address_B': b['cleaned_address'],
                'latitude_A': a['latitude'], 'longitude_A': a['longitude'],
                'latitude_B': b['latitude'], 'longitude_B': b['longitude']
            })

potential_df = pd.DataFrame(potential_matches)

# ================================================================
# 5. Similaridade + Distância Geográfica
# ================================================================
potential_df['geo_distance_km'] = potential_df.apply(
    lambda r: haversine_distance(r['latitude_A'], r['longitude_A'], r['latitude_B'], r['longitude_B']), axis=1
)

geo_df = potential_df[potential_df['geo_distance_km'] <= 1.0].copy()

geo_df['name_similarity'] = geo_df.apply(
    lambda r: levenshtein_ratio(r['cleaned_hotel_name_A'], r['cleaned_hotel_name_B']), axis=1
)
geo_df['address_similarity'] = geo_df.apply(
    lambda r: levenshtein_ratio(r['cleaned_address_A'], r['cleaned_address_B']), axis=1
)

# ================================================================
# 6. Busca dos Melhores Parâmetros
# ================================================================
name_similarity_thresholds = [0.75, 0.80]
address_similarity_thresholds = [0.50, 0.70, 0.90]
final_max_distance_kms = [0.5, 0.6, 0.7, 0.8, 0.9]
flexible_name_sim_threshold = 0.95
flexible_geo_dist_threshold = 0.5

combinations = list(itertools.product(
    name_similarity_thresholds,
    address_similarity_thresholds,
    final_max_distance_kms
))
total_combinations = len(combinations)

print(f"\n Iniciando busca dos melhores parâmetros ({total_combinations} combinações)...\n")

best_f1 = 0
best_params = None

for idx, (name_t, addr_t, dist_t) in enumerate(combinations, start=1):
    print(f"→ Testando combinação {idx}/{total_combinations}: "
          f"name={name_t}, address={addr_t}, dist={dist_t}...", end="\r")
    
    temp_df = apply_matching_rules(
        geo_df,
        name_t,
        addr_t,
        dist_t,
        flexible_name_sim_threshold,
        flexible_geo_dist_threshold
    )

    metrics = calculate_metrics(temp_df, train_truth_df)

    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_params = (name_t, addr_t, dist_t, metrics)
    
    time.sleep(0.1)  # apenas para dar tempo do print atualizar no terminal

print("\n\n Busca concluída!")
print(f"Melhores parâmetros encontrados:")
print(f"  Métricas: {best_params[3]}")

# ================================================================
# 7. Aplicação Final com os Melhores Parâmetros
# ================================================================
final_df = apply_matching_rules(
    geo_df,
    best_params[0],
    best_params[1],
    best_params[2],
    flexible_name_sim_threshold,
    flexible_geo_dist_threshold
)

final_metrics = calculate_metrics(final_df, train_truth_df)

print("\nResultados finais:")
for k, v in final_metrics.items():
    print(f"{k}: {v}")

# ================================================================
# 8. Exportar Output Final
# ================================================================
output_for_submission = final_df[['id_A', 'id_B', 'predicted_match']].copy()
output_for_submission.to_csv('output.csv', index=False)

print("\n Arquivo 'output.csv' salvo.")