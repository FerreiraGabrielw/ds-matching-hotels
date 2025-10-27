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
import requests


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

# ================================================================
# 9. Enriquecimento dos Dados via API 
# ================================================================
print("\n" + "=" * 50)
print("INICIANDO ETAPA DE ENRIQUECIMENTO DOS DADOS VIA API (REVISADO)")
print("=" * 50)

API_URL = "http://localhost:8000/enrich"
API_HEALTH_URL = "http://localhost:8000/healthz"
API_URL_AVAILABLE = False

print(f"Verificando API mock em {API_HEALTH_URL}...")
try:
    health_check = requests.get(API_HEALTH_URL, timeout=5)
    health_check.raise_for_status()
    print("API mock está ok.")
    API_URL_AVAILABLE = True
except requests.exceptions.RequestException as e:
    print(f"ERRO: Não foi possível conectar à API mock. Certifique-se de que está rodando. Detalhes: {e}")
    print("Prosseguindo sem enriquecimento de dados da API. Os campos enriquecidos ficarão vazios.")

# 1. Filtrar apenas os pares que tiveram match
matched_pairs_df = final_df[final_df["predicted_match"] == 1].copy()

# 2. Juntar com os dados originais de hotels_A e hotels_B para obter as infos de nome, cidade, país
matched_pairs_with_details = matched_pairs_df.merge(
    hotels_A[['id_A', 'hotel_name', 'city', 'country']],
    on='id_A',
    how='left'
).rename(columns={
    'hotel_name': 'hotel_name_A',
    'city': 'city_A',
    'country': 'country_A'
})

# Para o lado B
matched_pairs_with_details = matched_pairs_with_details.merge(
    hotels_B[['id_B', 'hotel_name', 'city', 'country']],
    on='id_B',
    how='left'
).rename(columns={
    'hotel_name': 'hotel_name_B',
    'city': 'city_B',
    'country': 'country_B'
})


# 3. Coletar os perfis únicos de hotéis (nome, cidade, país) DENTRE OS HOTÉIS QUE TIVERAM MATCH
unique_matched_hotels_A_profiles = matched_pairs_with_details[['hotel_name_A', 'city_A', 'country_A']].drop_duplicates().rename(columns={
    'hotel_name_A': 'hotel_name', 'city_A': 'city', 'country_A': 'country'
})

# Perfis do lado B que tiveram match
unique_matched_hotels_B_profiles = matched_pairs_with_details[['hotel_name_B', 'city_B', 'country_B']].drop_duplicates().rename(columns={
    'hotel_name_B': 'hotel_name', 'city_B': 'city', 'country_B': 'country'
})

# Combinar todos os perfis únicos de hotéis envolvidos em qualquer match
all_unique_matched_hotels_to_enrich = pd.concat([unique_matched_hotels_A_profiles, unique_matched_hotels_B_profiles]).drop_duplicates().reset_index(drop=True)


print(f"Total de pares com match: {len(matched_pairs_df)}")
print(f"Total de perfis únicos de hotéis envolvidos nos matches para enriquecer: {len(all_unique_matched_hotels_to_enrich)}")
print("Iniciando chamadas à API para enriquecimento...")

enriched_results_lookup = {} # Dicionário para armazenar os resultados do enriquecimento

if API_URL_AVAILABLE:
    for index, row in all_unique_matched_hotels_to_enrich.iterrows():
        payload = {
            "hotel_name": row['hotel_name'],
            "city": row['city'],
            "country": row['country']
        }
        
        # Criar uma chave única para o cache de resultados
        enrichment_key = (row['hotel_name'], row['city'], row['country'])
        
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            enriched_info = response.json()
            
            # Armazenar os dados enriquecidos usando a chave
            enriched_results_lookup[enrichment_key] = {
                'enriched_category_stars': enriched_info.get('category_stars'),
                'enriched_review_score': enriched_info.get('review_score'),
                'enriched_amenities': ", ".join(enriched_info.get('amenities', []))
            }
            
            if (index + 1) % 50 == 0:
                print(f"  {index + 1}/{len(all_unique_matched_hotels_to_enrich)} perfis enriquecidos...")
                
        except requests.exceptions.RequestException as e:
            print(f"Erro ao enriquecer hotel '{row['hotel_name']}' ({row['city']}, {row['country']}): {e}")
            enriched_results_lookup[enrichment_key] = {
                'enriched_category_stars': None,
                'enriched_review_score': None,
                'enriched_amenities': None
            }
        time.sleep(0.01)
else:
    print("API mock não está disponível. Pulando chamadas de enriquecimento.")
    for index, row in all_unique_matched_hotels_to_enrich.iterrows():
        enrichment_key = (row['hotel_name'], row['city'], row['country'])
        enriched_results_lookup[enrichment_key] = {
            'enriched_category_stars': None,
            'enriched_review_score': None,
            'enriched_amenities': None
        }

print("\nEnriquecimento via API concluído!")
print("-" * 50)

# 4. Criar um DataFrame de lookup a partir dos resultados enriquecidos
enriched_df_map = pd.DataFrame([
    {
        'hotel_name_lookup': key[0],
        'city_lookup': key[1],
        'country_lookup': key[2],
        **value 
    } for key, value in enriched_results_lookup.items()
])

# 5. Unir os dados enriquecidos de volta ao DataFrame de pares com match
# Primeiro, para o lado A do match
final_matched_enriched = matched_pairs_with_details.merge(
    enriched_df_map,
    left_on=['hotel_name_A', 'city_A', 'country_A'],
    right_on=['hotel_name_lookup', 'city_lookup', 'country_lookup'],
    how='left'
)
final_matched_enriched.rename(columns={
    'enriched_category_stars': 'enriched_category_stars_A',
    'enriched_review_score': 'enriched_review_score_A',
    'enriched_amenities': 'enriched_amenities_A'
}, inplace=True)
final_matched_enriched.drop(columns=['hotel_name_lookup', 'city_lookup', 'country_lookup'], inplace=True)

# Em seguida, para o lado B do match
final_matched_enriched = final_matched_enriched.merge(
    enriched_df_map,
    left_on=['hotel_name_B', 'city_B', 'country_B'],
    right_on=['hotel_name_lookup', 'city_lookup', 'country_lookup'],
    how='left',
    suffixes=('_A_suffix', '_B_suffix')
)
final_matched_enriched.rename(columns={
    'enriched_category_stars': 'enriched_category_stars_B',
    'enriched_review_score': 'enriched_review_score_B',
    'enriched_amenities': 'enriched_amenities_B'
}, inplace=True)
final_matched_enriched.drop(columns=['hotel_name_lookup', 'city_lookup', 'country_lookup'], inplace=True)

# ================================================================
# 10. Exportar o Resultado Final Enriquecido
# ================================================================
output_enriched_csv_path = 'output_final_enriquecido.csv'
final_matched_enriched.to_csv(output_enriched_csv_path, index=False)
print(f"\nArquivo '{output_enriched_csv_path}' salvo com os dados dos matches e enriquecimento.")