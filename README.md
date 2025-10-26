<<<<<<< HEAD
# Desafio de Matching de Hotéis + Enriquecimento via API

## Objetivo
Unir duas bases (`hotels_A.csv`, `hotels_B.csv`) **sem chave compartilhada** e construir um algoritmo de matching. Em seguida, **enriquecer** o resultado consultando uma **API externa simulada**.

## Arquivos
- `data/hotels_A.csv`, `data/hotels_B.csv`: bases de origem com `id_A`/`id_B` e colunas descritivas.
- `data/train.csv`: pares rotulados para treino/validação (`is_match` 0/1).
- `evaluate.py`: calcula *precision*, *recall*, *F1* do seu `output.csv`.
- `mock_api.py`: API FastAPI para enriquecimento (`/enrich`).

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
#.\.venv\Scripts\Activate.ps1
# Windows (cmd)
# .\.venv\Scripts\activate.bat
pip install -U fastapi uvicorn pydantic pandas numpy
# (Opcional) Subir API mock
python mock_api.py
# Avaliar uma submissão
python evaluate.py --pred output.csv --truth data/train.csv --out metrics.json
```

### Formato de saída esperado (`output.csv`)
Colunas obrigatórias: `id_A,id_B,predicted_match` (0/1).  
Você pode incluir colunas extras (ex.: `score`), serão ignoradas na métrica.

### Enriquecimento
Com a API rodando localmente em `http://localhost:8000`:
```bash
curl -X POST http://localhost:8000/enrich -H 'Content-Type: application/json' \
  -d '{"hotel_name":"Hotel Paulista","city":"São Paulo","country":"BR"}'
```

## Critérios de Avaliação
- Qualidade do matching (F1).
- Escalabilidade (blocking, vetorização, paralelismo).
- Limpeza de código, testes e reprodutibilidade.
- Uso correto da API de enriquecimento e consistência dos dados.

Boa sorte e bons experimentos! ✨
=======
# ds-matching-hotels
>>>>>>> 9ea18a8f41072e592469e428728fb6e3da49d101
