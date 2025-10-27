# Case Matching e Enriquecimento de Hotéis 

Este case demonstra a construção de um pipeline para identificar e consolidar registros de hotéis duplicados entre duas bases de dados, seguido pelo enriquecimento dos dados matchados via integração com uma API externa.

## Solução

*   Matching:Utilização de blocking (textual e geográfico) e métricas de similaridade (Levenshtein, Haversine) para identificar hotéis correspondentes.
*   Enriquecimento Via API: Integração com uma API RESTful (simulada) para adicionar informações valiosas como classificação por review, pontuação de avaliações e comodidades aos hotéis que tiveram matching.
*   Output Consolidado: Geração de um arquivo final que apresenta os hotéis que obtiveram matching com suas informações descritivas e enriquecidas de forma unificada.

## Executar o Projeto

Siga estes passos para rodar a solução:

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/FerreiraGabrielw/ds-matching-hotels.git
    cd ds-matching-hotels
    ```

2.  **Configure o Ambiente:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Inicie a Mock API:**
    Abra um **novo terminal** (com o ambiente virtual ativado) e execute:
    ```bash
    python scripts/mock_api.py
    ```
    Mantenha este terminal aberto durante a execução do pipeline principal.

4.  **Execute o Pipeline Principal:**
    No terminal original (com o ambiente virtual ativado), execute:
    ```bash
    python src/hotel_matching_pipeline_gabrielferreira.py
    ```

Após a execução, os resultados serão:
*   `output.csv`: Pares matchados (id_A, id_B, predicted_match) para avaliação F1.
*   `output_final_enriquecido.csv`: O entregável final, com os dados dos hotéis matchados e enriquecidos.

## Referências 

Este projeto foi desenvolvido com o apoio dos seguintes materiais:

*   **Data Matching in the Hotel Industry:** https://mtrdesign.medium.com/data-matching-in-the-hotel-industry-9d74a1f1951d
*   **A Study of Machine Learning Based Approach for Hotels' Matching:** https://www.researchgate.net/publication/361714582_A_Study_of_Machine_Learning_Based_Approach_for_Hotels%27_Matching