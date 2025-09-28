# Scripts de Análise de Normalização

Esta pasta contém os scripts utilitários para análise e processamento de normalização de áudio.

## Scripts Disponíveis

### 📊 `generate_normalization_report.py`
Gera relatórios detalhados de análise de normalização para o dataset.

**Uso:**
```bash
# Análise completa
uv run python scripts/generate_normalization_report.py --dataset-path organized_audio

# Teste rápido
uv run python scripts/generate_normalization_report.py --quick-test --dataset-path test_audio_sample
```

### 📈 `visualize_normalization.py`
Cria visualizações publication-ready dos dados de normalização.

**Uso:**
```bash
# Todas as visualizações
uv run python scripts/visualize_normalization.py --reports-dir normalization_reports

# Apenas figura resumo para publicação
uv run python scripts/visualize_normalization.py --publication-only
```

### 🧪 `test_normalizer.py`
Script de teste para validar o funcionamento do normalizador de áudio.

**Uso:**
```bash
uv run python scripts/test_normalizer.py
```

## Execução Automatizada

Para executar todo o pipeline de análise, use o script principal na raiz:

```bash
# Análise completa do dataset real
python run_full_analysis.py
```

## Estrutura dos Outputs

Os scripts geram os seguintes arquivos:

```
normalization_reports/
├── complete_dataset_analysis_detailed.csv      # Métricas por arquivo
├── complete_dataset_analysis_class_summary.csv # Estatísticas por classe
├── complete_dataset_analysis_comparison.csv    # Comparação entre métodos
├── complete_dataset_analysis_report.txt        # Relatório textual
└── figures/                                    # Visualizações
    ├── level_distributions.png
    ├── normalization_effectiveness.png
    ├── class_comparison.png
    ├── gain_distributions.png
    ├── class_statistics.png
    └── publication_summary.png
```

## Para Tese de Doutorado

Estes scripts geram:
- **Dados quantitativos** para justificar escolhas metodológicas
- **Figuras publication-ready** para capítulos da tese
- **Comparações estatísticas** rigorosas entre métodos
- **Caracterização do dataset** com métricas acadêmicas
