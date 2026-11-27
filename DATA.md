# Data documentation

IESF-Net combines regional carbon allowance price histories with shared textual information and market-specific institutional actions. This document describes the data interfaces of the source-code release.

## 1. Data sources and coverage

| Data | Source / public reference portal |
|---|---|
| Hubei (HBEA) | [Hubei Carbon Emissions Trading Center](https://www.hbets.cn/) |
| Guangdong (GDEA) | [Guangzhou Carbon Emissions Exchange](https://www.cnemission.com/) |
| Shanghai (SHEA) | [Shanghai Environment and Energy Exchange](https://overview.cneeex.com/); [Domestic Carbon Market Quotations](https://ets.sceex.com.cn/internal.htm?k=guo_nei_xing_qing&orderby=tradeTime+desc&url=mrhq_gn) |
| Shenzhen (SZEA) | [Domestic Carbon Market Quotations — Sichuan United Environment Exchange](https://ets.sceex.com.cn/internal.htm?k=guo_nei_xing_qing&orderby=tradeTime+desc&url=mrhq_gn) |
| News and market information | [Carbon Emissions Network](http://www.tanpaifang.com/) and [China Carbon Trading Network](http://www.tanjiaoyi.com/) |
| Formal institutional documents | [Ministry of Ecology and Environment](https://www.mee.gov.cn/), local competent authorities and the relevant trading institutions |

Regional trading institutions are the underlying sources of the market quotations, as reported by the authors. The public reference portals above include both original exchange websites and an aggregated domestic quotation service. The latter provides date-range queries for HBEA, GDEA, SHEA and SZEA, including closing price, volume and turnover. These links are reference access routes; the exact historical acquisition endpoints for the original experimental files and full-period coverage have not been verified. Use the regional allowance product codes, not national CEA or CCER, and distinguish closing prices from average transaction prices.

The text collection covers national and regional carbon market news, trading conditions, macro-level policies, industry developments, green finance and formal institutional arrangements. Formal documents are organized with reference to their issuing authority, version and applicable scope.

### Market observations

| Market | Observation period | Total | Reserved history | Train | Validation | Test |
|---|---|---:|---:|---:|---:|---:|
| Hubei | 2014-04-02 – 2025-12-31 | 2,706 | 90 | 1,831 | 392 | 393 |
| Guangdong | 2013-12-19 – 2025-12-31 | 2,626 | 90 | 1,763 | 380 | 393 |
| Shanghai | 2013-12-19 – 2025-12-30 | 1,803 | 90 | 1,115 | 256 | 342 |
| Shenzhen | 2022-04-21 – 2025-12-26 | 651 | 90 | 365 | 84 | 112 |

The chronological partitions follow `configs/default.yaml` and the manuscript's reported sample sizes. The forecast target is the next recorded price observation.

### Shared textual corpus

The authors report **69,842 unique effective documents** after cleaning, deduplication and alignment to at least one market. This is the size of the common corpus, not four separate market-specific corpora. The complete inventory has not been independently recounted. The markets share document-level BERT semantic vectors and FinBERT sentiment probabilities; institutional actions (PII) are assigned separately according to policy applicability. Market-level daily features are aggregated on each market's own observation calendar and may differ.

For Hubei, manuscript Table 5 reports text coverage from April 2014 to 14 November 2025: 2,449 of 2,673 inventory observations have newly available text (91.62%); the remaining 33 price observations follow the inventory period. Coverage over the full price calendar is 90.50% (2,449 / 2,706). These are manuscript-reported or derived statistics. The 69,842 count is used here in the shared-corpus sense specified by the authors.

## 2. Input files and fields

The expected local layout is shown below. The data files are not distributed with the repository.

```text
data/
  raw/
    prices/
      Hubei_observed_prices.csv
      Guangdong_observed_prices.csv
      Shanghai_observed_prices.csv
      Shenzhen_observed_prices.csv
    documents.jsonl
    actions.jsonl                 # optional external action register
  interim/                        # generated locally
  processed/
    Hubei.npz
    Guangdong.npz
    Shanghai.npz
    Shenzhen.npz
```

| Format | Fields and meaning |
|---|---|
| Price CSV | `date` (YYYY-MM-DD), `market`, `product`, `close` (CNY/tCO2), `volume` (trading volume), `turnover` (trading value) |
| Document JSONL | `document_id`, `title`, `body`, `publish_datetime`, `date_only_publish_flag`, `availability` (optional first-availability timestamp), `markets` (market labels), `source_url`, `source_type`, `publisher`, `issuer`, `instrument_id`, `version` |
| PII action JSONL | `canonical_action_id`, `categories`, `mechanisms`, `scope`, `markets` (or `market`), `issuer_tier`, `confidence`, `effective_date`, `availability`, `supporting_document_ids`, optional `supporting_documents` |

The price loader requires `date` and `close`, retains positive closing prices and replaces invalid or missing volume/turnover with zero (negative values are clipped to zero). Source volume/turnover units must be checked and harmonized before use; the loader does not convert them. Document and action timestamps use ISO date/time formats. The default pipeline extracts an action register under `data/interim/pii/`; an external register is optional.

## 3. Model-ready representation

Let `T` denote the number of market observations, including reserved history. The preprocessing pipeline generates one NPZ archive per market. The following arrays specify the data representation, not actual empirical values.

| Array | Shape | Description |
|---|---|---|
| `market`, `dates`, `partition` | scalar, `(T,)`, `(5,)` | Market, calendar and split metadata |
| `close` | `(T,)` | Closing prices / target series |
| `X_raw` | `(T, 4)` | Close, price change, log(1 + volume), log(1 + turnover) |
| `R` | `(T, 4)` | Four grouped causal CEEMDAN components |
| `sem` | `(T, 768)` | Mean BERT vector of newly available documents |
| `csi` | `(T, 8)` | Daily sentiment statistics from FinBERT |
| `doc_avail` | `(T,)` | Newly available text mask |
| `pii_attr` | `(T, 7, 16)` | Institutional attributes by category |
| `pii_new`, `pii_age` | `(T, 7)` each | Category update mask and event age |
| `pii_dt`, `pii_avail` | `(T,)` each | Memory interval and institutional availability |
| `lex`, `count` | `(T, 128)`, `(T, 2)` | Alternative lexical and news-count features |

Semantic vectors use frozen Chinese BERT; document sentiment uses frozen Chinese FinBERT (three probabilities). Encoder identifiers and revisions are in `configs/default.yaml`. The eight CSI features are mean tone, mean positive/neutral/negative probabilities, mean absolute tone, tone dispersion, log document count and document availability, in that order. The seven PII categories are Allocation, Compliance, Offsets, Coverage, Trading, MRV and Finance. The 16 attributes encode action counts, confidence, mechanism, scope, issuer tier, effective-date offset and supporting-document count; their order is defined in `iesf/text/pii_register.py`.

### Forecast inputs (default configuration)

Let `B` be batch size and `L` the scale-specific lookback (10, 30, 60 or 90 observations).

| Input / representation | Tensor shape |
|---|---|
| Standardized market features | `(B, L, 4)` |
| Standardized component for the selected scale | `(B, L, 1)` |
| Concatenated numerical GRU input | `(B, L, 5)` |
| Semantic history (before projection) | `(B, L, 768)` |
| CSI history (before projection) | `(B, L, 8)` |
| Text availability mask | `(B, L)` |
| PII-TAM memory history | `(B, L, 64)` |
| PII availability mask | `(B, L)` |
| Encoded / fused history | `(B, L, 64)` |
| Standardized component forecasts | `(B, 4)` |
| Final price forecast after component restoration and summation | `(B,)` |

The first three scales use numerical, Semantic, CSI and PII information; the fourth uses numerical information only. PII memory is computed over the historical prefix, not reset at the start of each lookback window. `lex` and `count` are comparison features, not separate branches of the default model. The model uses `MarketTensors` and target indices rather than one concatenated static input. See `iesf/data/dataset.py`, `iesf/models/iesf_net.py`, `iesf/models/pii_tam.py` and `configs/iesf_net.yaml`.

## 4. Temporal alignment and information availability

- Each origin has a 15:00 information cutoff (UTC+08:00). Text enters at the first eligible market cutoff on or after its availability; date-only releases are treated as available at day end. Publication, availability and policy effective dates are distinct.
- Shared document-level encodings are aggregated on market calendars. The current loader also filters Semantic/CSI by `markets`: null/empty labels permit sharing; explicit labels restrict eligibility. PII requires an applicable market assignment. Days without new text use zero vectors and an availability mask; institutional memory can persist and decay.
- For target index `i`, the forecast origin is `i-1`, and a window of length `L` reads indices `i-L` through `i-1`. CEEMDAN uses the observed price prefix through each decomposition origin (with its configured history cap). Frequency grouping and numerical normalization are fitted on reserved history and training rows only; the lexical comparison representation uses training-aligned documents.
- These are implementation rules, not an independent certification of the historical experiment. Retrospective document-scope merging, action-attribute consolidation and cache provenance require point-in-time verification on the original records.

## 5. Data preparation and availability

With authorized input files and dependencies installed, the preprocessing entry point is:

```bash
python scripts/run_pipeline.py --stages corpus encode pii decompose build
```

See `README.md` and `REVIEW_GUIDE.md` for execution and causality checks. Synthetic smoke tests demonstrate the interfaces, not empirical reproducibility.

This source-code release does not distribute the original price records, full text, derived features, action registers, trained checkpoints or empirical forecasts. The reference portals above provide access routes to market quotations and underlying publications, subject to availability and redistribution terms. They are not a guarantee of access to the exact historical experimental dataset. Further data access is subject to the actual retained materials, applicable permissions and journal requirements.
