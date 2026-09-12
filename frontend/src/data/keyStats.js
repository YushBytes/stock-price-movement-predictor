// Transcribed from the ML project's README.md ("Data Acquisition",
// "Engineered Feature Model") and notebooks/stock_price_movement_predictor.ipynb
// (Sections 4, 11, 12, 13). Cross-checked against results/final_comparison.csv
// (819 test rows) and the 126-test pytest run. Not computed by the frontend.

export const keyStats = [
  {
    id: 'observations',
    value: '5,457',
    label: 'Raw OHLCV Observations',
    sub: 'SPY, 2005-01-03 to 2026-09-11',
  },
  {
    id: 'features',
    value: '28',
    label: 'Engineered Features',
    sub: '20 raw lagged OHLCV + 8 technical indicators',
  },
  {
    id: 'test-observations',
    value: '819',
    label: 'Test Observations',
    sub: 'Held out, never used for fitting',
  },
  {
    id: 'indicator-families',
    value: '5',
    label: 'Indicator Families',
    sub: 'SMA, RSI, MACD, Return, Volatility',
  },
  {
    id: 'tests',
    value: '126 / 126',
    label: 'Tests Passing',
    sub: 'Deterministic, network-free unit tests',
  },
  {
    id: 'test-period',
    value: '2023 – 2026',
    label: 'Held-Out Test Period',
    sub: '2023-06-06 to 2026-09-10',
  },
]

// Supplementary figures referenced in the Leakage Prevention and
// Project Overview sections (README.md "Target Construction and
// Leakage Prevention", "Raw Price/Volume Model", "Engineered Feature Model").
export const featureCounts = {
  raw: 20,
  technicalIndicators: 8,
  engineered: 28,
}

export const splitDates = {
  train: { start: '2005-01-03', end: '2020-03-05', rows: 3819 },
  validation: { start: '2020-03-06', end: '2023-06-05', rows: 818 },
  test: { start: '2023-06-06', end: '2026-09-10', rows: 819 },
}
