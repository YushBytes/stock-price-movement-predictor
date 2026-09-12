// Transcribed from src/preprocessing.py::compute_class_balance() output on
// the real test partition (notebooks/stock_price_movement_predictor.ipynb,
// Section 8 "Class Balance"; also reported in README.md "Target
// Construction and Leakage Prevention"). This is the TEST partition's
// distribution specifically -- the number every model in modelResults.js
// was evaluated against.

export const testClassBalance = {
  total: 819,
  up: { count: 463, pct: 56.53 },
  downOrFlat: { count: 356, pct: 43.47 },
}
