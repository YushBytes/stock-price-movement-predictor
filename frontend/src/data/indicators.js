// Transcribed from src/features.py (TECHNICAL_INDICATOR_WINDOWS,
// TECHNICAL_INDICATOR_COLUMNS) and README.md "Technical Indicators".
// 8 columns grouped into 5 indicator families, matching the ML project's
// own grouping exactly.

export const indicatorFamilies = [
  {
    id: 'sma',
    name: 'Simple Moving Average',
    columns: ['SMA_10', 'SMA_20'],
    window: '10 / 20 days',
    description:
      'A trailing average of the closing price. Where price sits relative to its own recent average is a classic (not guaranteed) momentum signal.',
  },
  {
    id: 'rsi',
    name: 'RSI',
    columns: ['RSI_14'],
    window: '14 days',
    description:
      'Relative Strength Index, 0-100. Ratio of recent average gains to losses; extreme readings sometimes precede a reversal or continuation.',
  },
  {
    id: 'macd',
    name: 'MACD',
    columns: ['MACD', 'MACD_SIGNAL', 'MACD_HIST'],
    window: '12 / 26 / 9 days',
    description:
      'Difference between a fast and slow exponential moving average of price, plus a signal line. Crossovers are a widely used trend-change heuristic.',
  },
  {
    id: 'return',
    name: 'Daily Return',
    columns: ['RETURN_1D'],
    window: '1 day',
    description: "The simplest possible signal: yesterday's percentage price change.",
  },
  {
    id: 'volatility',
    name: 'Rolling Volatility',
    columns: ['VOLATILITY_20'],
    window: '20 days',
    description:
      'Rolling standard deviation of daily returns. Captures how turbulent recent trading has been, which affects how reliable any directional signal is.',
  },
]
