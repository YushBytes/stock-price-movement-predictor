const POINTS = [
  {
    title: 'Next-day direction',
    body: 'Binary classification: will tomorrow’s close be higher (Up) or not (Down/Flat) than today’s?',
  },
  {
    title: 'Historical OHLCV data',
    body: 'Real daily Open/High/Low/Close/Volume for SPY, sourced from Yahoo Finance via yfinance.',
  },
  {
    title: 'Leak-free target',
    body: 'The label uses tomorrow’s close only to construct itself — never as a model input.',
  },
  {
    title: 'Chronological split',
    body: 'Train / validation / test are split strictly by date. No shuffling, no random sampling.',
  },
  {
    title: 'Naive baselines first',
    body: 'Persistence and Majority Class are evaluated before any model, as the bar to beat.',
  },
  {
    title: 'Raw vs. engineered',
    body: 'Two Logistic Regression models compared: raw lagged prices vs. raw + technical indicators.',
  },
]

export default function ProjectOverview() {
  return (
    <section id="overview" className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Overview</span>
        <h2 className="mt-3 text-3xl">What this project does</h2>
        <p className="mt-4 text-slate-400">
          A rigorous, honestly-evaluated time-series classification experiment — six ideas, twenty
          seconds.
        </p>
      </div>

      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {POINTS.map((point, index) => (
          <div key={point.title} className="card p-6">
            <span className="font-mono text-xs text-accent-400">{String(index + 1).padStart(2, '0')}</span>
            <h3 className="mt-2 text-base font-semibold text-white">{point.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{point.body}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
