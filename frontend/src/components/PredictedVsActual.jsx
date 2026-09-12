import predictedVsActual from '../assets/predicted_vs_actual.png'

export default function PredictedVsActual() {
  return (
    <section id="prediction" className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Prediction</span>
        <h2 className="mt-3 text-3xl">Predicted vs. actual direction</h2>
        <p className="mt-4 text-slate-400">
          A static figure generated directly by the experiment notebook — not a live or interactive
          chart.
        </p>
      </div>

      <figure className="card mt-10 overflow-hidden p-3 sm:p-4">
        <img
          src={predictedVsActual}
          alt="Actual versus predicted next-day market direction over the held-out SPY test period."
          className="w-full rounded-xl"
        />
        <figcaption className="px-2 pb-1 pt-4 text-sm text-slate-400">
          Engineered Logistic Regression predictions versus actual target direction across the
          chronological test period.
        </figcaption>
      </figure>
    </section>
  )
}
