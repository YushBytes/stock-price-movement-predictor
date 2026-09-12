const TAGS = ['SPY', 'Daily OHLCV', 'Binary Classification']

export default function Hero() {
  return (
    <section id="top" className="relative overflow-hidden px-6 pb-20 pt-24 sm:pt-32">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[32rem] bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,rgba(45,212,191,0.16),transparent)]"
      />

      <div className="mx-auto flex max-w-4xl flex-col items-center text-center">
        <span className="section-eyebrow animate-fade-up">GCSRM 2026 · Option A</span>

        <h1
          className="mt-5 animate-fade-up text-4xl font-semibold text-white sm:text-5xl md:text-6xl"
          style={{ animationDelay: '80ms' }}
        >
          Stock Price Movement Predictor
        </h1>

        <p
          className="mt-6 max-w-2xl animate-fade-up text-lg text-slate-400 sm:text-xl"
          style={{ animationDelay: '160ms' }}
        >
          Leak-Free Time-Series Machine Learning for Next-Day Market Direction
        </p>

        <div
          className="mt-8 flex animate-fade-up flex-wrap items-center justify-center gap-2"
          style={{ animationDelay: '220ms' }}
        >
          {TAGS.map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-sm text-slate-300"
            >
              {tag}
            </span>
          ))}
        </div>

        <div
          className="mt-10 flex animate-fade-up flex-col gap-3 sm:flex-row"
          style={{ animationDelay: '280ms' }}
        >
          <a href="#results" className="btn-primary">
            Explore Results
          </a>
          <a href="#overview" className="btn-secondary">
            View Methodology
          </a>
        </div>

        <p
          className="mt-12 max-w-xl animate-fade-up text-xs text-slate-500"
          style={{ animationDelay: '340ms' }}
        >
          Educational research project. Not a trading system, not financial advice, and no claim
          of profitable or reliable market prediction.
        </p>
      </div>
    </section>
  )
}
