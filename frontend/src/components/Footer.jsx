import { siteConfig } from '../data/siteConfig.js'

export default function Footer() {
  return (
    <footer className="border-t border-white/10 px-6 py-12">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-white">Stock Price Movement Predictor</p>
          <p className="mt-1 text-xs text-slate-500">Option A — Time-Series ML</p>
        </div>

        <a
          href={siteConfig.githubUrl}
          target="_blank"
          rel="noreferrer noopener"
          className="btn-secondary w-fit"
        >
          View on GitHub
        </a>
      </div>

      <p className="mx-auto mt-8 max-w-6xl text-xs text-slate-500">
        Educational research project. Not financial advice.
      </p>
    </footer>
  )
}
