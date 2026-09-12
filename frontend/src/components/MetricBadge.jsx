export default function MetricBadge({ label, modelName, value }) {
  return (
    <div className="card flex items-center gap-3 px-5 py-4">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500/15 text-accent-400">
        <svg viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5" aria-hidden="true">
          <path
            fillRule="evenodd"
            d="M10 1.5a8.5 8.5 0 100 17 8.5 8.5 0 000-17zM13.36 8.36a.75.75 0 10-1.06-1.06L9 10.6 7.7 9.3a.75.75 0 00-1.06 1.06l1.83 1.83a.75.75 0 001.06 0l3.83-3.83z"
            clipRule="evenodd"
          />
        </svg>
      </span>
      <div className="flex flex-col">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</span>
        <span className="text-sm text-slate-100">
          <span className="font-semibold text-white">{modelName}</span> — {value}
        </span>
      </div>
    </div>
  )
}
