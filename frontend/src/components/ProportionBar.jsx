export default function ProportionBar({ segments }) {
  return (
    <div className="flex flex-col gap-4">
      <div
        className="flex h-10 w-full overflow-hidden rounded-full border border-white/10 bg-white/[0.03]"
        role="img"
        aria-label={segments.map((s) => `${s.label} ${s.pct}%`).join(', ')}
      >
        {segments.map((segment) => (
          <div
            key={segment.label}
            className={`flex items-center justify-center text-xs font-semibold ${segment.className}`}
            style={{ width: `${segment.pct}%` }}
          >
            {segment.pct >= 15 ? `${segment.pct}%` : null}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-6">
        {segments.map((segment) => (
          <div key={segment.label} className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${segment.dotClassName}`} aria-hidden="true" />
            <span className="text-sm text-slate-300">
              <span className="font-semibold text-white">{segment.label}</span> — {segment.count} (
              {segment.pct}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
