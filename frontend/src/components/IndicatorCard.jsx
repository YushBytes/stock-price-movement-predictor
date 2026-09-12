export default function IndicatorCard({ name, columns, window: windowLabel, description }) {
  return (
    <div className="card flex flex-col gap-3 p-6 transition duration-300 hover:border-accent-500/30 hover:bg-white/[0.05]">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold text-white">{name}</h3>
        <span className="whitespace-nowrap rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-[11px] font-medium text-slate-300">
          {windowLabel}
        </span>
      </div>
      <p className="text-sm leading-relaxed text-slate-400">{description}</p>
      <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
        {columns.map((col) => (
          <code
            key={col}
            className="rounded-md border border-white/10 bg-black/30 px-2 py-0.5 font-mono text-[11px] text-accent-400"
          >
            {col}
          </code>
        ))}
      </div>
    </div>
  )
}
