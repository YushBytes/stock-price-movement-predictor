export default function Timeline({ steps }) {
  return (
    <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {steps.map((step, index) => (
        <li key={step.label} className="relative flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-accent-500/40 bg-accent-500/10 font-mono text-xs font-semibold text-accent-400">
              {index + 1}
            </span>
            <span className="text-sm font-semibold text-white">{step.label}</span>
          </div>
          <div className="card ml-1 flex flex-1 flex-col gap-1 border-l-2 border-l-accent-500/30 p-4">
            {step.range ? (
              <span className="font-mono text-xs text-slate-300">{step.range}</span>
            ) : null}
            {step.detail ? <span className="text-xs text-slate-400">{step.detail}</span> : null}
          </div>
          {index < steps.length - 1 ? (
            <span
              aria-hidden="true"
              className="absolute -right-2 top-4 hidden text-slate-600 sm:block lg:right-[-14px]"
            >
              →
            </span>
          ) : null}
        </li>
      ))}
    </ol>
  )
}
