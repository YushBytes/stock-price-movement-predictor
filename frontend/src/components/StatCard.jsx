export default function StatCard({ value, label, sub }) {
  return (
    <div className="card flex flex-col gap-2 p-6 transition duration-300 hover:border-white/20 hover:bg-white/[0.05]">
      <span className="font-mono text-3xl font-semibold text-white sm:text-4xl">{value}</span>
      <span className="text-sm font-medium text-slate-200">{label}</span>
      {sub ? <span className="text-xs text-slate-400">{sub}</span> : null}
    </div>
  )
}
