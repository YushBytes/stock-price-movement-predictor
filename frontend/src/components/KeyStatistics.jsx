import StatCard from './StatCard.jsx'
import { keyStats } from '../data/keyStats.js'

export default function KeyStatistics() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {keyStats.map((stat) => (
          <StatCard key={stat.id} value={stat.value} label={stat.label} sub={stat.sub} />
        ))}
      </div>
    </section>
  )
}
