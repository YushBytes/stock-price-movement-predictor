import IndicatorCard from './IndicatorCard.jsx'
import { indicatorFamilies } from '../data/indicators.js'

export default function TechnicalIndicators() {
  return (
    <section id="indicators" className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <span className="section-eyebrow">Indicators</span>
        <h2 className="mt-3 text-3xl">Technical indicators</h2>
        <p className="mt-4 text-slate-400">
          Five indicator families, eight feature columns, all computed causally from historical
          price data via the maintained <code className="text-accent-400">ta</code> library and
          pandas.
        </p>
      </div>

      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {indicatorFamilies.map((family) => (
          <IndicatorCard
            key={family.id}
            name={family.name}
            columns={family.columns}
            window={family.window}
            description={family.description}
          />
        ))}
      </div>
    </section>
  )
}
