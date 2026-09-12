import Navbar from '../components/Navbar.jsx'
import Hero from '../components/Hero.jsx'
import ProjectOverview from '../components/ProjectOverview.jsx'
import KeyStatistics from '../components/KeyStatistics.jsx'
import ModelComparison from '../components/ModelComparison.jsx'
import TechnicalIndicators from '../components/TechnicalIndicators.jsx'
import LeakagePrevention from '../components/LeakagePrevention.jsx'
import PredictedVsActual from '../components/PredictedVsActual.jsx'
import ClassBalance from '../components/ClassBalance.jsx'
import FinalFindings from '../components/FinalFindings.jsx'
import Limitations from '../components/Limitations.jsx'
import Footer from '../components/Footer.jsx'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-ink-950">
      <Navbar />
      <main>
        <Hero />
        <ProjectOverview />
        <KeyStatistics />
        <ModelComparison />
        <TechnicalIndicators />
        <LeakagePrevention />
        <PredictedVsActual />
        <ClassBalance />
        <FinalFindings />
        <Limitations />
      </main>
      <Footer />
    </div>
  )
}
