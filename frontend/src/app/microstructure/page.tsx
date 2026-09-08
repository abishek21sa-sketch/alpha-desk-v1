import { api, ApiError } from "@/lib/api";
import ErrorState from "@/components/ErrorState";
import StatTile from "@/components/StatTile";
import PerAssetBarChart from "@/components/PerAssetBarChart";

const SCENARIO_LABELS: Record<string, string> = {
  low_risk_aversion: "Low Risk Aversion",
  moderate_risk_aversion: "Moderate Risk Aversion",
  high_risk_aversion: "High Risk Aversion",
};

function pct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}

export default async function MicrostructurePage() {
  let report;
  try {
    report = await api.microstructureReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  const scenarios = Object.entries(report.scenarios);
  const pnlByScenario = Object.fromEntries(
    scenarios.map(([name, s]) => [SCENARIO_LABELS[name] ?? name, s.mean_pnl_per_session]),
  );
  const c = report.calibration;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section className="card" style={{ borderColor: "var(--status-warning)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: "color-mix(in srgb, var(--status-warning) 18%, transparent)",
              color: "var(--status-warning)",
            }}
          >
            Simulated order flow, real volatility calibration
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          Unlike every other strategy on this platform, order arrivals and the price path here are{" "}
          <strong style={{ color: "var(--text-primary)" }}>simulated</strong> (real exchange tick data is
          600MB+/day in a proprietary binary protocol with no off-the-shelf parser). What <em>is</em> real:
          the volatility driving the simulation — {report.symbol}&apos;s trailing daily volatility of{" "}
          <strong style={{ color: "var(--text-primary)" }}>{(c.daily_vol * 100).toFixed(2)}%</strong>, pulled
          from this platform&apos;s own warehouse. See docs/MICROSTRUCTURE_STRATEGY.md for the full disclosure.
        </p>
      </section>

      <section>
        <p className="section-title">Calibration ({report.symbol})</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
          <StatTile label="Last price" value={`$${c.last_price.toFixed(2)}`} />
          <StatTile label="Daily volatility" value={`${(c.daily_vol * 100).toFixed(2)}%`} note="trailing 252 sessions, real" />
          <StatTile label="Sigma ($/day)" value={`$${c.sigma_dollar_per_day.toFixed(2)}`} />
          <StatTile
            label="Arrival intensity × dt"
            value={c.arrival_intensity_x_dt.toFixed(3)}
            note={c.arrival_intensity_x_dt < 0.1 ? "under 0.1 threshold" : "WARNING: near saturation"}
            status={c.arrival_intensity_x_dt < 0.1 ? "good" : "warning"}
          />
        </div>
      </section>

      <section>
        <p className="section-title">Mean P&amp;L per session, by risk aversion (gamma)</p>
        <div className="card">
          <PerAssetBarChart data={pnlByScenario} />
        </div>
      </section>

      <section>
        <p className="section-title">
          Scenario detail ({c.n_sessions_per_scenario} simulated sessions each, kappa={c.kappa}, arrival
          intensity={c.arrival_intensity}, inventory limit={c.inventory_limit})
        </p>
        <div className="card" style={{ padding: 0, overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Gamma</th>
                <th>Mean P&amp;L</th>
                <th>Std P&amp;L</th>
                <th>Fills/session</th>
                <th>Profitable</th>
                <th>Mean max |inventory|</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map(([name, s]) => (
                <tr key={name}>
                  <td>{SCENARIO_LABELS[name] ?? name}</td>
                  <td className="tabular-nums">{s.gamma}</td>
                  <td className="tabular-nums">${s.mean_pnl_per_session.toFixed(2)}</td>
                  <td className="tabular-nums">${s.std_pnl_per_session.toFixed(2)}</td>
                  <td className="tabular-nums">{s.mean_fills_per_session.toFixed(1)}</td>
                  <td className="tabular-nums">{pct(s.pct_sessions_profitable)}</td>
                  <td className="tabular-nums">{s.mean_max_abs_inventory.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Market Making (Avellaneda-Stoikov)</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        Optimal quoting under inventory risk, simulated order flow calibrated to real volatility — see the
        disclosure below before reading these numbers as real fills.
      </p>
    </div>
  );
}
