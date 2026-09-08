import { api, ApiError } from "@/lib/api";
import ErrorState from "@/components/ErrorState";
import StatTile from "@/components/StatTile";
import EquityCurveChart from "@/components/EquityCurveChart";
import type { VolRelValScenario } from "@/lib/types";

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function ScenarioCard({ title, subtitle, scenario }: { title: string; subtitle: string; scenario: VolRelValScenario }) {
  return (
    <section>
      <p className="section-title">{title}</p>
      <p style={{ margin: "-8px 0 12px 0", fontSize: 13, color: "var(--text-secondary)" }}>{subtitle}</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 16, marginBottom: 16 }}>
        <StatTile label="Days in position" value={pct(scenario.pct_days_in_position)} note={`${scenario.n_days} days total`} />
        <StatTile label="Gross Sharpe" value={scenario.gross_sharpe_annualized.toFixed(3)} />
        <StatTile
          label="Net Sharpe"
          value={scenario.net_sharpe_annualized.toFixed(3)}
          note={`cost drag ${scenario.cost_drag_annualized.toFixed(3)}`}
        />
        <StatTile
          label="Probabilistic Sharpe"
          value={scenario.probabilistic_sharpe_ratio.toFixed(3)}
          status={scenario.probabilistic_sharpe_ratio > 0.95 ? "good" : scenario.probabilistic_sharpe_ratio > 0.75 ? "warning" : "critical"}
        />
      </div>
      <div className="card">
        <EquityCurveChart data={scenario.equity_curve} />
      </div>
    </section>
  );
}

export default async function VolRelValPage() {
  let report;
  try {
    report = await api.volRelValReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section className="card">
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)" }}>
          Over {report.n_days_term_structure_history.toLocaleString()} days of real VIX/VIX3M history, the term
          structure has been in <strong style={{ color: "var(--text-primary)" }}>contango {pct(report.pct_contango_all_history)}</strong>{" "}
          of the time — the historically normal state, and the source of the persistent volatility risk premium
          this strategy trades via SVXY/VIXY rather than a synthetic futures curve. See
          docs/VOL_RELVAL_STRATEGY.md for the full disclosure, including a real, verified trace through the
          February 2018 &quot;Volpocalypse&quot;.
        </p>
      </section>

      <ScenarioCard
        title="Scenario 1 — SVXY long, contango (harvest the roll)"
        subtitle="Short front-month vol exposure while the curve prices in a normal term premium."
        scenario={report.scenarios.svxy_long_in_contango}
      />

      <ScenarioCard
        title="Scenario 2 — VIXY long, backwardation (crisis hedge)"
        subtitle="Long front-month vol exposure only when near-term vol is priced above further-out vol."
        scenario={report.scenarios.vixy_long_in_backwardation}
      />
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>VIX Term-Structure Relative Value</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        A fixed-rule strategy trading the VIX term structure via two real ETFs — the harvest side (SVXY in
        contango) and the mirror hedge side (VIXY in backwardation), reported separately since they do not
        show the same edge.
      </p>
    </div>
  );
}
