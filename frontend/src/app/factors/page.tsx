import { api, ApiError } from "@/lib/api";
import StatTile from "@/components/StatTile";
import ErrorState from "@/components/ErrorState";
import EquityCurveChart from "@/components/EquityCurveChart";
import FactorWeightsChart from "@/components/FactorWeightsChart";
import ICTable from "@/components/ICTable";

export default async function FactorsPage() {
  let report;
  try {
    report = await api.factorsReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  const airtight = report.probabilistic_sharpe_ratio >= 0.95;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section className="card" style={{ borderColor: airtight ? "var(--status-good)" : "var(--status-warning)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: airtight
                ? "color-mix(in srgb, var(--status-good) 18%, transparent)"
                : "color-mix(in srgb, var(--status-warning) 18%, transparent)",
              color: airtight ? "var(--status-good)" : "var(--status-warning)",
            }}
          >
            {airtight ? "Statistically significant" : "Positive, not airtight"}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          Walk-forward net Sharpe of <strong style={{ color: "var(--text-primary)" }}>{report.net_sharpe.toFixed(2)}</strong>{" "}
          across {report.n_rebalances} monthly rebalances, {report.n_folds} folds — Probabilistic Sharpe Ratio of{" "}
          {report.probabilistic_sharpe_ratio.toFixed(2)} means roughly a{" "}
          {(report.probabilistic_sharpe_ratio * 100).toFixed(0)}% probability the true edge is positive.
          Encouraging, but short of a 95% confidence bar. One pre-specified configuration was run — no
          multi-config sweep, so no DSR correction is needed here (see methodology note below).
        </p>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
        <StatTile label="Net Sharpe" value={report.net_sharpe.toFixed(2)} note={`gross ${report.gross_sharpe.toFixed(2)}`} />
        <StatTile label="Cost drag" value={report.cost_drag.toFixed(3)} note="Sharpe points" />
        <StatTile
          label="Probabilistic Sharpe Ratio"
          value={report.probabilistic_sharpe_ratio.toFixed(3)}
          status={airtight ? "good" : "warning"}
        />
        <StatTile label="Rebalances" value={report.n_rebalances.toString()} note={`${report.n_folds} walk-forward folds`} />
      </section>

      <section>
        <p className="section-title">Equity Curve (net of costs, out-of-sample)</p>
        <div className="card">
          <EquityCurveChart data={report.equity_curve} />
        </div>
      </section>

      <section>
        <p className="section-title">Calibrated Factor Weights by Fold</p>
        <div className="card">
          <FactorWeightsChart foldWeights={report.fold_weights} />
        </div>
      </section>

      <section>
        <p className="section-title">Information Coefficient by Fold</p>
        <div className="card">
          <ICTable foldWeights={report.fold_weights} />
        </div>
      </section>
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Cross-Sectional Equity Factor Long/Short</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        Momentum, value, quality, low-vol → IC-calibrated composite score (weights fit on
        training data, never hand-picked) → dollar-neutral long/short → walk-forward, cost-aware backtest.
      </p>
    </div>
  );
}
