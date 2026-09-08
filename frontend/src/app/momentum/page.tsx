import { api, ApiError } from "@/lib/api";
import StatTile from "@/components/StatTile";
import ErrorState from "@/components/ErrorState";
import EquityCurveChart from "@/components/EquityCurveChart";
import PerAssetBarChart from "@/components/PerAssetBarChart";

export default async function MomentumPage() {
  let report;
  try {
    report = await api.momentumReport();
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
            {airtight ? "Statistically significant" : "Strong, just short of significant"}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          Net Sharpe of <strong style={{ color: "var(--text-primary)" }}>{report.net_sharpe.toFixed(2)}</strong> across{" "}
          {report.n_rebalances} monthly rebalances and {report.n_assets} ETF proxies — PSR of{" "}
          {report.probabilistic_sharpe_ratio.toFixed(2)} means a {(report.probabilistic_sharpe_ratio * 100).toFixed(0)}%
          probability the true edge is positive, just short of a 95% bar. Fixed-rule strategy (12-month trailing sign,
          vol-targeted) — no fitted parameters, so no walk-forward CV or DSR correction applies here.
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
        <StatTile label="Assets / Rebalances" value={`${report.n_assets} / ${report.n_rebalances}`} />
      </section>

      <section>
        <p className="section-title">Equity Curve (net of costs)</p>
        <div className="card">
          <EquityCurveChart data={report.equity_curve} />
        </div>
      </section>

      <section>
        <p className="section-title">Per-Asset Net Sharpe</p>
        <div className="card">
          <PerAssetBarChart data={report.per_asset_sharpe} />
        </div>
      </section>
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Time-Series Momentum (TSMOM)</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        Sign of each asset&apos;s own trailing 12-month return, sized to a common target volatility — per-asset,
        not cross-sectional, across 10 liquid ETF proxies (no free bulk futures data exists).
      </p>
    </div>
  );
}
