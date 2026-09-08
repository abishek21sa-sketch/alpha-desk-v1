import { api, ApiError } from "@/lib/api";
import StatTile from "@/components/StatTile";
import ErrorState from "@/components/ErrorState";
import EquityCurveChart from "@/components/EquityCurveChart";
import PerAssetBarChart from "@/components/PerAssetBarChart";

export default async function PeadPage() {
  let report;
  try {
    report = await api.peadReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  const sig = report.significance;
  const significant = sig.p_value < 0.05;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section className="card" style={{ borderColor: significant ? "var(--status-good)" : "var(--status-critical)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: significant
                ? "color-mix(in srgb, var(--status-good) 18%, transparent)"
                : "color-mix(in srgb, var(--status-critical) 18%, transparent)",
              color: significant ? "var(--status-good)" : "var(--status-critical)",
            }}
          >
            {significant ? "Statistically significant" : "No significant effect"}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          Across {report.n_events_total.toLocaleString()} real SEC 8-K filing events, mean 21-day drift after a
          positive announcement ({(sig.mean_drift_positive * 100).toFixed(2)}%) vs. a negative one (
          {(sig.mean_drift_negative * 100).toFixed(2)}%) — a spread of only{" "}
          {(sig.drift_spread * 100).toFixed(2)} points, Welch t-test p={sig.p_value.toFixed(3)}.{" "}
          <strong style={{ color: "var(--text-primary)" }}>Not remotely significant</strong> — a clean negative
          result, on a broad &quot;any 8-K&quot; proxy rather than item-filtered earnings releases specifically (see
          methodology). A real bug (self-fulfilling first-day return) was caught and fixed before trusting this —
          the fix dropped a fake 2.42 gross Sharpe down to a realistic 0.10.
        </p>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
        <StatTile
          label="Drift spread"
          value={`${(sig.drift_spread * 100).toFixed(2)}pp`}
          status={significant ? "good" : "critical"}
          note={`p=${sig.p_value.toFixed(3)}`}
        />
        <StatTile label="Net Sharpe" value={report.net_sharpe.toFixed(3)} note={`gross ${report.gross_sharpe.toFixed(3)}`} />
        <StatTile label="Cost drag" value={report.cost_drag.toFixed(3)} note="Sharpe points" />
        <StatTile
          label="Events traded"
          value={report.n_events_traded.toLocaleString()}
          note={`of ${report.n_events_total.toLocaleString()} total, max ${report.max_concurrent_trades} concurrent`}
        />
      </section>

      <section>
        <p className="section-title">Mean 21-Day Drift by Announcement Direction</p>
        <div className="card">
          <PerAssetBarChart
            data={{
              "Positive announcement": sig.mean_drift_positive,
              "Negative announcement": sig.mean_drift_negative,
            }}
          />
        </div>
      </section>

      <section>
        <p className="section-title">Equity Curve (calendar-time portfolio, net of costs)</p>
        <div className="card">
          <EquityCurveChart data={report.equity_curve} />
        </div>
      </section>
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Earnings/Event-Driven Drift (PEAD)</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        Real SEC EDGAR 8-K filing timestamps as the event set — does the announcement-day reaction predict
        continued drift in the same direction over the next month?
      </p>
    </div>
  );
}
