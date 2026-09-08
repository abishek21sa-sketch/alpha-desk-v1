import { api, ApiError } from "@/lib/api";
import StatTile from "@/components/StatTile";
import ErrorState from "@/components/ErrorState";
import EquityCurveChart from "@/components/EquityCurveChart";

export default async function PairsPage() {
  let report;
  try {
    report = await api.pairsReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  const notSignificant = report.deflated_sharpe_ratio < 0.95;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section
        className="card"
        style={{ borderColor: notSignificant ? "var(--status-warning)" : "var(--border)" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: notSignificant ? "color-mix(in srgb, var(--status-warning) 18%, transparent)" : "color-mix(in srgb, var(--status-good) 18%, transparent)",
              color: notSignificant ? "var(--status-warning)" : "var(--status-good)",
            }}
          >
            {notSignificant ? "Not statistically significant" : "Survives correction"}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          Best pair (<strong style={{ color: "var(--text-primary)" }}>{report.best_pair}</strong>) shows a{" "}
          {report.best_pair_net_sharpe.toFixed(2)} net Sharpe in isolation — but after Deflated Sharpe Ratio
          correction for having screened {report.n_pairs_screened} pairs and backtested {report.n_backtested}{" "}
          tradeable ones, it is <strong style={{ color: "var(--text-primary)" }}>indistinguishable from having
          gotten lucky</strong> at a 95% confidence level. Reported honestly, not hidden.
        </p>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
        <StatTile label="Pairs screened" value={report.n_pairs_screened.toLocaleString()} />
        <StatTile label="Tradeable (p<0.05)" value={report.n_tradeable.toString()} />
        <StatTile
          label="Best pair net Sharpe"
          value={report.best_pair_net_sharpe.toFixed(2)}
          note={`gross ${report.best_pair_gross_sharpe.toFixed(2)}, cost drag ${report.best_pair_cost_drag.toFixed(3)}`}
        />
        <StatTile
          label="Deflated Sharpe Ratio"
          value={report.deflated_sharpe_ratio.toFixed(3)}
          status={notSignificant ? "warning" : "good"}
          note="P(true Sharpe > luck benchmark)"
        />
        {report.pbo !== null && (
          <StatTile
            label="Prob. of Backtest Overfitting"
            value={report.pbo.toFixed(3)}
            status={report.pbo > 0.4 ? "critical" : "good"}
            note={`${report.pbo_n_combinations} CSCV combinations`}
          />
        )}
        <StatTile label="Half-life" value={`${report.best_pair_half_life_days.toFixed(0)}d`} note={`${report.best_pair_n_trades} trades`} />
      </section>

      <section>
        <p className="section-title">Best Pair Equity Curve ({report.best_pair}, net of costs)</p>
        <div className="card">
          <EquityCurveChart data={report.equity_curve} />
        </div>
      </section>

      <section>
        <p className="section-title">All {report.all_pairs.length} Backtested Pairs</p>
        <div className="card" style={{ padding: 0, overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Pair</th>
                <th>p-value</th>
                <th>Half-life</th>
                <th>Trades</th>
                <th>Net Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {report.all_pairs.map((p) => (
                <tr key={p.pair}>
                  <td style={{ fontWeight: p.pair === report.best_pair ? 700 : 400 }}>{p.pair}</td>
                  <td className="tabular-nums">{p.pvalue.toFixed(4)}</td>
                  <td className="tabular-nums">{p.half_life_days.toFixed(1)}d</td>
                  <td className="tabular-nums">{p.n_trades}</td>
                  <td
                    className="tabular-nums"
                    style={{ color: p.net_sharpe >= 0 ? "var(--status-good)" : "var(--status-critical)" }}
                  >
                    {p.net_sharpe.toFixed(2)}
                  </td>
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
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Statistical Arbitrage / Pairs Trading</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        Engle-Granger cointegration screening → Kalman-filter dynamic hedge ratio → z-score
        entry/exit → cost-aware backtest, deflated for how many pairs were tried.
      </p>
    </div>
  );
}
