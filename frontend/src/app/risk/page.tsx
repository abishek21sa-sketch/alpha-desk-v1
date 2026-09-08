import { api, ApiError } from "@/lib/api";
import ErrorState from "@/components/ErrorState";
import StatTile from "@/components/StatTile";

const STRATEGY_LABELS: Record<string, string> = {
  pairs: "Pairs Trading",
  factors: "Factor L/S",
  ml: "ML Classifier",
  momentum: "TSMOM",
  pead: "PEAD",
};

const WINDOW_LABELS: Record<string, string> = {
  covid_crash_2020: "COVID Crash (Feb–Mar 2020)",
  rate_shock_2022: "2022 Rate Shock (Jan–Oct 2022)",
};

function pct(v: number | null) {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

export default async function RiskPage() {
  let report;
  try {
    report = await api.riskReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  const strategies = Object.keys(report.var_cvar);
  const ex = report.execution_example;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section
        className="card"
        style={{ borderColor: "var(--status-warning)" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: "color-mix(in srgb, var(--status-warning) 18%, transparent)",
              color: "var(--status-warning)",
            }}
          >
            Calibration bug caught before shipping
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          The Almgren-Chriss execution model&apos;s first run priced liquidating 1% of AAPL&apos;s ADV over 5 days at{" "}
          <strong style={{ color: "var(--text-primary)" }}>3,062 basis points (30.6%)</strong> — the math was
          right, the impact coefficients were an arbitrary multiple of share price with no link to real trading
          volume. Recalibrated to a standard &quot;impact at 100% of ADV&quot; reference point, the same order now
          costs 0.04–0.19bps. See methodology below.
        </p>
      </section>

      <section>
        <p className="section-title">Historical VaR / CVaR (95% / 99% confidence)</p>
        <div className="card" style={{ padding: 0, overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>VaR 95%</th>
                <th>CVaR 95%</th>
                <th>VaR 99%</th>
                <th>CVaR 99%</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s) => {
                const v = report.var_cvar[s];
                return (
                  <tr key={s}>
                    <td>{STRATEGY_LABELS[s] ?? s}</td>
                    <td className="tabular-nums">{v.var_95 !== undefined ? pct(v.var_95) : "—"}</td>
                    <td className="tabular-nums">{v.cvar_95 !== undefined ? pct(v.cvar_95) : "—"}</td>
                    <td className="tabular-nums">{v.var_99 !== undefined ? pct(v.var_99) : "—"}</td>
                    <td className="tabular-nums">{v.cvar_99 !== undefined ? pct(v.cvar_99) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <p className="section-title">Historical Stress Replay (each strategy&apos;s own real returns)</p>
        <div className="card" style={{ padding: 0, overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Window</th>
                <th>Cumulative Return</th>
                <th>Max Drawdown</th>
              </tr>
            </thead>
            <tbody>
              {strategies.flatMap((s) =>
                (report.stress[s] ?? []).map((w) => (
                  <tr key={`${s}-${w.window}`}>
                    <td>{STRATEGY_LABELS[s] ?? s}</td>
                    <td>{WINDOW_LABELS[w.window] ?? w.window}</td>
                    <td className="tabular-nums" style={{ color: !w.has_coverage ? "var(--text-muted)" : undefined }}>
                      {w.has_coverage ? pct(w.cumulative_return) : "no coverage"}
                    </td>
                    <td
                      className="tabular-nums"
                      style={{ color: w.has_coverage ? "var(--status-critical)" : "var(--text-muted)" }}
                    >
                      {w.has_coverage ? pct(w.max_drawdown) : "—"}
                    </td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <p className="section-title">
          Almgren-Chriss Optimal Execution — {ex.symbol} ({ex.shares_to_sell.toLocaleString()} shares, ~
          {((ex.shares_to_sell * ex.last_price) / ex.adv_dollars * 100).toFixed(1)}% of ADV, 5-day horizon)
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
          {Object.entries(ex.scenarios).map(([label, s]) => (
            <StatTile
              key={label}
              label={label.replace("_", " ")}
              value={`${s.expected_cost_bps.toFixed(2)}bps`}
              note={`cost std ${s.cost_std_bps.toFixed(0)}bps · λ=${s.risk_aversion.toExponential(0)}`}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Risk & Execution Control Tower</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        Historical VaR/CVaR and real crisis-window replay across every strategy, plus Almgren-Chriss optimal
        execution — wraps every phase built so far rather than adding a new strategy.
      </p>
    </div>
  );
}
