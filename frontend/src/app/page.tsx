import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import StatTile from "@/components/StatTile";
import ErrorState from "@/components/ErrorState";
import type {
  AlpacaConnectivityStatus,
  DataBackboneStatus,
  FactorStrategyReport,
  MicrostructureReport,
  MLStrategyReport,
  MomentumStrategyReport,
  PairsStrategyReport,
  PeadStrategyReport,
  VolRelValReport,
} from "@/lib/types";

async function safe<T>(fn: () => Promise<T>): Promise<T | { error: string }> {
  try {
    return await fn();
  } catch (e) {
    return { error: e instanceof ApiError ? e.message : "Could not reach the API." };
  }
}

function isError(x: unknown): x is { error: string } {
  return typeof x === "object" && x !== null && "error" in x;
}

export default async function OverviewPage() {
  const [dataStatus, pairs, factors, ml, momentum, pead, microstructure, volRelVal, alpaca] = await Promise.all([
    safe(api.dataStatus),
    safe(api.pairsReport),
    safe(api.factorsReport),
    safe(api.mlReport),
    safe(api.momentumReport),
    safe(api.peadReport),
    safe(api.microstructureReport),
    safe(api.volRelValReport),
    safe(api.alpacaStatus),
  ]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 6px 0", letterSpacing: "-0.01em" }}>
          Alpha Desk V1
        </h1>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 640 }}>
          A quantitative trading research, validation, and risk platform built on real market,
          fundamentals, and macro data — every strategy below is validated through the same
          spine (purged walk-forward CV, deflated Sharpe, cost modeling) before its result is
          reported.
        </p>
      </div>

      <section>
        <p className="section-title">Data Backbone</p>
        {isError(dataStatus) ? (
          <ErrorState message={dataStatus.error} />
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
            <StatTile
              label="Price rows"
              value={(dataStatus as DataBackboneStatus).prices_rows.toLocaleString()}
              note={`${(dataStatus as DataBackboneStatus).prices_symbols} symbols`}
            />
            <StatTile
              label="Macro observations"
              value={(dataStatus as DataBackboneStatus).macro_rows.toLocaleString()}
              note={`${(dataStatus as DataBackboneStatus).macro_series} FRED series`}
            />
            <StatTile
              label="Fundamentals rows"
              value={(dataStatus as DataBackboneStatus).fundamentals_rows.toLocaleString()}
              note={`${(dataStatus as DataBackboneStatus).fundamentals_tickers} tickers, SEC EDGAR`}
            />
          </div>
        )}
      </section>

      <section>
        <p className="section-title">Strategy Summary</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
          <Link href="/pairs" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 3 — Pairs Trading</p>
              {isError(pairs) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    {(pairs as PairsStrategyReport).best_pair_net_sharpe.toFixed(2)}
                    <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400 }}> Sharpe</span>
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-critical)" }}>
                    Deflated: {(pairs as PairsStrategyReport).deflated_sharpe_ratio.toFixed(2)} — not significant
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/factors" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 4 — Factor Long/Short</p>
              {isError(factors) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    {(factors as FactorStrategyReport).net_sharpe.toFixed(2)}
                    <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400 }}> Sharpe</span>
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-good)" }}>
                    PSR: {(factors as FactorStrategyReport).probabilistic_sharpe_ratio.toFixed(2)} — positive, not airtight
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/ml" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 5 — ML Classifier</p>
              {isError(ml) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    {(ml as MLStrategyReport).pooled_metrics.pr_auc.toFixed(3)}
                    <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400 }}> PR-AUC</span>
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-warning)" }}>
                    ~base rate ({(ml as MLStrategyReport).pooled_metrics.base_rate.toFixed(2)}) — no reliable skill
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/momentum" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 6 — TSMOM</p>
              {isError(momentum) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    {(momentum as MomentumStrategyReport).net_sharpe.toFixed(2)}
                    <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400 }}> Sharpe</span>
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-good)" }}>
                    PSR: {(momentum as MomentumStrategyReport).probabilistic_sharpe_ratio.toFixed(2)} — strong, just short
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/pead" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 7 — PEAD</p>
              {isError(pead) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    p={(pead as PeadStrategyReport).significance.p_value.toFixed(2)}
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-critical)" }}>
                    not significant — honest negative result
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/microstructure" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 9 — Market Making</p>
              {isError(microstructure) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    ${(microstructure as MicrostructureReport).scenarios.moderate_risk_aversion.mean_pnl_per_session.toFixed(2)}
                    <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400 }}> mean P&amp;L/session</span>
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-warning)" }}>
                    simulated order flow, real vol calibration
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/vol-relval" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 10 — Vol Relative Value</p>
              {isError(volRelVal) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums">
                    {(volRelVal as VolRelValReport).scenarios.svxy_long_in_contango.net_sharpe_annualized.toFixed(2)}
                    <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 400 }}> Sharpe (SVXY)</span>
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-good)" }}>
                    PSR: {(volRelVal as VolRelValReport).scenarios.svxy_long_in_contango.probabilistic_sharpe_ratio.toFixed(2)} — high confidence
                  </p>
                </>
              )}
            </div>
          </Link>

          <Link href="/execution" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="card">
              <p className="stat-tile-label">Phase 11 — Alpaca Execution</p>
              {isError(alpaca) ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Not yet generated</p>
              ) : (
                <>
                  <p className="stat-tile-value tabular-nums" style={{ fontSize: 18 }}>
                    {(alpaca as AlpacaConnectivityStatus).configured
                      ? (alpaca as AlpacaConnectivityStatus).connected
                        ? "Connected"
                        : "Configured, not connected"
                      : "Not configured"}
                  </p>
                  <p className="stat-tile-note" style={{ color: "var(--status-warning)" }}>
                    client built &amp; tested, real credentials not found on this machine
                  </p>
                </>
              )}
            </div>
          </Link>
        </div>
      </section>

      <section>
        <p className="section-title">Roadmap</p>
        <div className="card" style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8 }}>
          Phase 1 Data Backbone · Phase 2 Validation Engine · Phase 3 Pairs Trading · Phase 4
          Factor Long/Short · Phase 5 ML Classifier · Phase 6 TSMOM · Phase 7 PEAD · Phase 8
          Risk &amp; Execution · Phase 9 Microstructure / Market Making · Phase 10 Vol
          Relative Value · Phase 11 Alpaca Paper-Trading Execution —{" "}
          <strong style={{ color: "var(--text-primary)" }}>done</strong>.
        </div>
      </section>
    </div>
  );
}
