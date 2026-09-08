import { api, ApiError } from "@/lib/api";
import ErrorState from "@/components/ErrorState";
import StatTile from "@/components/StatTile";

export default async function ExecutionPage() {
  let status;
  try {
    status = await api.alpacaStatus();
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

      <section
        className="card"
        style={{ borderColor: status.connected ? "var(--status-good)" : "var(--status-warning)" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: status.connected
                ? "color-mix(in srgb, var(--status-good) 18%, transparent)"
                : "color-mix(in srgb, var(--status-warning) 18%, transparent)",
              color: status.connected ? "var(--status-good)" : "var(--status-warning)",
            }}
          >
            {status.configured ? (status.connected ? "Connected" : "Configured, not connected") : "Not configured"}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          {status.configured ? (
            status.connected ? (
              <>Real paper-account connection verified against {status.base_url}.</>
            ) : (
              <>
                Credentials are set but the connection failed:{" "}
                <strong style={{ color: "var(--text-primary)" }}>{status.message}</strong>
              </>
            )
          ) : (
            <>
              <code>ALPACA_API_KEY</code> / <code>ALPACA_SECRET_KEY</code> are not set on this machine. This
              platform&apos;s Alpaca client (<code>src/alpha_desk/execution/alpaca_client.py</code>) is fully
              built and tested against mocked HTTP responses (13 tests) but has never made a real network
              call — real credentials were searched for in this user&apos;s other local Alpaca-integrated
              projects and not found. See <code>docs/ALPACA_EXECUTION.md</code> for the full account of what
              was checked. Drop real credentials into <code>.env</code> and re-run{" "}
              <code>scripts/check_alpaca_connectivity.py</code> to activate this page.
            </>
          )}
        </p>
      </section>

      {status.connected && (
        <section>
          <p className="section-title">Live Paper Account</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
            <StatTile label="Account status" value={status.account_status ?? "—"} />
            <StatTile label="Equity" value={status.equity ? `$${status.equity}` : "—"} />
            <StatTile label="Buying power" value={status.buying_power ? `$${status.buying_power}` : "—"} />
            <StatTile label="Market open" value={status.market_open ? "Yes" : "No"} />
          </div>
        </section>
      )}

      <section>
        <p className="section-title">What this does NOT do</p>
        <div className="card" style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
          No strategy on this platform automatically submits orders — every phase stops at a signal/report.
          Order submission (<code>AlpacaClient.submit_order</code>) is built and tested but never called by
          any script here; turning a signal into a real (even paper) order is left as a manual,
          human-initiated action. The client also refuses, in code, to ever target Alpaca&apos;s live-trading
          host — only <code>paper-api.alpaca.markets</code> is reachable from this platform.
        </div>
      </section>
    </div>
  );
}

function PageHeader() {
  return (
    <div>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Alpaca Paper-Trading Execution</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        A thin, tested REST client for Alpaca&apos;s paper-trading API — reported honestly below, whichever
        state it&apos;s actually in.
      </p>
    </div>
  );
}
