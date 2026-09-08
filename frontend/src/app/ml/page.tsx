import { api, ApiError } from "@/lib/api";
import StatTile from "@/components/StatTile";
import ErrorState from "@/components/ErrorState";
import EquityCurveChart from "@/components/EquityCurveChart";
import CalibrationChart from "@/components/CalibrationChart";

export default async function MLPage() {
  let report;
  try {
    report = await api.mlReport();
  } catch (e) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PageHeader />
        <ErrorState message={e instanceof ApiError ? e.message : "Could not reach the API."} />
      </div>
    );
  }

  const pm = report.pooled_metrics;
  const hasSkill = pm.pr_auc > pm.base_rate + 0.02;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <PageHeader />

      <section className="card" style={{ borderColor: hasSkill ? "var(--status-good)" : "var(--status-critical)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span
            className="badge"
            style={{
              background: hasSkill
                ? "color-mix(in srgb, var(--status-good) 18%, transparent)"
                : "color-mix(in srgb, var(--status-critical) 18%, transparent)",
              color: hasSkill ? "var(--status-good)" : "var(--status-critical)",
            }}
          >
            {hasSkill ? "Shows measurable skill" : "No reliable skill detected"}
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", maxWidth: 720 }}>
          PR-AUC of <strong style={{ color: "var(--text-primary)" }}>{pm.pr_auc.toFixed(3)}</strong> against a{" "}
          {pm.base_rate.toFixed(2)} base rate is essentially at chance. The classifier IS well-calibrated
          (see below) — it&apos;s just calibrated to &quot;I don&apos;t know.&quot; Trading these near-chance
          predictions anyway produced a {report.net_sharpe.toFixed(2)} net Sharpe (PSR{" "}
          {report.probabilistic_sharpe_ratio.toFixed(2)}) — a presentable-looking number from a model with no
          measurable edge, reported specifically as an illustration of why PR-AUC/Brier must be checked{" "}
          <em>before</em> the Sharpe, not instead of it.
        </p>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
        <StatTile
          label="PR-AUC"
          value={pm.pr_auc.toFixed(3)}
          status={hasSkill ? "good" : "critical"}
          note={`base rate ${pm.base_rate.toFixed(2)}`}
        />
        <StatTile label="Brier score" value={pm.brier_score.toFixed(4)} note="0.25 = coin flip" />
        <StatTile label="Log-loss" value={pm.log_loss.toFixed(4)} note="ln(2) ≈ 0.693 = coin flip" />
        <StatTile label="Net Sharpe" value={report.net_sharpe.toFixed(2)} note={`PSR ${report.probabilistic_sharpe_ratio.toFixed(2)}`} />
        <StatTile label="OOS predictions" value={pm.n_obs.toLocaleString()} note={`${report.n_folds} folds, ${report.n_folds_skipped} skipped`} />
      </section>

      <section>
        <p className="section-title">Calibration (held-out predictions)</p>
        <div className="card">
          <CalibrationChart bins={pm.calibration_bins} />
        </div>
      </section>

      <section>
        <p className="section-title">Equity Curve (trading the predictions, net of costs)</p>
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
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px 0" }}>Machine Learning Return Classifier</h1>
      <p style={{ margin: 0, fontSize: 13, color: "var(--text-secondary)" }}>
        A calibrated gradient-boosted classifier (HistGradientBoostingClassifier) predicting
        whether a stock beats the cross-sectional median return — evaluated as a classifier
        first, traded second.
      </p>
    </div>
  );
}
