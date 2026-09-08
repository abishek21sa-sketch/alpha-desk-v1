import type { FoldWeights } from "@/lib/types";

const FACTOR_ORDER = ["momentum", "low_vol", "value", "quality"];
const FACTOR_LABELS: Record<string, string> = {
  momentum: "Momentum",
  low_vol: "Low-Vol",
  value: "Value",
  quality: "Quality",
};

export default function ICTable({ foldWeights }: { foldWeights: FoldWeights[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Fold</th>
            {FACTOR_ORDER.map((f) => (
              <th key={f}>{FACTOR_LABELS[f]} IC</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {foldWeights.map((fw) => (
            <tr key={fw.fold}>
              <td>{fw.fold}</td>
              {FACTOR_ORDER.map((f) => {
                const ic = fw.ic_by_factor[f];
                const included = (fw.weights[f] ?? 0) > 0;
                return (
                  <td
                    key={f}
                    className="tabular-nums"
                    style={{
                      color: ic > 0 ? "var(--status-good)" : "var(--status-critical)",
                      fontWeight: included ? 700 : 400,
                      opacity: included ? 1 : 0.6,
                    }}
                    title={included ? "Included in composite score" : "Excluded (non-positive IC)"}
                  >
                    {ic >= 0 ? "+" : ""}
                    {ic.toFixed(3)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 10 }}>
        Bold = included in that fold&apos;s composite score (positive IC only, weight fit from training data).
      </p>
    </div>
  );
}
