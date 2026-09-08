"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FoldWeights } from "@/lib/types";

// Fixed categorical order/color -- never cycled, never re-assigned by rank.
const FACTOR_COLORS: Record<string, string> = {
  momentum: "var(--series-blue)",
  low_vol: "var(--series-orange)",
  value: "var(--series-aqua)",
  quality: "var(--series-yellow)",
};
const FACTOR_ORDER = ["momentum", "low_vol", "value", "quality"];
const FACTOR_LABELS: Record<string, string> = {
  momentum: "Momentum",
  low_vol: "Low-Vol",
  value: "Value",
  quality: "Quality",
};

export default function FactorWeightsChart({ foldWeights }: { foldWeights: FoldWeights[] }) {
  const data = foldWeights.map((fw) => ({
    fold: `Fold ${fw.fold}`,
    ...fw.weights,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="fold"
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--gridline)" }}
          tickLine={false}
        />
        <YAxis
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        <Tooltip
          formatter={(value, name) => [
            typeof value === "number" ? `${(value * 100).toFixed(1)}%` : value,
            FACTOR_LABELS[String(name)] ?? name,
          ]}
          contentStyle={{
            background: "var(--surface-card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend
          formatter={(name: string) => (
            <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>
              {FACTOR_LABELS[name] ?? name}
            </span>
          )}
          iconType="circle"
          iconSize={8}
        />
        {FACTOR_ORDER.map((factor) => (
          <Bar key={factor} dataKey={factor} stackId="w" fill={FACTOR_COLORS[factor]} radius={0} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
