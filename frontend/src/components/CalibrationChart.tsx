"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CalibrationBin } from "@/lib/types";
import { useColorScheme } from "@/lib/useColorScheme";

// Recharts' Scatter does not resolve CSS custom properties for `fill`
// (confirmed: renders solid black instead) -- resolved hex literals here,
// matching the palette's dark/light categorical blue step.
const BLUE = { light: "#2a78d6", dark: "#3987e5" };
const MUTED = { light: "#898781", dark: "#898781" };

export default function CalibrationChart({ bins }: { bins: CalibrationBin[] }) {
  const scheme = useColorScheme();
  const domainMin = Math.min(0.3, ...bins.map((b) => Math.min(b.mean_predicted, b.mean_actual)) as number[]) - 0.02;
  const domainMax = Math.max(0.7, ...bins.map((b) => Math.max(b.mean_predicted, b.mean_actual)) as number[]) + 0.02;
  const diagonal = [
    { mean_predicted: domainMin, perfect: domainMin },
    { mean_predicted: domainMax, perfect: domainMax },
  ];

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" />
        <XAxis
          dataKey="mean_predicted"
          type="number"
          domain={[domainMin, domainMax]}
          tickFormatter={(v: number) => v.toFixed(2)}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={{ stroke: "var(--gridline)" }}
          tickLine={false}
          name="Mean predicted"
        />
        <YAxis
          dataKey="mean_actual"
          type="number"
          domain={[domainMin, domainMax]}
          tickFormatter={(v: number) => v.toFixed(2)}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={44}
          name="Mean actual"
        />
        <Tooltip
          cursor={{ stroke: "var(--gridline)" }}
          formatter={(value) => (typeof value === "number" ? value.toFixed(3) : value)}
          contentStyle={{
            background: "var(--surface-card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend
          formatter={(name: string) => <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>{name}</span>}
        />
        <Line
          data={diagonal}
          dataKey="perfect"
          stroke={MUTED[scheme]}
          strokeDasharray="4 4"
          strokeWidth={1.5}
          dot={false}
          legendType="none"
          name="Perfect calibration"
          isAnimationActive={false}
        />
        <Scatter data={bins} dataKey="mean_actual" fill={BLUE[scheme]} name="Observed bins" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
