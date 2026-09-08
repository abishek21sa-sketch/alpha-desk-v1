"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useColorScheme } from "@/lib/useColorScheme";

// Bar fill doesn't reliably resolve CSS custom properties in this Recharts
// version (confirmed on Scatter; treated the same way here rather than
// risk it) -- resolved hex literals per color scheme.
const GOOD = { light: "#0ca30c", dark: "#0ca30c" };
const CRITICAL = { light: "#d03b3b", dark: "#e66767" };

export default function PerAssetBarChart({ data }: { data: Record<string, number> }) {
  const scheme = useColorScheme();
  const rows = Object.entries(data)
    .map(([symbol, sharpe]) => ({ symbol, sharpe }))
    .sort((a, b) => b.sharpe - a.sharpe);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="symbol"
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--gridline)" }}
          tickLine={false}
        />
        <YAxis
          tickFormatter={(v: number) => v.toFixed(1)}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={36}
        />
        <Tooltip
          formatter={(value) => (typeof value === "number" ? value.toFixed(2) : value)}
          contentStyle={{
            background: "var(--surface-card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Bar dataKey="sharpe" radius={[3, 3, 0, 0]}>
          {rows.map((row) => (
            <Cell key={row.symbol} fill={row.sharpe >= 0 ? GOOD[scheme] : CRITICAL[scheme]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
