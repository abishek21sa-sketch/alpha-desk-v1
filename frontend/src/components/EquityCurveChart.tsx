"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityCurvePoint } from "@/lib/types";

function formatPct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function formatDateTick(d: string) {
  const date = new Date(d);
  return date.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

export default function EquityCurveChart({ data }: { data: EquityCurvePoint[] }) {
  const isPositive = data.length > 0 && data[data.length - 1].cumulative_return >= 0;
  const color = isPositive ? "var(--status-good)" : "var(--status-critical)";

  // Recharts' auto domain pads generously (sometimes near-symmetric around
  // 0 regardless of the data's actual range), which flattens the visible
  // line for a series that never moves far from zero. A tight, data-driven
  // domain (with a little breathing room) keeps the actual movement legible.
  const values = data.map((d) => d.cumulative_return);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const pad = Math.max((max - min) * 0.1, 0.01);
  const domain: [number, number] = [min - pad, max + pad];

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.22} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={formatDateTick}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={{ stroke: "var(--gridline)" }}
          tickLine={false}
          minTickGap={40}
        />
        <YAxis
          domain={domain}
          tickFormatter={formatPct}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={56}
        />
        <Tooltip
          formatter={(value) => [typeof value === "number" ? formatPct(value) : value, "Cumulative return"]}
          labelFormatter={(label) =>
            typeof label === "string" ? new Date(label).toLocaleDateString("en-US", { dateStyle: "medium" }) : label
          }
          contentStyle={{
            background: "var(--surface-card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "var(--text-secondary)" }}
        />
        <Area
          type="monotone"
          dataKey="cumulative_return"
          stroke={color}
          strokeWidth={2.5}
          fill="url(#equityFill)"
          dot={false}
          activeDot={{ r: 4 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
