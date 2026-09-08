type Status = "good" | "warning" | "serious" | "critical" | null;

export default function StatTile({
  label,
  value,
  note,
  status = null,
}: {
  label: string;
  value: string;
  note?: string;
  status?: Status;
}) {
  const statusColor = status ? `var(--status-${status})` : undefined;
  return (
    <div className="card">
      <p className="stat-tile-label">{label}</p>
      <p className="stat-tile-value tabular-nums" style={statusColor ? { color: statusColor } : undefined}>
        {value}
      </p>
      {note && <p className="stat-tile-note">{note}</p>}
    </div>
  );
}
