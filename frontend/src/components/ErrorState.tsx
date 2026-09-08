export default function ErrorState({ message }: { message: string }) {
  return (
    <div
      className="card"
      style={{ borderColor: "var(--status-warning)", display: "flex", gap: 12, alignItems: "flex-start" }}
    >
      <span style={{ fontSize: 18 }} aria-hidden>
        ⚠
      </span>
      <div>
        <p style={{ margin: 0, fontWeight: 600, fontSize: 14 }}>Artifact not available yet</p>
        <p style={{ margin: "4px 0 0 0", fontSize: 13, color: "var(--text-secondary)" }}>{message}</p>
      </div>
    </div>
  );
}
