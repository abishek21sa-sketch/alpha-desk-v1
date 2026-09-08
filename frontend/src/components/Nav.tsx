import Link from "next/link";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/pairs", label: "Pairs Trading" },
  { href: "/factors", label: "Factor L/S" },
  { href: "/ml", label: "ML Classifier" },
  { href: "/momentum", label: "TSMOM" },
  { href: "/pead", label: "PEAD" },
  { href: "/risk", label: "Risk" },
  { href: "/microstructure", label: "Market Making" },
  { href: "/vol-relval", label: "Vol Relative Value" },
  { href: "/execution", label: "Execution" },
];

export default function Nav() {
  return (
    <header
      style={{
        borderBottom: "1px solid var(--border)",
        background: "var(--surface-card)",
      }}
    >
      <div
        style={{
          maxWidth: 1120,
          margin: "0 auto",
          padding: "14px 24px",
          display: "flex",
          alignItems: "center",
          gap: 32,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>
            Alpha Desk
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>V1</span>
        </div>
        <nav style={{ display: "flex", gap: 4 }}>
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              style={{
                fontSize: 13,
                color: "var(--text-secondary)",
                padding: "6px 10px",
                borderRadius: 6,
                textDecoration: "none",
              }}
              className="nav-link"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
