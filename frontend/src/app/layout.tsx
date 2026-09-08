import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Nav from "@/components/Nav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Alpha Desk V1",
  description: "Quantitative trading research, validation, and risk platform.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Nav />
        <main style={{ maxWidth: 1120, margin: "0 auto", padding: "32px 24px", width: "100%", flex: 1 }}>
          {children}
        </main>
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "16px 24px",
            fontSize: 12,
            color: "var(--text-muted)",
            textAlign: "center",
          }}
        >
          Research artifacts, not investment advice. See each page&apos;s methodology notes for known limitations.
        </footer>
      </body>
    </html>
  );
}
