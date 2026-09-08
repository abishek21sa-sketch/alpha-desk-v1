"use client";

import { useEffect, useState } from "react";

/**
 * Recharts does not reliably resolve CSS custom properties (`var(--x)`) for
 * every prop on every mark type -- confirmed against this app's palette:
 * `stroke="var(--status-good)"` on an Area/Line resolves correctly, but
 * `fill="var(--series-blue)"` on a Scatter symbol computed to solid black
 * (getComputedStyle showed rgb(0,0,0)), making the dots invisible on the
 * dark surface. Charts that hit this need a resolved hex literal instead
 * of a var() reference -- this hook is the one place that resolution
 * happens, driven by the same media query the CSS tokens themselves use.
 */
export function useColorScheme(): "light" | "dark" {
  // Must default to "dark" (matching the server-rendered guess, since
  // `window` doesn't exist during SSR) and correct it inside the effect --
  // a lazy useState initializer would read the real value on the client's
  // very first render and mismatch the server-rendered HTML during
  // hydration. That means this effect legitimately calls setState
  // synchronously for the initial read, which is what
  // react-hooks/set-state-in-effect normally warns against; it's correct
  // here, not an oversight.
  const [scheme, setScheme] = useState<"light" | "dark">("dark");

  useEffect(() => {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see comment above
    setScheme(mql.matches ? "dark" : "light");
    const handler = (e: MediaQueryListEvent) => setScheme(e.matches ? "dark" : "light");
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);

  return scheme;
}
