import type { IpoStatus } from "@/lib/api";

type IconName = "check" | "warning" | "error" | "spinner" | "circle" | "chevron" | "up" | "down" | "search" | "upload" | "link";

const PATHS: Record<Exclude<IconName, "spinner">, string> = {
  check: "M5 12.5l4.5 4.5L19 7.5",
  warning: "M12 4l9 16H3L12 4zm0 6v4m0 3v.01",
  error: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zm-3 6l6 6m0-6l-6 6",
  circle: "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16z",
  chevron: "M8 10l4 4 4-4",
  up: "M12 19V5m-6 6l6-6 6 6",
  down: "M12 5v14m6-6l-6 6-6-6",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zm9 16l-4-4",
  upload: "M12 16V4m-5 5l5-5 5 5M5 20h14",
  link: "M10 14l4-4m-5.5-1.5L10 7a4 4 0 0 1 6 6l-1.5 1.5m-5 0L8 16a4 4 0 0 1-6-6l1.5-1.5",
};

export function Icon({ name, className = "h-4 w-4" }: { name: IconName; className?: string }) {
  if (name === "spinner")
    return (
      <svg viewBox="0 0 24 24" className={`spin ${className}`} aria-hidden fill="none">
        <circle cx="12" cy="12" r="8" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2.5" />
        <path d="M20 12a8 8 0 0 0-8-8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    );
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <path d={PATHS[name]} />
    </svg>
  );
}

/** A finance term with a plain-language explanation on hover, focus or tap. */
export function Term({ children, tip }: { children: React.ReactNode; tip: string }) {
  return (
    <span className="group relative inline-block">
      <button type="button" className="cursor-help border-b border-dotted border-ink-3 text-inherit"
        aria-label={`${typeof children === "string" ? children : "Term"}: ${tip}`}>
        {children}
      </button>
      <span role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 hidden w-64 -translate-x-1/2 rounded-md border border-line bg-surface p-2 text-left text-xs font-normal leading-snug text-ink-2 shadow-sm group-focus-within:block group-hover:block">
        {tip}
      </span>
    </span>
  );
}

export const GLOSSARY = {
  pop: "The share price closes above the offer price on its first day of trading.",
  offer: "The price at which the company sells its new shares to investors, set the night before trading starts.",
  range: "The price range the company first proposes in its filings. Pricing above it usually signals strong demand.",
  prospectus: "The document a company files with the SEC before going public, describing the business, risks and deal.",
  underwriter: "The investment banks that sell the shares. Prestige is rated 0–9 on the Carter-Manaster scale.",
  vix: "A measure of expected stock-market volatility. Higher means a more nervous market.",
  bucket: "First-day return groups: overpriced (fell), fair (0–10%), underpriced (10–30%), heavily underpriced (>30%).",
};

const STATUS_STYLE: Record<IpoStatus, { bg: string; ink: string }> = {
  priced: { bg: "var(--success-bg)", ink: "var(--success-ink)" },
  registration: { bg: "var(--info-bg)", ink: "var(--info-ink)" },
  withdrawn: { bg: "var(--warning-bg)", ink: "var(--warning-ink)" },
  public: { bg: "var(--surface-2)", ink: "var(--text-2)" },
  none: { bg: "var(--surface-2)", ink: "var(--text-2)" },
};

export function StatusBadge({ status, label }: { status: IpoStatus; label: string }) {
  const s = STATUS_STYLE[status];
  return (
    <span className="inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ background: s.bg, color: s.ink }}>
      {label}
    </span>
  );
}
