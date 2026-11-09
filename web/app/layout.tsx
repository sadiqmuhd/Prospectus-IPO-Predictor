import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Prospectus Bot",
  description: "Reads an IPO prospectus and predicts its first-day return. A data science course project.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <header className="border-b border-line">
          <nav className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
            <Link href="/" className="font-semibold tracking-tight">
              Prospectus Bot
            </Link>
            <div className="flex gap-5 text-sm text-ink-2">
              <Link href="/" className="hover:text-ink">Analyze</Link>
              <Link href="/about" className="hover:text-ink">About the model</Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-5xl px-4 pb-10 text-xs text-ink-3">
          Research demo from a data science course project on US and Nigerian IPOs. Not investment advice.
        </footer>
      </body>
    </html>
  );
}
