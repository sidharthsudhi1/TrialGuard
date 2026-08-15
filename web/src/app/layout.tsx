import type { Metadata } from "next";
import Link from "next/link";
import { BudgetBar } from "../components/BudgetBar";
import "./globals.css";

export const metadata: Metadata = {
  title: "TrialGuard",
  description:
    "Self-verifying clinical-trial eligibility — every verdict cited or flagged unverifiable.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="site-header">
            <div>
              <Link href="/" className="brand">
                TrialGuard
              </Link>
              <p className="tagline">
                Self-verifying eligibility. Every verdict is cited from the trial
                text, or flagged unverifiable — never forced.
              </p>
            </div>
            <div className="header-meta">
              <BudgetBar />
            </div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
