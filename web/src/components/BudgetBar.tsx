"use client";

import { useEffect, useState } from "react";
import { fetchBudget } from "../lib/api";
import type { BudgetInfo } from "../lib/types";

export function BudgetBar() {
  const [budget, setBudget] = useState<BudgetInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchBudget()
      .then((b) => {
        if (!cancelled) setBudget(b);
      })
      .catch(() => {
        if (!cancelled) setBudget(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!budget) return null;
  const pct = budget.usd_cap > 0 ? Math.min(100, (budget.usd_spent / budget.usd_cap) * 100) : 0;

  return (
    <div className="budget" aria-live="polite">
      <div className="budget-label">
        Daily ledger ${budget.usd_spent.toFixed(3)} / ${budget.usd_cap.toFixed(2)}
        {budget.exhausted ? " — exhausted" : ""}
      </div>
      <div className="budget-track" aria-hidden>
        <div className="budget-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
