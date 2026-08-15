import { CriterionVerdictBadge } from "./VerdictBadge";
import type { Assessment } from "../lib/types";

export function CriterionRow({ a }: { a: Assessment }) {
  const quote = (a.quote || "").trim();
  return (
    <li className={`criterion ${a.verdict === "unverifiable" ? "criterion-unv" : ""}`}>
      <div className="criterion-head">
        <CriterionVerdictBadge verdict={a.verdict} />
        <span className="kind">[{a.kind || "inclusion"}]</span>
        <span className="criterion-text">{a.criterion}</span>
      </div>
      {a.grounding_failure ? (
        <p className="quote-fail">
          Quote not verbatim in source — downgraded to unverifiable, never forced.
        </p>
      ) : quote && a.grounded ? (
        <p className="quote-ok">
          Grounded citation: <em>&ldquo;{quote}&rdquo;</em>
        </p>
      ) : null}
    </li>
  );
}
