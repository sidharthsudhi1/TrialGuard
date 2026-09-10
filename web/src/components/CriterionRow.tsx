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
      ) : quote && a.grounded && a.weak_absence ? (
        // The verdict claims the patient does NOT match this disqualifier, and a
        // verbatim quote cannot establish absence. It is still a real quote, so
        // it is shown -- but the badge must not vouch for what it does not prove.
        <p className="quote-weak">
          Quote verified, but it does not establish absence:{" "}
          <em>&ldquo;{quote}&rdquo;</em>
          <br />
          <span className="muted">
            This criterion asks whether something is absent. Read the quote
            yourself before relying on the verdict.
          </span>
        </p>
      ) : quote && a.grounded ? (
        <p className="quote-ok">
          Grounded citation: <em>&ldquo;{quote}&rdquo;</em>
        </p>
      ) : null}
    </li>
  );
}
