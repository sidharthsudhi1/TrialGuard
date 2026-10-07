"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { CriterionRow } from "../../../components/CriterionRow";
import { SyntheticNotice } from "../../../components/SyntheticNotice";
import { TrialVerdictBadge } from "../../../components/VerdictBadge";
import { assessStreamUrl } from "../../../lib/api";
import type {
  CriterionEvent,
  HeadEvent,
  SummaryEvent,
  TrialEvent,
} from "../../../lib/types";

const TIER_ORDER = { eligible: 0, needs_review: 1, excluded: 2 } as const;

// Tiered contract: eligible, then needs_review by fewest unstated facts, then
// excluded; rank breaks ties. With 100 trials landing in completion order,
// append order would bury an early eligible trial under later exclusions.
function byTier(rank: Map<string, number>) {
  return (a: TrialEvent, b: TrialEvent) =>
    TIER_ORDER[a.trial_tier ?? "needs_review"] -
      TIER_ORDER[b.trial_tier ?? "needs_review"] ||
    (a.n_unknown ?? 0) - (b.n_unknown ?? 0) ||
    (rank.get(a.nct_id) ?? 1e9) - (rank.get(b.nct_id) ?? 1e9);
}

export default function AssessPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;
  const [events, setEvents] = useState<TrialEvent[]>([]);
  // Criteria streamed for trials that have not finished yet, keyed by nct_id.
  // Dropped the moment that trial's authoritative event lands.
  const [pending, setPending] = useState<Record<string, CriterionEvent[]>>({});
  const [status, setStatus] = useState("Connecting…");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [head, setHead] = useState<HeadEvent | null>(null);
  const [total, setTotal] = useState<number | null>(null);
  const [rank, setRank] = useState<Map<string, number>>(new Map());
  const [stopped, setStopped] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    try {
      const saved = JSON.parse(sessionStorage.getItem(`tg-job-${jobId}`) || "null");
      if (saved?.nct_ids) {
        setTotal(saved.nct_ids.length);
        setRank(new Map(saved.nct_ids.map((n: string, i: number) => [n, i])));
      }
    } catch {
      /* no saved job: progress shows without a total */
    }
    const url = assessStreamUrl(jobId);
    const es = new EventSource(url);
    setStatus("Assessing trials…");

    const collected: TrialEvent[] = [];

    const onCriterion = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as CriterionEvent;
        setPending((prev) => ({
          ...prev,
          [data.nct_id]: [...(prev[data.nct_id] || []), data],
        }));
      } catch {
        /* ignore malformed */
      }
    };
    const onTrial = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as TrialEvent;
        collected.push(data);
        setEvents([...collected]);
        // The graded result supersedes everything streamed for this trial.
        setPending((prev) => {
          const next = { ...prev };
          delete next[data.nct_id];
          return next;
        });
        sessionStorage.setItem(`tg-job-results-${jobId}`, JSON.stringify(collected));
      } catch {
        /* ignore malformed */
      }
    };
    const onHead = (ev: MessageEvent) => {
      try {
        setHead(JSON.parse(ev.data) as HeadEvent);
      } catch {
        /* ignore malformed */
      }
    };
    const onSummary = (ev: MessageEvent) => {
      let partial = false;
      try {
        const data = JSON.parse(ev.data) as SummaryEvent;
        if (data.status === "partial") {
          partial = true;
          setStopped(
            `Stopped after ${data.n_assessed} of ${data.n_trials} trials: ` +
              (data.stopped?.message || "the daily budget ran out") +
              ". The results below are complete for the trials shown."
          );
        }
      } catch {
        /* older servers send no body */
      }
      setStatus(partial ? "Stopped early." : "Complete.");
      setDone(true);
      es.close();
    };
    const onJobError = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data);
        setError(
          typeof data.error === "string" ? data.error : JSON.stringify(data)
        );
      } catch {
        setError("Assessment failed.");
      }
      setDone(true);
      es.close();
    };

    es.addEventListener("criterion", onCriterion);
    es.addEventListener("trial", onTrial);
    es.addEventListener("head", onHead);
    es.addEventListener("summary", onSummary);
    // Application terminal errors use event name "error" with a data payload.
    es.addEventListener("error", (ev) => {
      if (ev instanceof MessageEvent && typeof ev.data === "string" && ev.data) {
        onJobError(ev);
      }
    });

    return () => {
      es.close();
    };
  }, [jobId]);

  return (
    <main>
      <SyntheticNotice />
      <p className="status-line">
        {status}
        {total != null && !done ? ` ${events.length} of ${total} assessed.` : ""}
      </p>
      {head && !done && (
        <p className="muted">
          Top {head.n} ranked trials assessed. Still assessing the remaining{" "}
          {head.of - head.n}; they slot into the list below as they finish.
        </p>
      )}
      {stopped && <p className="muted">{stopped}</p>}
      {error && <p className="error">{error}</p>}
      <div className="assessment-list">
        {[...events].sort(byTier(rank)).map((ev) => (
          <article key={ev.nct_id} className="assessment-block">
            <div className="assessment-head">
              <h2>
                <Link href={`/trials/${ev.nct_id}?job=${jobId}`}>{ev.nct_id}</Link>
              </h2>
              <TrialVerdictBadge verdict={ev.trial_verdict} />
              {ev.title && <span className="muted">{ev.title}</span>}
            </div>
            {ev.error && <p className="error">{ev.error}</p>}
            {ev.criteria_truncated && (
              <p className="muted">
                {ev.truncated_block
                  ? "Every criterion assessed here passed, but this trial has more than the cap allows. Eligible is withheld because it is a claim about all criteria, and some were never assessed."
                  : "Criteria list truncated at cap — some criteria were not assessed."}
              </p>
            )}
            <ul className="criterion-list">
              {(ev.assessments || []).map((a, i) => (
                <CriterionRow key={`${ev.nct_id}-${i}`} a={a} />
              ))}
            </ul>
            <p style={{ marginTop: "0.75rem" }}>
              <Link href={`/trials/${ev.nct_id}?job=${jobId}`}>
                View quote in eligibility source →
              </Link>
            </p>
          </article>
        ))}
      </div>
      {Object.entries(pending).map(([nctId, criteria]) => (
        <article key={`pending-${nctId}`} className="assessment-block">
          <div className="assessment-head">
            <h2>{nctId}</h2>
            <span className="muted">assessing…</span>
          </div>
          <ul className="criterion-list">
            {criteria.map((c, i) => (
              <li key={`${nctId}-p-${i}`} className="criterion">
                <div className="criterion-head">
                  <span className="muted">{c.verdict}</span>
                  <span className="criterion-text">{c.criterion}</span>
                </div>
              </li>
            ))}
          </ul>
          <p className="muted">
            Provisional — shown as the analyst produces them. Quotes are not
            verified until the trial completes.
          </p>
        </article>
      ))}
      {done && events.length === 0 && !error && (
        <p className="muted">No trial events received.</p>
      )}
      <p style={{ marginTop: "1.25rem" }}>
        <Link href="/">← Back to search</Link>
      </p>
    </main>
  );
}
