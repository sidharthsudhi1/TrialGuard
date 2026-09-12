"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { CriterionRow } from "../../../components/CriterionRow";
import { SyntheticNotice } from "../../../components/SyntheticNotice";
import { TrialVerdictBadge } from "../../../components/VerdictBadge";
import { assessStreamUrl } from "../../../lib/api";
import type { CriterionEvent, TrialEvent } from "../../../lib/types";

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

  useEffect(() => {
    if (!jobId) return;
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
    const onSummary = () => {
      setStatus("Complete.");
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
      <p className="status-line">{status}</p>
      {error && <p className="error">{error}</p>}
      <div className="assessment-list">
        {events.map((ev) => (
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
