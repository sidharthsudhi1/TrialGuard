"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { CriterionRow } from "../../../components/CriterionRow";
import { SyntheticNotice } from "../../../components/SyntheticNotice";
import { TrialVerdictBadge } from "../../../components/VerdictBadge";
import { assessStreamUrl } from "../../../lib/api";
import type { TrialEvent } from "../../../lib/types";

export default function AssessPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;
  const [events, setEvents] = useState<TrialEvent[]>([]);
  const [status, setStatus] = useState("Connecting…");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!jobId) return;
    const url = assessStreamUrl(jobId);
    const es = new EventSource(url);
    setStatus("Assessing trials…");

    const collected: TrialEvent[] = [];

    const onTrial = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as TrialEvent;
        collected.push(data);
        setEvents([...collected]);
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
              <p className="muted">Criteria list truncated at cap — roll-up may be incomplete.</p>
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
      {done && events.length === 0 && !error && (
        <p className="muted">No trial events received.</p>
      )}
      <p style={{ marginTop: "1.25rem" }}>
        <Link href="/">← Back to search</Link>
      </p>
    </main>
  );
}
