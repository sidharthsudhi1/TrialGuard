"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { EligibilityHighlight } from "../../../components/EligibilityHighlight";
import { SyntheticNotice } from "../../../components/SyntheticNotice";
import { fetchTrial } from "../../../lib/api";
import type { Assessment, TrialDetail } from "../../../lib/types";

function TrialDetailInner() {
  const params = useParams<{ nctId: string }>();
  const search = useSearchParams();
  const nctId = params.nctId;
  const jobId = search.get("job");

  const [trial, setTrial] = useState<TrialDetail | null>(null);
  const [quotes, setQuotes] = useState<Assessment[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchTrial(nctId)
      .then((t) => {
        if (!cancelled) setTrial(t);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [nctId]);

  useEffect(() => {
    if (!jobId) return;
    try {
      const raw = sessionStorage.getItem(`tg-assess-${jobId}`);
      if (raw) {
        const all = JSON.parse(raw) as Record<string, Assessment[]>;
        if (all[nctId]) setQuotes(all[nctId]);
      }
    } catch {
      /* ignore */
    }
  }, [jobId, nctId]);

  // Also accept quotes passed via session from the assess page live store.
  useEffect(() => {
    if (!jobId) return;
    try {
      const raw = sessionStorage.getItem(`tg-job-results-${jobId}`);
      if (!raw) return;
      const events = JSON.parse(raw) as {
        nct_id: string;
        assessments: Assessment[];
      }[];
      const match = events.find((e) => e.nct_id === nctId);
      if (match?.assessments) setQuotes(match.assessments);
    } catch {
      /* ignore */
    }
  }, [jobId, nctId]);

  return (
    <main>
      <SyntheticNotice />
      {error && <p className="error">{error}</p>}
      {!trial && !error && <p className="muted">Loading trial…</p>}
      {trial && (
        <section className="panel">
          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.45rem",
              margin: "0 0 0.35rem",
            }}
          >
            {trial.nct_id}
          </h1>
          {trial.title && <p style={{ margin: "0 0 0.5rem" }}>{trial.title}</p>}
          <p className="muted">
            {[trial.status, trial.phase, (trial.conditions || []).slice(0, 3).join(", ")]
              .filter(Boolean)
              .join(" · ")}
          </p>
          <h2
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.1rem",
              margin: "1.25rem 0 0.35rem",
            }}
          >
            Eligibility source
          </h2>
          <p className="muted">
            Assessed quotes are highlighted only when they appear verbatim in this
            text. Ungrounded quotes are not fuzzy-matched.
          </p>
          <EligibilityHighlight source={trial.eligibility_raw || ""} quotes={quotes} />
        </section>
      )}
      <p style={{ marginTop: "1.25rem" }}>
        {jobId ? (
          <Link href={`/assess/${jobId}`}>← Back to assessment</Link>
        ) : (
          <Link href="/">← Back to search</Link>
        )}
      </p>
    </main>
  );
}

export default function TrialDetailPage() {
  return (
    <Suspense fallback={<p className="muted">Loading…</p>}>
      <TrialDetailInner />
    </Suspense>
  );
}
