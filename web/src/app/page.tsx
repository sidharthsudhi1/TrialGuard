"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { SyntheticNotice } from "../components/SyntheticNotice";
import { fetchLimits, searchTrials, startAssess } from "../lib/api";
import type { Limits, SearchTrial } from "../lib/types";
import { freshness } from "@/lib/freshness";

const PRESETS: { label: string; note: string }[] = [
  {
    label: "NSCLC stage IV",
    note: "58-year-old woman with stage IV non-small cell lung cancer, ECOG performance status 1, never-smoker, EGFR wild-type. No prior systemic therapy. Adequate organ function.",
  },
  {
    label: "Breast cancer adjuvant",
    note: "45-year-old woman with early-stage hormone receptor-positive breast cancer, status post lumpectomy, planning adjuvant endocrine therapy. No metastatic disease. ECOG 0.",
  },
];

export default function SearchPage() {
  const router = useRouter();
  const [note, setNote] = useState(PRESETS[0].note);
  const [trials, setTrials] = useState<SearchTrial[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<"search" | "assess" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<string | null>(null);
  // Depth is opt-in: assessing a wider pool is what raises how many eligible
  // trials surface (measured 6x from top-10 to top-100), and it is also what
  // costs money and minutes. The user makes that trade explicitly.
  const [deep, setDeep] = useState(false);
  const [limits, setLimits] = useState<Limits | null>(null);

  useEffect(() => {
    fetchLimits()
      .then(setLimits)
      .catch(() => setLimits(null));
  }, []);

  const searchTopK = deep && limits ? limits.max_assess_trials_deep : 5;
  const quote =
    limits && selected.size
      ? {
          usd: selected.size * limits.usd_per_trial,
          minutes:
            (Math.ceil(selected.size / limits.assess_workers) *
              limits.seconds_per_trial) /
            60,
        }
      : null;

  async function onSearch() {
    setError(null);
    setBusy("search");
    setSelected(new Set());
    try {
      const res = await searchTrials(note, searchTopK);
      setTrials(res.trials);
      setLatency(
        res.latency_ms?.total_ms != null
          ? `${res.latency_ms.total_ms.toFixed(0)} ms total`
          : null
      );
      if (!res.trials.length) setError("No candidate trials returned for this note.");
    } catch (e) {
      setTrials([]);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  function toggle(nct: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(nct)) next.delete(nct);
      else next.add(nct);
      return next;
    });
  }

  async function onAssess() {
    if (!selected.size) return;
    setError(null);
    setBusy("assess");
    try {
      const ids = [...selected];
      const { job_id } = await startAssess(note, ids, deep);
      sessionStorage.setItem(
        `tg-job-${job_id}`,
        JSON.stringify({ note, nct_ids: ids })
      );
      router.push(`/assess/${job_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(null);
    }
  }

  return (
    <main>
      <SyntheticNotice />
      <section className="panel">
        <label htmlFor="note">Synthetic patient note</label>
        <textarea
          id="note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Paste a synthetic clinical narrative…"
        />
        <div className="actions">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="btn-ghost"
              onClick={() => setNote(p.note)}
            >
              {p.label}
            </button>
          ))}
          <button
            type="button"
            className="btn-primary"
            disabled={busy !== null || !note.trim()}
            onClick={onSearch}
          >
            {busy === "search" ? "Searching…" : "Search trials"}
          </button>
        </div>
        {limits && (
          <label className="muted" style={{ display: "block", marginTop: "0.5rem" }}>
            <input
              type="checkbox"
              checked={deep}
              disabled={busy !== null}
              onChange={(e) => setDeep(e.target.checked)}
            />{" "}
            Deep search — return up to {limits.max_assess_trials_deep} candidates
            instead of 5. A wider assessed pool surfaces roughly 6x more eligible
            trials (measured on TREC 2021/2022), and costs proportionally more
            time and money. You still choose which trials to assess.
          </label>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      {trials.length > 0 && (
        <section className="panel" style={{ marginTop: "1rem" }}>
          <div className="actions" style={{ marginTop: 0, marginBottom: "0.75rem" }}>
            <strong>
              {trials.length} candidate{trials.length === 1 ? "" : "s"}
            </strong>
            {latency && <span className="muted">{latency}</span>}
            <button
              type="button"
              className="btn-primary"
              disabled={busy !== null || selected.size === 0}
              onClick={onAssess}
            >
              {busy === "assess"
                ? "Starting…"
                : `Assess selected (${selected.size})`}
            </button>
          </div>
          {quote && (
            <p className="muted">
              {selected.size} trial{selected.size === 1 ? "" : "s"} ≈ $
              {quote.usd.toFixed(4)}, about{" "}
              {quote.minutes < 1
                ? `${Math.round(quote.minutes * 60)} s`
                : `${quote.minutes.toFixed(1)} min`}
              . Results stream as each trial finishes.
            </p>
          )}
          <div className="results">
            {trials.map((t) => (
              <label key={t.nct_id} className="trial-row">
                <input
                  type="checkbox"
                  checked={selected.has(t.nct_id)}
                  onChange={() => toggle(t.nct_id)}
                />
                <div>
                  <p className="trial-title">
                    <Link href={`/trials/${t.nct_id}`}>{t.nct_id}</Link>
                    {t.title ? ` — ${t.title}` : ""}
                  </p>
                  <p className="trial-meta">
                    {[t.status, t.phase, (t.conditions || []).slice(0, 2).join(", ")]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                  {(() => {
                    const f = freshness(t.last_updated);
                    if (!f) return null;
                    return (
                      <p className={f.stale ? "trial-meta stale" : "trial-meta"}>
                        {f.label}
                      </p>
                    );
                  })()}
                </div>
                <span className="score">{t.score.toFixed(4)}</span>
              </label>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
