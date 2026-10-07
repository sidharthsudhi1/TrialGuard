"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { SyntheticNotice } from "../components/SyntheticNotice";
import { ApiError, fetchLimits, searchTrials, startAssess } from "../lib/api";
import type { Limits, Preset, SearchTrial } from "../lib/types";
import { freshness } from "@/lib/freshness";

// Served by /api/limits rather than hardcoded. A note the UI offers but the
// API's allowlist does not know is uncacheable, and these two were exactly that
// for the life of the demo: every run paid a fresh call per trial. Kept here as
// the fallback only for when limits has not loaded yet.
const FALLBACK_PRESETS: Preset[] = [
  {
    label: "NSCLC stage IV",
    note: "58-year-old woman with stage IV non-small cell lung cancer, ECOG performance status 1, never-smoker, EGFR wild-type. No prior systemic therapy. Adequate organ function.",
  },
];

export default function SearchPage() {
  const router = useRouter();
  const [note, setNote] = useState(FALLBACK_PRESETS[0].note);
  const [trials, setTrials] = useState<SearchTrial[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<"search" | "assess" | null>(null);
  // Seconds left on a rate-limit cooldown. A 429 is not a failure and must not
  // read as one: shown as a bare error it looks like the deploy is broken, which
  // is what makes a user retry into the limit they already hit.
  const [cooldown, setCooldown] = useState(0);
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
  // Whatever the API says it will cache. Falling back only until it answers.
  const presets =
    limits?.presets?.length ? limits.presets : FALLBACK_PRESETS;
  const minutesFor = (n: number) => {
    if (!limits) return 0;
    const workers =
      deep && selected.size > limits.deep_head
        ? limits.deep_job_workers
        : limits.assess_workers;
    return (Math.ceil(n / workers) * limits.seconds_per_trial) / 60;
  };
  const quote =
    limits && selected.size
      ? {
          usd: selected.size * limits.usd_per_trial,
          minutes: minutesFor(selected.size),
          // Progressive: the top-ranked head lands first, the rest behind it.
          headMinutes:
            deep && selected.size > limits.deep_head
              ? minutesFor(limits.deep_head)
              : null,
        }
      : null;
  const fmt = (m: number) =>
    m < 1 ? `${Math.round(m * 60)} s` : `${m.toFixed(1)} min`;

  async function onSearch() {
    setError(null);
    setBusy("search");
    setSelected(new Set());
    try {
      const res = await searchTrials(note, searchTopK, deep);
      setTrials(res.trials);
      // Deep is a progressive run over the whole ranked pool, so it starts with
      // everything selected; the user can still drop trials before starting.
      if (deep) setSelected(new Set(res.trials.map((t) => t.nct_id)));
      setLatency(
        res.latency_ms?.total_ms != null
          ? `${res.latency_ms.total_ms.toFixed(0)} ms total`
          : null
      );
      if (!res.trials.length) setError("No candidate trials returned for this note.");
    } catch (e) {
      setTrials([]);
      setError(e instanceof Error ? e.message : String(e));
      setCooldown(e instanceof ApiError && e.isRateLimited ? e.retryAfter ?? 60 : 0);
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

  useEffect(() => {
    if (cooldown <= 0) return;
    const id = setTimeout(() => {
      setCooldown((s) => {
        if (s <= 1) setError(null);
        return s - 1;
      });
    }, 1000);
    return () => clearTimeout(id);
  }, [cooldown]);

  async function onAssess() {
    if (!selected.size) return;
    setError(null);
    setCooldown(0);
    setBusy("assess");
    try {
      // Rank order, not click order: the API assesses in the order sent, and a
      // progressive deep job only means "top N first" if the top N go first.
      const ids = trials.map((t) => t.nct_id).filter((n) => selected.has(n));
      const { job_id } = await startAssess(note, ids, deep);
      sessionStorage.setItem(
        `tg-job-${job_id}`,
        JSON.stringify({ note, nct_ids: ids })
      );
      router.push(`/assess/${job_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setCooldown(e instanceof ApiError && e.isRateLimited ? e.retryAfter ?? 60 : 0);
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
          {presets.map((p) => (
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
            Deep search — rank up to {limits.max_assess_trials_deep} candidates
            instead of 5 and assess them in rank order. The top{" "}
            {limits.deep_head} arrive in about the usual time; the rest keep
            streaming in behind them. Assessing 100 instead of 10 surfaced about
            6x more eligible trials on TREC 2021/2022, and costs proportionally
            more time and money.
          </label>
        )}
        {error && (
          <p className={cooldown > 0 ? "muted" : "error"}>
            {cooldown > 0
              ? `Too many assessments started in the last minute. Try again in ${cooldown}s.`
              : error}
          </p>
        )}
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
              {quote.usd.toFixed(4)}, about {fmt(quote.minutes)}
              {quote.headMinutes != null && limits
                ? ` in all; the top ${limits.deep_head} in about ${fmt(quote.headMinutes)}`
                : ""}
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
