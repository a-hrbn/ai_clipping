"use client";

import { useState, FormEvent, useRef } from "react";

interface Props {
  onSubmitUrl: (url: string, profile: string | null) => void;
  onSubmitFile: (file: File, profile: string | null) => void;
  submitting: boolean;
  uploading: boolean;
}

const ALLOWED_TYPES = new Set([
  "video/mp4",
  "video/x-matroska",
  "video/quicktime",
  "video/x-msvideo",
  "video/webm",
  "video/mp2t",
  "video/x-flv",
]);

const MAX_SIZE_GB = 4;

// Streamers with a trained ML model.
// Add new entries here once a model has been trained via:
//   python -m ml.auto_train_short_form --per-streamer
const STREAMERS: { value: string; label: string }[] = [
  { value: "jynxzi", label: "Jynxzi" },
];

function validateFile(f: File): string {
  if (f.size > MAX_SIZE_GB * 1_000_000_000) {
    return `Datei zu groß — Maximum ist ${MAX_SIZE_GB} GB`;
  }
  if (!ALLOWED_TYPES.has(f.type) && f.type !== "") {
    return `Format nicht unterstützt: ${f.type}`;
  }
  return "";
}

export default function UrlInput({ onSubmitUrl, onSubmitFile, submitting, uploading }: Props) {
  const [tab, setTab] = useState<"url" | "file">("url");
  const [url, setUrl] = useState("");
  const [urlError, setUrlError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleUrlSubmit(e: FormEvent) {
    e.preventDefault();
    setUrlError("");
    if (!url.match(/https?:\/\/(www\.)?twitch\.tv\/videos\/\d+/)) {
      setUrlError("Gültige Twitch VOD URL erforderlich (https://www.twitch.tv/videos/…)");
      return;
    }
    onSubmitUrl(url, selectedProfile);
  }

  function handleFileSubmit(e: FormEvent) {
    e.preventDefault();
    setFileError("");
    if (!file) {
      setFileError("Bitte eine Videodatei auswählen");
      return;
    }
    onSubmitFile(file, selectedProfile);
  }

  function applyFile(f: File) {
    const err = validateFile(f);
    if (err) { setFileError(err); return; }
    setFile(f);
    setFileError("");
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    if (f) applyFile(f);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    if (uploading) return;
    const f = e.dataTransfer.files?.[0] ?? null;
    if (f) applyFile(f);
  }

  return (
    <div className="space-y-6">

      {/* Streamer selector */}
      <div className="space-y-3">
        <label className="block text-xs font-medium uppercase tracking-widest text-muted">
          Streamer
        </label>
        <div className="flex flex-wrap gap-2">
          {/* No streamer — server default */}
          <ProfilePill
            label="Allgemein"
            selected={selectedProfile === null}
            disabled={false}
            onClick={() => setSelectedProfile(null)}
            isStreamer={false}
          />
          {STREAMERS.map((s) => (
            <ProfilePill
              key={s.value}
              label={s.label}
              selected={selectedProfile === s.value}
              disabled={false}
              onClick={() => setSelectedProfile(s.value)}
              isStreamer
            />
          ))}
        </div>
      </div>

      <div className="border-t border-border" />

      {/* Tab bar */}
      <div className="flex gap-0 border-b border-border">
        {(["url", "file"] as const).map((t) => (
          <button
            key={t}
            type="button"
            disabled={false}
            onClick={() => setTab(t)}
            className={[
              "relative px-0 mr-8 pb-3 text-sm font-semibold uppercase tracking-widest transition-colors cursor-pointer",
              tab === t
                ? "text-accent"
                : "text-muted hover:text-dim",
            ].join(" ")}
          >
            {t === "url" ? "Twitch URL" : "Lokale Datei"}
            {tab === t && (
              <span className="absolute bottom-0 left-0 right-0 h-px bg-accent" />
            )}
          </button>
        ))}
      </div>

      {/* URL tab */}
      {tab === "url" && (
        <form onSubmit={handleUrlSubmit} className="space-y-5">
          <div className="space-y-2">
            <label className="block text-xs font-medium uppercase tracking-widest text-muted">
              VOD-Link
            </label>
            <div className="flex gap-3">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://www.twitch.tv/videos/2345678901"
                disabled={submitting || uploading}
                className={[
                  "flex-1 bg-surface border px-4 py-3 text-sm text-tx placeholder-muted",
                  "focus:outline-none transition-colors",
                  urlError
                    ? "border-danger/50 focus:border-danger"
                    : "border-border focus:border-border-bright",
                  submitting || uploading ? "opacity-50 cursor-not-allowed" : "",
                ].join(" ")}
              />
              <button
                type="submit"
                disabled={submitting || uploading || !url}
                className={[
                  "shrink-0 px-6 py-3 text-sm font-bold uppercase tracking-widest transition-all",
                  "bg-accent text-bg hover:brightness-110 active:brightness-95",
                  "disabled:opacity-30 disabled:cursor-not-allowed",
                ].join(" ")}
              >
                {submitting || uploading ? (
                  <span className="flex items-center gap-2">
                    <span className="inline-block w-3 h-3 border border-bg border-t-transparent rounded-full animate-spin-slow" />
                    Warte
                  </span>
                ) : (
                  "Analysieren"
                )}
              </button>
            </div>
            {urlError && (
              <p className="text-xs text-danger mt-1">{urlError}</p>
            )}
          </div>
        </form>
      )}

      {/* File tab */}
      {tab === "file" && (
        <form onSubmit={handleFileSubmit} className="space-y-5">
          <div className="space-y-2">
            <label className="block text-xs font-medium uppercase tracking-widest text-muted">
              Videodatei
            </label>

            {/* Drop zone with film-frame corners */}
            <div
              role="button"
              tabIndex={0}
              onClick={() => !uploading && fileInputRef.current?.click()}
              onKeyDown={(e) => e.key === "Enter" && !uploading && fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); if (!uploading) setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={[
                "relative flex flex-col items-center justify-center gap-3 px-6 py-10",
                "cursor-pointer select-none transition-all duration-200",
                "bg-surface border",
                dragging
                  ? "border-accent bg-accent-dim"
                  : file
                  ? "border-accent/40 bg-accent-dim/50"
                  : "border-border hover:border-border-bright",
                uploading ? "opacity-50 cursor-not-allowed pointer-events-none" : "",
              ].join(" ")}
            >
              {/* Corner marks */}
              {["top-0 left-0 border-t border-l", "top-0 right-0 border-t border-r", "bottom-0 left-0 border-b border-l", "bottom-0 right-0 border-b border-r"].map((pos, i) => (
                <span
                  key={i}
                  className={`absolute w-3 h-3 ${pos} ${dragging || file ? "border-accent" : "border-border-bright"} transition-colors`}
                />
              ))}

              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
                disabled={uploading}
              />

              {file ? (
                <>
                  {/* File icon */}
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-accent">
                    <path d="M4 3h8l4 4v10a1 1 0 01-1 1H4a1 1 0 01-1-1V4a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2" fill="none"/>
                    <path d="M12 3v4h4" stroke="currentColor" strokeWidth="1.2"/>
                    <path d="M7 10h6M7 13h4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                  <div className="text-center">
                    <p className="text-sm font-semibold text-tx">{file.name}</p>
                    <p className="mt-0.5 text-xs text-muted font-mono">
                      {(file.size / 1_000_000).toFixed(1)} MB — klicken zum Ändern
                    </p>
                  </div>
                </>
              ) : (
                <>
                  {/* Upload icon */}
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-muted">
                    <path d="M10 13V5M6 9l4-4 4 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M3 15h14" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                  <div className="text-center">
                    <p className="text-sm text-dim">Datei hier ablegen oder klicken</p>
                    <p className="mt-0.5 text-xs text-muted font-mono">MP4 · MKV · MOV · AVI · WEBM — max {MAX_SIZE_GB} GB</p>
                  </div>
                </>
              )}
            </div>

            {fileError && (
              <p className="text-xs text-danger">{fileError}</p>
            )}
          </div>

          <button
            type="submit"
            disabled={uploading || !file}
            className={[
              "w-full py-3 text-sm font-bold uppercase tracking-widest transition-all",
              "bg-accent text-bg hover:brightness-110 active:brightness-95",
              "disabled:opacity-30 disabled:cursor-not-allowed",
            ].join(" ")}
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="inline-block w-3 h-3 border border-bg border-t-transparent rounded-full animate-spin-slow" />
                Wird hochgeladen…
              </span>
            ) : (
              "Analysieren"
            )}
          </button>
        </form>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProfilePill — individual selectable option
// ---------------------------------------------------------------------------

interface PillProps {
  label: string;
  selected: boolean;
  disabled: boolean;
  onClick: () => void;
  isStreamer: boolean;
}

function ProfilePill({ label, selected, disabled, onClick, isStreamer }: PillProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={[
        "px-3 py-1.5 text-xs font-semibold uppercase tracking-widest transition-all",
        "border focus:outline-none disabled:pointer-events-none disabled:opacity-40",
        selected
          ? isStreamer
            ? "border-accent bg-accent text-bg"
            : "border-accent/60 bg-accent-dim text-accent"
          : "border-border text-muted hover:border-border-bright hover:text-dim",
      ].join(" ")}
    >
      {isStreamer && (
        <span
          className={[
            "mr-1.5 inline-block w-1 h-1 rounded-full align-middle",
            selected ? "bg-bg" : "bg-accent",
          ].join(" ")}
        />
      )}
      {label}
    </button>
  );
}
