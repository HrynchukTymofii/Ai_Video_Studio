"use client";

import { useEffect, useRef, useState } from "react";

import { GenerationFeed } from "@/components/generation-feed";
import { ComposerControl, Studio, StudioComposer } from "@/components/studio";
import { VoiceGuideDialog } from "@/components/voice-guide-dialog";
import { VoiceSelect } from "@/components/voice-select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { submitTalkingHead } from "@/lib/api";
import { voiceOverCost } from "@/lib/pricing";
import { isTerminal, useJobPolling } from "@/lib/use-job-polling";

// ~900 spoken characters per minute (the backend prices at the same pace).
const CHARS_PER_MINUTE = 900;

function estimateDuration(chars: number): string {
  if (chars === 0) return "0:00";
  const seconds = Math.round((chars / CHARS_PER_MINUTE) * 60);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

// stable reference — an inline arrow would refetch the feed on every render
const onlyVoiceOvers = (job: { voiceOnly?: boolean }) => job.voiceOnly === true;

export default function VoicePage() {
  const [script, setScript] = useState("");
  const [voice, setVoice] = useState<string>("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const status = useJobPolling(jobId);
  // the queue takes more submissions while a job runs — only lock during the POST
  const busy = submitting;
  const cost = voiceOverCost(script);

  const terminalNotified = useRef<string | null>(null);
  useEffect(() => {
    if (jobId && isTerminal(status) && terminalNotified.current !== jobId) {
      terminalNotified.current = jobId;
      setRefreshKey((key) => key + 1);
    }
  }, [jobId, status]);

  const canSubmit = script.trim() !== "" && voice !== "";

  const handleSubmit = async () => {
    if (!canSubmit || busy) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const response = await submitTalkingHead({
        avatar: null,
        script,
        voice,
        voiceOnly: true,
      });
      setJobId(response.jobId);
      setScript("");
      setRefreshKey((key) => key + 1);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Submission failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Studio>
      <GenerationFeed
        kind="talking_head"
        refreshKey={refreshKey}
        status={status}
        activeJobId={jobId}
        // only narration jobs here — lip-sync videos live on the Video tab
        filterJobs={onlyVoiceOvers}
        emptyTitle="Voice Over Studio"
        emptyHint="Write your narration below — an S2 Pro voice clone reads it exactly as written. Punctuation drives the pacing, [pause] inserts real silence, and tags like [excited] or [whisper] direct the delivery."
      />

      <StudioComposer>
        <Textarea
          value={script}
          onChange={(event) => setScript(event.target.value)}
          placeholder="Write your narration… commas make short pauses, [pause:2] holds two seconds of silence"
          className="max-h-64 min-h-20 resize-none border-0 bg-transparent text-[15px] leading-relaxed shadow-none focus-visible:ring-0 dark:bg-transparent"
          disabled={busy}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              void handleSubmit();
            }
          }}
        />
        {submitError && (
          <Alert variant="destructive" className="mb-2">
            <AlertDescription>{submitError}</AlertDescription>
          </Alert>
        )}
        <div className="flex flex-wrap items-center gap-2 border-t pt-2.5">
          <ComposerControl label="Voice">
            <VoiceSelect value={voice} onChange={setVoice} disabled={busy} compact />
          </ComposerControl>
          <VoiceGuideDialog />
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            ≈ {estimateDuration(script.length)}
          </span>
          <Button
            size="lg"
            className="ml-auto rounded-lg font-mono text-xs uppercase tracking-widest"
            onClick={handleSubmit}
            disabled={busy || !canSubmit}
          >
            {busy ? "Synthesizing…" : `Generate · ${cost} cr`}
          </Button>
        </div>
      </StudioComposer>
    </Studio>
  );
}
