"use client";

import { useState } from "react";
import { BookOpenIcon, CheckIcon, CopyIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

const AI_PROMPT = `Write a voice-over narration script about [TOPIC], about [DURATION] long when read aloud at a natural pace (~150 words per minute).

Rules for the script:
- Plain spoken text only — no headings, no stage directions outside tags, no markdown, no emojis.
- Short sentences. One idea per sentence. Conversational, like talking to one person.
- Write out every number, unit, date and abbreviation the way it should be spoken ("twenty five percent", not "25%").
- End sentences with normal punctuation. Use commas for short pauses and "..." for a hesitation.
- Direct the delivery with inline square-bracket voice tags; each tag affects the speech after it. Available: [short pause], [long pause], [breath], [sigh], [emphasis], [excited], [slightly excited], [sad], [surprised], [calm], [whisper], [low voice], [volume up], [laughing], [chuckle], or a short free-form description like [professional broadcast tone].
- Use [pause] for a beat of exact silence, or [pause:2] for two seconds.
- Set the overall tone with one tag at the very start, then use at most 2–3 tags per paragraph. Never put a tag mid-word, and never put a colon inside a tag other than [pause:N].
- No intro like "Here's the script" — output the narration text only.`;

type Mark = { mark: string; effect: string };

/** Silence the app inserts itself: [pause] markers are cut out of the text and
 * replaced with exact silence, so they always work. */
const PACING_MARKS: Mark[] = [
  { mark: "[pause]", effect: "exact beat of silence (0.6 s) — never spoken" },
  { mark: "[pause:2]", effect: "exactly that many seconds of silence (up to 10)" },
  { mark: ",", effect: "short pause inside a sentence" },
  { mark: ". ! ?", effect: "ends the phrase with matching intonation" },
  { mark: "…", effect: "hesitation / trailing off" },
];

/** Tags interpreted by the S2 Pro voice model. Open vocabulary: these are
 * common examples, not an exhaustive list. */
const TAG_GROUPS: { title: string; marks: Mark[] }[] = [
  {
    title: "Breath & natural pauses",
    marks: [
      { mark: "[short pause]", effect: "a natural, brief breath-pause" },
      { mark: "[long pause]", effect: "a longer, dramatic pause" },
      { mark: "[breath]", effect: "an audible breath" },
      { mark: "[inhale]", effect: "a sharp intake of breath" },
      { mark: "[sigh]", effect: "a sigh before the next words" },
    ],
  },
  {
    title: "Emotion",
    marks: [
      { mark: "[excited]", effect: "brighter, more energetic delivery" },
      { mark: "[sad]", effect: "slower, lower, softer delivery" },
      { mark: "[angry]", effect: "harder, more forceful delivery" },
      { mark: "[surprised]", effect: "raised, reactive intonation" },
      { mark: "[calm]", effect: "relaxed, even delivery" },
      { mark: "[slightly …] / [very …]", effect: "dial intensity, e.g. [slightly sad], [very excited]" },
    ],
  },
  {
    title: "Delivery",
    marks: [
      { mark: "[emphasis]", effect: "stresses the words that follow" },
      { mark: "[whisper]", effect: "hushed, whispered voice" },
      { mark: "[low voice]", effect: "quieter, lower register" },
      { mark: "[volume up]", effect: "louder, projected voice" },
      { mark: "[laughing]", effect: "speaks through a laugh" },
      { mark: "[chuckle]", effect: "a short chuckle" },
    ],
  },
  {
    title: "Free-form",
    marks: [
      { mark: "[professional broadcast tone]", effect: "describe any style in a few words" },
      { mark: "[speaking slowly and warmly]", effect: "pace + warmth in one tag" },
      { mark: "[pitch up]", effect: "higher pitch for the following text" },
    ],
  },
];

function MarkTable({ rows }: { rows: Mark[] }) {
  return (
    <div className="overflow-hidden rounded-xl border">
      {rows.map((row) => (
        <div
          key={row.mark}
          className="flex items-start gap-4 border-b px-4 py-2.5 text-sm last:border-b-0"
        >
          <code className="w-44 shrink-0 font-mono text-xs text-primary">{row.mark}</code>
          <span className="text-foreground/85">{row.effect}</span>
        </div>
      ))}
    </div>
  );
}

export function VoiceGuideDialog() {
  const [copied, setCopied] = useState(false);

  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(AI_PROMPT);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard unavailable (http) — the user can still select the text
    }
  };

  return (
    <Dialog>
      <DialogTrigger
        render={
          <Button variant="outline" size="sm">
            <BookOpenIcon className="size-3.5" />
            How to write your script
          </Button>
        }
      />
      <DialogContent className="max-w-xl">
        <div className="max-h-[75vh] space-y-6 overflow-y-auto pr-2">
          <div className="pr-6">
            <DialogTitle>Writing a great narration script</DialogTitle>
            <p className="mt-1 text-sm text-muted-foreground">
              Punctuation sets the pacing; square-bracket tags direct the
              emotion and delivery.
            </p>
          </div>

          <section className="space-y-2">
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Pacing marks
            </h3>
            <MarkTable rows={PACING_MARKS} />
          </section>

          <section className="space-y-3">
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Voice tags <span className="normal-case">· interpreted by the model</span>
            </h3>
            <p className="text-xs text-muted-foreground">
              Put a tag in <strong>square brackets</strong> right before the text it should
              color — it lasts until the delivery changes again. The vocabulary is open: any
              short description works, these are just reliable starting points. Use 2–3 per
              paragraph at most; results vary by voice, so test a short line first.
            </p>
            {TAG_GROUPS.map((group) => (
              <div key={group.title} className="space-y-1.5">
                <p className="text-xs font-medium text-foreground/70">{group.title}</p>
                <MarkTable rows={group.marks} />
              </div>
            ))}
            <div className="rounded-xl bg-secondary/50 p-3 font-mono text-xs leading-relaxed text-foreground/85">
              [professional broadcast tone] The future is closer than ever. [short pause] Every
              breakthrough opens new possibilities… [emphasis] and what we build today will
              shape the next generation. [pause:1] [slightly excited] So let&apos;s begin.
            </div>
          </section>

          <section className="space-y-2">
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Tips that make the biggest difference
            </h3>
            <ul className="list-disc space-y-1.5 pl-5 text-sm text-foreground/85">
              <li>Write out numbers, dates and units as words — the voice reads literally.</li>
              <li>Short sentences sound natural; very long ones rush.</li>
              <li>Read it aloud once yourself — if you stumble, the voice will too.</li>
              <li>One minute of narration is roughly 150 words (900 characters).</li>
            </ul>
          </section>

          <section className="space-y-2">
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Let an AI write it for you
            </h3>
            <p className="text-xs text-muted-foreground">
              Paste this into ChatGPT or Claude, fill in the topic and duration,
              then paste the result into the script editor.
            </p>
            <div className="relative rounded-xl bg-secondary/50 p-4">
              <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground/85">
                {AI_PROMPT}
              </pre>
              <Button
                size="sm"
                variant="secondary"
                onClick={copyPrompt}
                className="absolute right-3 top-3"
              >
                {copied ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
