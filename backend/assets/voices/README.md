# Voice reference clips

S2 Pro clones a voice from a short reference recording — each preset in
`../voices.json` points at one WAV file in this folder.

## Adding a voice

1. Drop a WAV file here: **10–30 seconds**, one speaker, no music or background
   noise, natural speaking pace. 44.1 kHz mono 16-bit is ideal (other PCM WAV
   formats work too). No transcript is needed.
2. Add an entry to `../voices.json`:

   ```json
   {
     "id": "hu-female-anna",
     "name": "Anna (Hungarian, female)",
     "language": "hu",
     "ref_audio": "voices/hu-female-anna.wav"
   }
   ```

3. Restart the backend. The registry logs every voice it loads; broken entries
   (missing file, bad JSON) are skipped with a warning instead of crashing.

The cloned voice inherits timbre, style, and accent from the clip, so record in
the language you want the output in. Only use voices you have the right to use.
See [LANGUAGES.md](LANGUAGES.md) for adding your voice in another language.

## Bundled defaults

`en-male-david.wav` and `en-female-zira.wav` were synthesized with the Windows
speech engine so the app works out of the box. They sound robotic — replace
them with real recordings for production-quality results.
