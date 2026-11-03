# Adding a new language for your voice

S2 copies **everything** in the reference clip: your timbre, and also your accent
and pronunciation. An English recording reading Ukrainian text gives you
Ukrainian with an English accent. To sound natural in a language, give it a
reference clip **in that language**.

## Option 1: you speak the language (best)

1. Record **15–30 s** of yourself speaking it naturally. Use a quiet room, no music
   and one speaker.
2. Save it here as a WAV, e.g. `tim_uk.wav`.
3. Write down **exactly** what you said. An accurate transcript improves the
   clone (it's optional, but a wrong transcript hurts more than none).
4. Add an entry to `../voices.json`:

   ```json
   {
     "id": "tim-uk",
     "name": "Tim (Ukrainian)",
     "language": "uk",
     "ref_audio": "voices/tim_uk.wav",
     "ref_text": "<exact transcript, in that language's script>"
   }
   ```

5. Copy the files to the server and run `docker compose restart backend`
   (assets are a mounted volume, no rebuild needed).
6. Pick the new voice whenever the script is in that language.

## Option 2: you don't speak it

Try these in order:

- **Borrow a related language.** Use a reference in a close language you *do*
  speak, e.g. the Ukrainian clip for Polish text. You get a mild accent but it's
  still your real voice. No extra work.
- **Read it phonetically.** Ask an AI for ~30 s of simple text, practise, and
  record yourself reading it, then follow Option 1 using that text as `ref_text`.
  Works well for languages written the way they sound (Polish, Czech, Spanish…).
  The model copies your mispronunciations too, so take a few tries.
- **Voice conversion (not built yet).** Generate the speech from a native
  speaker's reference, then convert the timbre to yours with Seed-VC or RVC.
  You get native pronunciation in your voice. Needs a new pipeline step.

## Tips

- Expect occasional wrong word stress in lower-resource languages (Ukrainian,
  Polish). Rephrasing or adding punctuation usually fixes a stubborn word.
- Keep one voice entry per language (`tim-en`, `tim-uk`, `tim-pl`) and pick
  the one that matches the script.
