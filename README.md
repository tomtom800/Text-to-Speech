# TTS Podcast

Personal podcast feed generated from Readwise Reader queue tags.

## Queue tags

- `tts-en`
- `tts-de`

Items tagged with both are skipped as ambiguous.

## Architecture

- Source: Readwise Reader API
- TTS: OpenAI (`gpt-4o-mini-tts`, voice `fable`)
- Automation: GitHub Actions (hourly + manual trigger)
- Hosting: GitHub Pages
- State: `state/processed.json`
- Published artifacts: `docs/podcast.xml` and `docs/audio/*.mp3`

## Setup

1. Add secrets in GitHub repository settings:
   - `READWISE_TOKEN`
   - `OPENAI_API_KEY`
2. Enable GitHub Pages using the `docs/` folder on the default branch.
3. Update `PODCAST_BASE_URL` in `.github/workflows/build.yml` to your actual Pages URL.
4. Run the workflow manually once (`workflow_dispatch`).

## Reprocessing a single item

Delete that document entry from `state/processed.json` and rerun the workflow.

## Operational defaults

- `MAX_CHARS_PER_CHUNK=5500`
- `MAX_DOCS_PER_RUN=25`
- `RETAIN_MAX_ITEMS=200`

Set these as workflow env vars if you want different values.

## Notes

- Feed is rebuilt from state on each run to avoid stale/duplicate entries.
- Audio filenames and GUIDs include doc ID + content fingerprint to avoid collisions.
- On content change, prior files for that document are deleted before regeneration.
- Orphan audio files are cleaned automatically.
