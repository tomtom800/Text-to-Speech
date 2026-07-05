import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest import mock


os.environ.setdefault("QUICK_READS_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_TTS_API_KEY", "test-key")

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "build_tts_podcast.py"
SPEC = importlib.util.spec_from_file_location("build_tts_podcast", SCRIPT_PATH)
tts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tts)


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.reason = "OK"
        self.text = ""

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise tts.requests.HTTPError(response=self)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class QuickReadsApiTests(unittest.TestCase):
    def test_headers_use_bearer_authentication(self):
        self.assertEqual(
            tts.quick_reads_headers(),
            {"Authorization": "Bearer test-key"},
        )

    def test_list_paginates_and_deduplicates_articles(self):
        first = [{"id": "a", "tags": []}, {"id": "b", "tags": []}]
        second = [{"id": "b", "tags": []}, {"id": "c", "tags": []}]
        session = FakeSession([FakeResponse(first), FakeResponse(second), FakeResponse([])])

        articles = tts.fetch_article_summaries(session, page_size=2)

        self.assertEqual([article["id"] for article in articles], ["a", "b", "c"])
        self.assertEqual([call["params"]["offset"] for call in session.calls], [0, 2, 4])
        self.assertTrue(all(call["params"]["archived"] == "false" for call in session.calls))

    def test_full_article_preserves_summary_tags(self):
        summary = {"id": "article-1", "title": "Old", "tags": [{"name": "tts-en"}]}
        detail = {"id": "article-1", "title": "New", "content": "<p>Body</p>"}
        session = FakeSession([FakeResponse(detail)])

        article = tts.fetch_article(session, summary)

        self.assertEqual(article["title"], "New")
        self.assertEqual(article["tags"], summary["tags"])
        self.assertEqual(session.calls[0]["url"], "https://quickreads.app/api/articles/article-1")

    def test_queue_tags_are_exact_and_ambiguous_tags_are_skipped(self):
        self.assertEqual(tts.choose_queue_tag({"tags": [{"name": "tts-en"}]}), "tts-en")
        self.assertIsNone(tts.choose_queue_tag({"tags": [{"name": "TTS-EN"}]}))
        self.assertIsNone(
            tts.choose_queue_tag({"tags": [{"name": "tts-en"}, {"name": "tts-de"}]})
        )

    def test_malformed_list_and_detail_responses_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "JSON array"):
            tts.fetch_article_summaries(FakeSession([FakeResponse({"items": []})]))

        with self.assertRaisesRegex(RuntimeError, "invalid article"):
            tts.fetch_article(
                FakeSession([FakeResponse({"id": "wrong"})]),
                {"id": "expected", "tags": []},
            )


class StateMigrationTests(unittest.TestCase):
    def test_v2_migration_preserves_historical_episode_identity(self):
        raw = {
            "version": 2,
            "docs": {
                "old-id": {
                    "doc_id": "old-id",
                    "title": "Old article",
                    "fingerprint": "abc",
                    "processed_at": "2026-01-01T00:00:00Z",
                    "publish_mode": "combined",
                    "parts": [{"guid": "reader-old-id-abc", "filename": "old.mp3"}],
                }
            },
        }

        state = tts.normalize_state(raw)

        self.assertEqual(state["version"], 3)
        migrated = state["docs"]["readwise:old-id"]
        self.assertEqual(migrated["source_provider"], "readwise")
        self.assertEqual(migrated["parts"][0]["guid"], "reader-old-id-abc")
        self.assertEqual(migrated["parts"][0]["filename"], "old.mp3")

    def test_v1_migration_uses_provider_qualified_key(self):
        raw = {
            "processed": {
                "legacy-id": {
                    "title": "Legacy",
                    "processed_at": "2026-01-01T00:00:00Z",
                    "parts": [{"guid": "legacy-guid", "filename": "legacy.mp3"}],
                }
            }
        }

        state = tts.normalize_state(raw)

        self.assertIn("readwise:legacy-id", state["docs"])
        self.assertEqual(state["docs"]["readwise:legacy-id"]["parts"][0]["guid"], "legacy-guid")


class ProcessingTests(unittest.TestCase):
    def test_link_or_missing_content_is_too_short(self):
        for article in (
            {"id": "link-1", "title": "Link", "type": "link", "content": "<p>Ignored</p>"},
            {"id": "empty-1", "title": "Empty", "type": "article", "content": None},
        ):
            with self.subTest(article=article["id"]):
                status, part_count = tts.process_doc(
                    FakeSession([]), article, "tts-en", tts.empty_state()
                )
                self.assertEqual((status, part_count), ("too_short", 0))

    def test_quick_reads_content_uses_namespaced_state_files_and_guids(self):
        article = {
            "id": "new-id",
            "title": "Quick Reads Article",
            "url": "https://example.com/article",
            "type": "article",
            "content": "<p>" + ("Substantial article content. " * 20) + "</p>",
        }
        state = tts.empty_state()

        def fake_synthesize(**kwargs):
            kwargs["out_path"].write_bytes(b"mp3")

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(tts, "AUDIO_DIR", pathlib.Path(temp_dir)), mock.patch.object(
                tts, "synthesize_chunk_with_retry", side_effect=fake_synthesize
            ):
                status, part_count = tts.process_doc(FakeSession([]), article, "tts-en", state)

        self.assertEqual((status, part_count), ("processed", 1))
        record = state["docs"]["quickreads:new-id"]
        self.assertEqual(record["source_provider"], "quickreads")
        self.assertTrue(record["parts"][0]["guid"].startswith("quickreads-new-id-"))
        self.assertIn("_quickreads_docnew-id_", record["parts"][0]["filename"])

    def test_main_isolates_detail_fetch_failures(self):
        summaries = [
            {"id": "bad", "tags": [{"name": "tts-en"}]},
            {"id": "empty", "tags": [{"name": "tts-en"}]},
        ]
        empty_article = {
            "id": "empty",
            "title": "Empty",
            "type": "article",
            "content": None,
            "tags": [{"name": "tts-en"}],
        }

        with mock.patch.object(tts, "ensure_dirs"), mock.patch.object(
            tts, "load_state", return_value=tts.empty_state()
        ), mock.patch.object(tts, "fetch_article_summaries", return_value=summaries), mock.patch.object(
            tts, "fetch_article", side_effect=[RuntimeError("detail failed"), empty_article]
        ), mock.patch.object(tts, "enforce_retention", return_value=0), mock.patch.object(
            tts, "remove_orphan_audio_files", return_value=0
        ), mock.patch.object(tts, "build_feed"), mock.patch.object(tts, "save_state"), mock.patch(
            "builtins.print"
        ) as print_mock:
            tts.main()

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("Failed for article bad: detail failed", output)
        self.assertIn("Skipped too-short article empty", output)
        self.assertIn("failed articles: 1", output)


if __name__ == "__main__":
    unittest.main()
