import tempfile
import unittest
from pathlib import Path

from captions import write_ass, words_from_caption
from pipeline import _local_hooks, _parse_json_response


class CaptionTests(unittest.TestCase):
    def test_ass_uses_uppercase_karaoke_words_and_selected_color(self):
        words = [
            {"word": "Hello", "start": 0.1, "end": 0.45},
            {"word": "world", "start": 0.45, "end": 0.9},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = write_ass(words, Path(directory) / "caption.ass", "cyan")
            text = path.read_text(encoding="utf-8")
        self.assertIn("&H00FFFF00", text)
        self.assertIn(r"{\kf35}HELLO", text)
        self.assertIn(r"{\kf45}WORLD", text)

    def test_caption_text_gets_clip_relative_word_timing(self):
        words = words_from_caption("edit this line", 3)
        self.assertEqual(len(words), 3)
        self.assertEqual(words[0]["start"], 0)
        self.assertAlmostEqual(words[-1]["end"], 3)


class PipelineTests(unittest.TestCase):
    def test_local_hooks_are_bounded_by_source_duration(self):
        transcript = [
            {"word": "first", "start": 0.3, "end": 0.7},
            {"word": "thought", "start": 0.8, "end": 1.2},
            {"word": "second", "start": 6.0, "end": 6.4},
        ]
        clips = _local_hooks(transcript, 7, 2)
        self.assertTrue(clips)
        self.assertTrue(all(0 <= clip["start"] < clip["end"] <= 7 for clip in clips))

    def test_gemini_json_parser_accepts_fenced_json(self):
        parsed = _parse_json_response('```json\n{"clips": []}\n```')
        self.assertEqual(parsed, {"clips": []})


if __name__ == "__main__":
    unittest.main()
