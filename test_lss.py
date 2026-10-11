"""モデル・Vault を使わない自己テスト。 uv run --frozen --offline python -m unittest -v"""
import importlib.util
import unittest

import lss

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def ch(file, line, headings=(), text="x"):
    return {"file": file, "title": lss.note_title(file), "headings": list(headings), "line": line, "part": 1, "parts": 1,
            "text": text, "birth": 0, "mtime": 0}


class Split(unittest.TestCase):
    def test_headings(self):
        md = "---\ncreated: 2024-01-01\n---\n# A\n本文A\n## B\n```\n# コード内\n```\n本文B\n# C#\n本文C\n"
        s = lss.split_markdown(md)
        self.assertEqual([x["headings"] for x in s], [["(frontmatter)"], ["A"], ["A", "B"], ["C#"]])
        self.assertEqual([x["line"] for x in s], [2, 5, 7, 12])

    def test_chunk_keeps_every_char(self):
        text = "".join(f"文{i}です。" for i in range(300)) + "\n\n" + "x" * 3000
        parts = lss.chunk_text(text, 1200)
        self.assertTrue(all(len(p) <= 1200 for p in parts))
        self.assertEqual("".join(parts).replace("\n", "").replace(" ", ""), text.replace("\n", ""))

    def test_fit_tokens_splits_instead_of_truncating(self):
        big = ch("Notes/big.md", 3, ["h"], "あいうえお。" * 800)  # 4800文字 > MAX_TOKENS
        fitted, splits = lss.fit_tokens([big, ch("Notes/s.md", 1)], count=len)  # 1文字=1トークンの偽カウンター
        parts = [c for c in fitted if c["file"] == "Notes/big.md"]
        self.assertTrue(splits and all(c["tokens"] <= lss.MAX_TOKENS for c in fitted))
        self.assertEqual("".join(c["text"] for c in parts), big["text"])
        self.assertEqual([(c["part"], c["parts"]) for c in parts], [(i + 1, len(parts)) for i in range(len(parts))])
        with self.assertRaises(lss.BuildError):
            lss.fit_tokens([ch("Notes/s.md", 1, text="x" * 50)], count=lambda s: 10**6)

    def test_formats(self):
        self.assertEqual(lss.note_title("Notion/会議 0123456789abcdef0123456789abcdef.md"), "会議")
        self.assertEqual(lss.doc_text("", "x"), "title: none | text: x")
        self.assertEqual(lss.chunk_title(ch("dailynotes/2025-03-01.md", 1, ["やったこと"])), "2025-03-01 > やったこと")


class Sample(unittest.TestCase):
    def test_plan(self):
        files = [{"rel": f"{d}/{i:04}.md", "size": i} for d, n in lss.EXPECTED_FILES.items() for i in range(n)]
        s = lss.pick_sample(files)
        for d, k in lss.SAMPLE_PLAN.items():
            g = [f["size"] for f in s if f["rel"].startswith(d + "/")]
            self.assertEqual((len(g), min(g), max(g)), (k, 0, lss.EXPECTED_FILES[d] - 1))

    def test_daily_date(self):
        self.assertEqual(lss.daily_date("dailynotes/2025-03-01.md"), "2025-03-01")
        self.assertIsNone(lss.daily_date("dailynotes/2025-02-30.md"))
        self.assertIsNone(lss.daily_date("Notes/2025-03-01.md"))


class Rank(unittest.TestCase):
    chunks = [ch("Notes/a.md", 1, ["x"]), ch("Notes/a.md", 5, ["y"]), ch("Notes/a.md", 9, ["z"]), ch("Notes/a.md", 13, ["w"]),
              ch("Notes/b.md", 1), ch("dailynotes/2025-01-01.md", 1), ch("dailynotes/2025-06-01.md", 1),
              ch("dailynotes/2025-03-01.md", 1), ch("dailynotes/2025-12-01.md", 1), ch("dailynotes/2025-11-01.md", 1)]
    scores = [0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.5, 0.8, 0.1, 0.5]

    def test_distinct_notes_and_limits(self):
        hits = lss.rank_notes(self.chunks, self.scores, 5)
        self.assertEqual([h["best"]["file"] for h in hits],
                         ["Notes/a.md", "Notes/b.md", "dailynotes/2025-01-01.md", "dailynotes/2025-03-01.md", "dailynotes/2025-11-01.md"])
        self.assertEqual(len(hits[0]["others"]), lss.OTHERS)
        self.assertEqual(len(lss.rank_notes(self.chunks, self.scores, 7)), 7)

    def test_recent(self):
        hits = lss.recent_daily(self.chunks, self.scores, 2, 3)  # 候補: 01-01, 03-01, 11-01(同点は新しい方)
        self.assertEqual([(h["date"], h["relevance_rank"]) for h in hits], [("2025-11-01", 3), ("2025-03-01", 2)])
        self.assertNotIn("2025-12-01", [h["date"] for h in lss.recent_daily(self.chunks, self.scores, 5, 4)])


class Images(unittest.TestCase):
    def test_image_inputs(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        with TemporaryDirectory(dir=lss.ROOT) as tmp:
            data = Path(tmp)
            expected = {**lss.EXPECTED_LINKS, "total": 2}
            with patch.object(lss, "DATA", data), patch.object(lss, "EXPECTED_LINKS", expected):
                self.assertEqual(lss.image_inputs({"z.png": [], "a.png": []}, True), ["a.png", "z.png"])
                with self.assertRaises(lss.BuildError):
                    lss.image_inputs({"a.png": []}, True)
                (data / "images").mkdir()
                (data / "images" / "sample.txt").write_text("sample.jpg\n", encoding="utf-8")
                self.assertEqual(lss.image_inputs({"z.png": []}, False), ["sample.jpg"])

    def test_refs(self):
        refs = lss.parse_image_refs("![[a.png]] [[b.PNG|200]] ![[note]]\n![x](attachments/c%20d.jpg) ![](https://e.com/e.png) | [[f.png\\|100]] |")
        self.assertEqual([(r["target"], r["syntax"], r["external"], r["line"]) for r in refs],
                         [("a.png", "wiki-embed", False, 1), ("b.PNG", "wiki-link", False, 1), ("f.png", "wiki-link", False, 2),
                          ("attachments/c d.jpg", "md-embed", False, 2), ("https://e.com/e.png", "md-embed", True, 2)])

    def test_resolve(self):
        att = ["attachments/a.png", "attachments/sub/a.png", "attachments/B.png"]
        self.assertEqual(lss.resolve_image("a.png", "Notes/n.md", att), (None, "ambiguous:2"))
        self.assertEqual(lss.resolve_image("b.png", "dailynotes/2025-01-01.md", att), ("attachments/B.png", None))
        self.assertEqual(lss.resolve_image("attachments/b.png", "Notes/n.md", att), ("attachments/B.png", None))
        self.assertEqual(lss.resolve_image("other/b.png", "Notes/n.md", att), (None, "path-not-found"))
        self.assertEqual(lss.resolve_image("b", "Notes/n.md", att), (None, "not-found"))


class Guard(unittest.TestCase):
    def test_no_write_to_vault(self):
        with self.assertRaises(RuntimeError):
            lss.write_text(lss.VAULT / "x.txt", "x")
        with self.assertRaises(RuntimeError):
            lss.write_text(lss.ROOT / "x.txt", "x")

    @unittest.skipUnless(HAS_NUMPY, "numpy は uv sync 後に入る")
    def test_index_roundtrip_and_complete_flag(self):
        import json

        import numpy as np

        name = "__selftest__"
        v = np.eye(2, lss.DIM, dtype=np.float32)
        meta = lss.model_meta("text")
        lss.save_index(name, meta, [ch("Notes/a.md", 1), ch("Notes/b.md", 1)], v)
        lss.save_index(name, meta, [ch("Notes/a.md", 1), ch("Notes/b.md", 1)], v)  # 上書きしても読める
        self.assertEqual(lss.load_index(name)[0]["count"], 2)
        p = lss.DATA / "index" / name / "meta.json"
        lss.write_text(p, json.dumps({**json.loads(p.read_text(encoding="utf-8")), "complete": False}))
        with self.assertRaises(lss.UserError):
            lss.load_index(name)
        lss.remove_tree(lss.DATA / "index" / name)
        self.assertEqual(lss.check_same_model([{**meta, "dim": lss.DIM}]), "text")
        with self.assertRaises(lss.UserError):
            lss.check_same_model([{**meta, "dim": lss.DIM, "revision": "other"}])


if __name__ == "__main__":
    unittest.main()
