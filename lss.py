"""lss.py — Obsidian Vault のローカル Semantic Search(EmbeddingGemma 2 公式チェックポイント + sentence-transformers, CPU)

  uv run --no-project --no-python-downloads --python 3.14 lss.py preflight   依存パッケージなしの事前確認(AI)
  uv run --frozen lss.py download                                             モデルの初回取得(人間・通信あり)
  uv run --frozen --offline lss.py verify [--images 画像1 画像2]              オフライン検証
  uv run --frozen --offline lss.py build --name sample50 --sample             段階1: 50件
  uv run --frozen --offline lss.py build --name full                          段階2: 全件
  uv run --frozen --offline lss.py search --index full                        起動したまま何度でも検索
  uv run --frozen --offline lss.py search --index full --eval questions.txt --repeat 2
  uv run --frozen --offline lss.py links                                      段階3: ノート→画像のリンク表
  uv run --frozen --offline lss.py images --mode ocr|vec --name images-ocr|images-vec
  uv run --frozen --offline lss.py images --mode vec --name images-vec-all --all
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

PROCESS_START = time.perf_counter()
for _s in (sys.stdout, sys.stderr):  # リダイレクト時に cp932 で落ちないように(PowerShell 側も UTF-8 にする)
    _s.reconfigure(encoding="utf-8")

VAULT = Path(r"C:\Users\yomor\OneDrive\Desktop\vault")
VAULT_NAME = "vault"  # Obsidian上のVault名(段階1で人間がリンクを開いて確認)
SOURCE_DIRS = ("Notes", "Notion", "dailynotes")
ATTACH_DIR = "attachments"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HF_HOME = ROOT / ".hf-home"  # モデルのキャッシュ。.gitignore 済み

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"  # 2026-10-06 の公式リポジトリのコミット
CONFIGS = {  # 同じチェックポイントから読み込む部分を選ぶ(開発者ガイドの config_kwargs)
    "text": {"vision_config": None, "audio_config": None},  # 270M
    "vision": {"audio_config": None},  # 440M
}
QUERY_PROMPT = "SearchQuery"  # = "task: search result | query: "
PROMPT_VERSION = "q:SearchQuery|d:title-text|v1"  # 入力形式を変えたら上げる
DIM = 768
MAX_CHARS = 1200  # 見出し節をこの文字数以下に分ける
MAX_TOKENS = 2048  # 文書1件の上限。超えたら黙って切らずに分割する
MIN_SPLIT_CHARS = 100  # これ以下でもトークン超過なら停止
BATCH = 16
SAMPLE_PLAN = {"Notes": 10, "Notion": 30, "dailynotes": 10}
TOP = 5
OTHERS = 2  # 同じノートの他の箇所の表示上限
RECENT_CANDIDATES = 10
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".avif", ".heic"}
EXPECTED_FILES = {"Notes": 72, "Notion": 2389, "dailynotes": 110}
EXPECTED_ATTACHMENTS = 367
EXPECTED_LINKS = {"notesNotion": 108, "dailynotes": 249, "overlap": 0, "total": 357,
                  "notesNotion_unresolved_embeds": 517, "notesNotion_external": 10}
FORMAT = 3

# huggingface_hub は import 時に環境変数を読む。download 以外は常にオフライン。
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" if sys.argv[1:2] == ["download"] else "1"


class UserError(Exception):
    """利用者に読ませるエラー(スタックトレースを出さない)"""


class BuildError(Exception):
    """索引作成の失敗。details に原因を持つ"""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


# ---------- Vault 読み取り専用ガード ----------
def inside(child, parent):
    c, p = os.path.normcase(str(Path(child).resolve())), os.path.normcase(str(Path(parent).resolve()))
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


if inside(ROOT, VAULT):
    sys.exit(f"プロジェクトがVault内にあります: {ROOT}")


def out(path):
    """書き込み先の確認。data/ の外、Vault の中には書かない。書き込みは必ずこれを通す。"""
    path = Path(path)
    if not inside(path, DATA) or inside(path, VAULT):
        raise RuntimeError(f"書き込み禁止の場所です: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path, text):
    out(path).write_text(text, encoding="utf-8")


def append_jsonl(path, obj):
    with out(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def remove_tree(path):
    if Path(path).exists():
        shutil.rmtree(out(path))


# ---------- Vault の列挙(メタデータのみ) ----------
def list_files(rel_dir, exts):
    root = VAULT / rel_dir
    if not root.is_dir():
        raise UserError(f"フォルダがありません: {root}")
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            p = Path(dirpath) / name
            if name.startswith(".") or p.suffix.lower() not in exts:
                continue
            st = p.stat()
            files.append({"abs": p, "rel": p.relative_to(VAULT).as_posix(), "size": st.st_size, "mtime": st.st_mtime,
                          "birth": getattr(st, "st_birthtime", st.st_ctime), "attrs": getattr(st, "st_file_attributes", 0)})
    return sorted(files, key=lambda f: f["rel"])


def online_only(f):  # OneDrive: RECALL_ON_DATA_ACCESS(0x400000) / OFFLINE(0x1000)
    return bool(f["attrs"] & (0x400000 | 0x1000))


def pick_sample(files, plan=SAMPLE_PLAN):
    """段階1(速度試験)。各フォルダをサイズ順に並べ、最短から最長まで等間隔に選ぶ。"""
    picked = []
    for d, k in plan.items():
        g = sorted((f for f in files if f["rel"].startswith(d + "/")), key=lambda f: (f["size"], f["rel"]))
        if len(g) <= k or k < 2:
            picked += g[:k]
        else:
            picked += [g[i] for i in sorted({round(i * (len(g) - 1) / (k - 1)) for i in range(k)})]
    return picked


def daily_date(rel):
    m = re.fullmatch(r"dailynotes/(\d{4})-(\d{2})-(\d{2})\.md", rel)
    if not m:
        return None
    try:
        return dt.date(*map(int, m.groups())).isoformat()
    except ValueError:
        return None


# ---------- Markdown 分割 ----------
def split_markdown(md):
    """[{headings, line(本文の最初の行, 1始まり), text}] を返す。"""
    lines = md.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections, stack, buf = [], [], []
    start, fence, i = 1, None, 0

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append({"headings": [h for _, h in stack], "line": start, "text": text})
        buf.clear()

    if lines[0] == "---" and "---" in lines[1:]:
        end = lines.index("---", 1)
        fm = "\n".join(lines[1:end]).strip()
        if fm:
            sections.append({"headings": ["(frontmatter)"], "line": 2, "text": fm})
        i = end + 1
    for i in range(i, len(lines)):
        line = lines[i]
        f = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        h = None if (fence or f) else re.match(r"^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$", line)
        if f:
            if not fence:
                fence = f.group(1)
            elif f.group(1)[0] == fence[0] and len(f.group(1)) >= len(fence):
                fence = None
        elif h and h.group(2):
            flush()
            level = len(h.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, h.group(2)))
            continue
        if not buf and not line.strip():
            continue
        if not buf:
            start = i + 1
        buf.append(line)
    flush()
    return sections


def chunk_text(text, limit=MAX_CHARS):
    """limit 文字以下に分ける。段落 → 文 → 強制分割。空白以外の文字は捨てない。"""
    if len(text) <= limit:
        return [text]
    pieces = []
    for p in re.split(r"\n[ \t]*\n", text):
        if len(p) <= limit:
            pieces.append(p)
        else:
            sents = [s for s in re.findall(r"[^。！？!?\n]*(?:[。！？!?\n]+|$)", p) if s]
            pieces += _pack([part for s in sents for part in _hard_split(s, limit)], limit, "")
    return _pack(pieces, limit, "\n\n")


def _hard_split(s, limit):
    parts = []
    while len(s) > limit:
        parts.append(s[:limit])
        s = s[limit:]
    return parts + ([s] if s else [])


def _pack(pieces, limit, sep):
    packed, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(sep) + len(p) > limit:
            packed.append(cur)
            cur = p
        else:
            cur = cur + sep + p if cur else p
    if cur:
        packed.append(cur)
    return [s.strip() for s in packed if s.strip()]


# ---------- 入力形式 ----------
def note_title(rel):  # Notion書き出しの末尾32桁ハッシュを外す
    return re.sub(r"\s+[0-9a-f]{32}$", "", Path(rel).stem, flags=re.I)


def chunk_title(c):
    return " > ".join([c["title"], *c["headings"]])


def doc_text(title, text):  # モデルカードの文書形式。prompt_name="Document" は title: none になるので使わない
    return f"title: {title or 'none'} | text: {text}"


def fit_tokens(chunks, count):
    """実際のトークナイザーで数え、MAX_TOKENS を超える文書は半分以下に分ける(黙った切り捨てを防ぐ)。"""
    fitted, splits = [], []
    for c in chunks:
        queue = [c]
        while queue:
            x = queue.pop(0)
            n = count(doc_text(chunk_title(x), x["text"]))
            if n <= MAX_TOKENS:
                fitted.append({**x, "tokens": n})
                continue
            if len(x["text"]) <= MIN_SPLIT_CHARS:
                raise BuildError(f"短いのにトークン数が上限を超えます: {x['file']}:{x['line']}", {"file": x["file"], "tokens": n})
            parts = chunk_text(x["text"], math.ceil(len(x["text"]) / 2))
            splits.append({"file": x["file"], "line": x["line"], "tokens": n, "into": len(parts)})
            queue = [{**x, "text": p} for p in parts] + queue
    groups = {}
    for c in fitted:
        groups.setdefault((c.get("image"), c["file"], c["line"], tuple(c["headings"])), []).append(c)
    for g in groups.values():
        for k, c in enumerate(g):
            c["part"], c["parts"] = k + 1, len(g)
    return fitted, splits


# ---------- モデル(オフラインで読む) ----------
def load_model(config):
    import torch
    from sentence_transformers import SentenceTransformer

    try:
        m = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", local_files_only=True,
                                config_kwargs=CONFIGS[config], model_kwargs={"torch_dtype": torch.float32})
    except OSError as e:
        raise UserError(f"モデルがキャッシュにありません({e.__class__.__name__})。人間が次を実行してください: uv run --frozen lss.py download")
    dtypes = {p.dtype for p in m.parameters()}
    if dtypes != {torch.float32}:
        raise UserError(f"float32 で読み込めていません: {dtypes}")
    m.max_seq_length = MAX_TOKENS
    return m


def token_counter(m):
    tok = getattr(m.tokenizer, "tokenizer", m.tokenizer)  # プロセッサの場合は中のトークナイザー
    return lambda s: len(tok(s, add_special_tokens=True)["input_ids"])


def check_vectors(v):
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != DIM:
        raise BuildError(f"ベクトルの形が想定外です: {v.shape}")
    if not np.isfinite(v).all():
        raise BuildError("NaN/Inf を含むベクトルが返りました")
    norms = np.linalg.norm(v, axis=1)
    if np.abs(norms - 1).max() > 1e-3:
        raise BuildError(f"正規化されていないベクトルがあります: {norms.min():.4f}〜{norms.max():.4f}")
    return v


def encode(m, inputs, prompt_name=None):
    try:
        v = m.encode(inputs, prompt_name=prompt_name, batch_size=BATCH, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=len(inputs) > BATCH)
    except Exception as e:  # 1件でも失敗したら索引全体を止める
        raise BuildError(f"埋め込み中に例外: {e!r}")
    return check_vectors(v)


# ---------- 索引の保存・読込 ----------
def save_index(name, meta, chunks, vectors):
    """全件成功時だけ呼ぶ。tmp に書き、旧索引を退避してから入れ替える(途中で失敗しても完成版が1つ残る)。"""
    import numpy as np

    final = DATA / "index" / name
    tmp, old = final.with_name(name + ".tmp"), final.with_name(name + ".old")
    remove_tree(tmp)
    remove_tree(old)
    np.save(out(tmp / "vectors.npy"), vectors)
    write_text(tmp / "chunks.json", json.dumps(chunks, ensure_ascii=False))
    meta = {**meta, "format": FORMAT, "complete": True, "count": len(chunks), "dim": int(vectors.shape[1])}
    write_text(tmp / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    if final.exists():
        final.rename(out(old))
    try:
        tmp.rename(out(final))
    except OSError:
        if old.exists():
            old.rename(final)
        raise
    remove_tree(old)


def load_index(name):
    import numpy as np

    d = DATA / "index" / name
    if not (d / "meta.json").exists():
        raise UserError(f"索引がありません: {d}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("complete") is not True:
        raise UserError(f"完成版でない、または古い形式の索引です: {name}")
    chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(d / "vectors.npy")
    if len(chunks) != meta["count"] or vectors.shape != (meta["count"], meta["dim"]):
        raise UserError(f"索引の件数が一致しません: {name}")
    return meta, chunks, vectors


def model_meta(config):
    return {"model": MODEL_ID, "revision": REVISION, "config": config, "dtype": "float32",
            "prompt_version": PROMPT_VERSION, "max_tokens": MAX_TOKENS}


def check_same_model(metas):
    """索引どうし・現在のコードと、モデル・リビジョン・入力形式・次元が一致するか。読み込む構成を返す。"""
    for m in metas:
        bad = [k for k in ("model", "revision", "prompt_version") if m.get(k) != model_meta("text")[k]]
        if bad or m["dim"] != DIM:
            raise UserError(f"索引のモデル情報が現在と違います({', '.join(bad) or 'dim'})。索引を作り直してください")
    configs = {m["config"] for m in metas}
    if configs == {"text", "vision"}:
        v = DATA / "verify.json"
        if not (v.exists() and json.loads(v.read_text(encoding="utf-8")).get("text_vision_compatible") is True):
            raise UserError("text構成とvision構成の索引を混ぜるには、verify --images で互換性(cos≥0.9999)の確認が必要です")
    return "vision" if "vision" in configs else "text"


# ---------- 順位付け ----------
def rank_notes(chunks, scores, limit, keep=lambda c: True):
    """異なるノートを limit 件。各ノートは代表箇所1件+別の節を最大 OTHERS 件。同点はパスの降順(日記は新しい日付が先)。"""
    order = [i for i in range(len(chunks)) if keep(chunks[i])]
    order.sort(key=lambda i: chunks[i]["file"], reverse=True)
    order.sort(key=lambda i: -scores[i])
    hits = {}
    for r, i in enumerate(order):
        if len(hits) >= limit and r >= 300:
            break
        c = chunks[i]
        g = hits.get(c["file"])
        if g is None:
            if len(hits) < limit:
                hits[c["file"]] = {"best": c, "score": scores[i], "others": []}
            continue
        same = lambda x: (x["line"], x["headings"], x.get("image")) == (c["line"], c["headings"], c.get("image"))
        if len(g["others"]) < OTHERS and not same(g["best"]) and not any(same(o["c"]) for o in g["others"]):
            g["others"].append({"c": c, "score": scores[i]})
    return list(hits.values())


RECENT_NOTE = "候補は関連度上位{n}件の日記です。関連する日記が{n}件より少ない場合、関連の薄い日記も候補に入り、日付が新しければ表示されます。"


def recent_daily(chunks, scores, top, candidates):
    """関連度上位 candidates 件の日記を選び、ファイル名の日付が新しい順に top 件。閾値は使わない。"""
    hits = rank_notes(chunks, scores, candidates, lambda c: daily_date(c["file"]) is not None)
    for r, h in enumerate(hits, 1):
        h["relevance_rank"], h["date"] = r, daily_date(h["best"]["file"])
    return sorted(hits, key=lambda h: h["date"], reverse=True)[:top]


# ---------- 画像リンク ----------
def parse_image_refs(md):
    refs = []
    for i, line in enumerate(md.replace("\r\n", "\n").split("\n"), 1):
        found = [(m.group(2).split("|")[0].split("#")[0].rstrip("\\"), "wiki-embed" if m.group(1) else "wiki-link", m.group(0))
                 for m in re.finditer(r"(!?)\[\[([^\]\n]+?)\]\]", line)]
        found += [(urllib.parse.unquote(m.group(2)), "md-embed" if m.group(1) else "md-link", m.group(0))
                  for m in re.finditer(r'(!?)\[[^\]\n]*\]\(\s*<?([^)>\n]+?)>?(?:\s+"[^"]*")?\s*\)', line)]
        for target, syntax, raw in found:
            target = target.strip()
            ref = {"line": i, "syntax": syntax, "raw": raw, "target": target}
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) and not re.match(r"^[a-z]:[\\/]", target, re.I):
                if syntax.endswith("embed") or Path(re.split(r"[?#]", target)[0]).suffix.lower() in IMAGE_EXTS:
                    refs.append({**ref, "external": True})  # 取得しない。記録のみ
            elif Path(target).suffix.lower() in IMAGE_EXTS:
                refs.append({**ref, "external": False})
    return refs


def resolve_image(target, note_rel, attach):
    """attach = attachments の rel 一覧。拡張子違い・あいまい一致は推測しない。"""
    by_rel = {a.lower(): a for a in attach}
    t = target.replace("\\", "/").removeprefix("./")
    if "/" in t:
        for c in (t, f"{Path(note_rel).parent.as_posix()}/{t}"):
            hit = by_rel.get(os.path.normpath(c).replace("\\", "/").lower())
            if hit:
                return hit, None
        return None, "path-not-found"
    hits = [a for a in attach if Path(a).name.lower() == t.lower()]
    return (hits[0], None) if len(hits) == 1 else (None, f"ambiguous:{len(hits)}" if hits else "not-found")


def group_of(rel):
    return "dailynotes" if rel.startswith("dailynotes/") else "notesNotion"


# ---------- 表示 ----------
def fmt_time(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def heading(c):
    return (" > ".join(c["headings"]) or "(見出しなし)") + (f" ({c['part']}/{c['parts']})" if c["parts"] > 1 else "")


def snippet(text, n=160):
    s = " ".join(text.split())
    return s[:n] + "…" if len(s) > n else s


def print_hits(q, hits, recent, candidates):
    print(f"\n検索語: {q}")
    print(f"表示: 直近モード(関連度上位{candidates}件の日記をファイル名の日付が新しい順に最大{len(hits)}件)" if recent
          else f"表示: 関連度順に異なる{len(hits)}ノート")
    for n, h in enumerate(hits, 1):
        c = h["best"]
        print(f"\n[{n}] {h['score']:.3f}  {c['file']}" + (f"  (候補内の関連度 {h['relevance_rank']}位)" if recent else ""))
        if daily_date(c["file"]):
            print(f"    日記の日付: {daily_date(c['file'])}(ファイル名。本文の出来事の日付とは限らない)")
        print(f"    見出し: {heading(c)}")
        if c.get("image"):
            print(f"    画像経由: {VAULT / c['image']} ({'画像の埋め込みで一致' if c['kind'] == 'image-vec' else '画像内の文字(OCR)で一致'})")
        if c["text"]:
            print(f"    抜粋: {snippet(c['text'])}")
        print(f"    参照: {VAULT / c['file']}:{c['line']}")
        print(f"    Obsidian: obsidian://open?vault={urllib.parse.quote(VAULT_NAME)}&file={urllib.parse.quote(c['file'], safe='')}")
        print(f"    ファイル日時(OSの記録。本文の出来事の日付ではない): 作成 {fmt_time(c['birth'])} / 更新 {fmt_time(c['mtime'])}")
        if h["others"]:
            print("    同じノートの他の箇所: " + " / ".join(f"{heading(o['c'])} L{o['c']['line']} ({o['score']:.3f})" for o in h["others"]))
    if recent:
        print(f"\n※{RECENT_NOTE.format(n=candidates)}")


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def hit_table(hits, recent):
    head = "| 順位 | 判定 | スコア |" + (" 日記の日付 | 候補内の関連度 |" if recent else "") + " ノート | 見出し | 抜粋 | 同じノートの他の箇所 |\n"
    head += "|---|---|---|" + ("---|---|" if recent else "") + "---|---|---|---|\n"
    rows = ""
    for n, h in enumerate(hits, 1):
        c = h["best"]
        extra = f" {h['date']} | {h['relevance_rank']}位 |" if recent else ""
        others = " / ".join(f"{heading(o['c'])} L{o['c']['line']}" for o in h["others"])
        rows += f"| {n} |  | {h['score']:.3f} |{extra} {cell(c['file'])}:{c['line']} | {cell(heading(c))} | {cell(snippet(c['text'] or '(画像)', 120))} | {cell(others)} |\n"
    return head + rows + "\n"


KEYWORD_FORM = """### キーワード検索との比較(人間が先に記入)

| 項目 | 記入 |
|---|---|
| 使った検索語(試した順) |  |
| 試行回数 / 所要時間(分) |  |
| キーワード検索で見つけた関連ノート |  |
| Semantic Searchだけで見つかった関連ノート |  |
| キーワード検索だけで見つかった関連ノート |  |
| どちらでも見つからなかったが後で分かった関連ノート |  |

"""


# ---------- コマンド ----------
def cmd_preflight(a):
    """依存パッケージ不要。Vault は一覧と属性だけを見る(本文は読まない)。"""
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for d in SOURCE_DIRS:
        files = list_files(d, {".md"})
        print(f"{d}: {len(files)} 件 (既知 {EXPECTED_FILES[d]})")
        ok &= len(files) == EXPECTED_FILES[d]
    bad = [f["rel"] for f in list_files("dailynotes", {".md"}) if not daily_date(f["rel"])]
    print(f"yyyy-MM-dd.md でない日記: {len(bad)} 件 {bad}")
    att = list_files(ATTACH_DIR, IMAGE_EXTS)
    print(f"attachments の画像: {len(att)} 件 (既知 {EXPECTED_ATTACHMENTS})")
    online = [f["rel"] for d in (*SOURCE_DIRS, ATTACH_DIR) for f in list_files(d, {".md"} | IMAGE_EXTS) if online_only(f)]
    print(f"OneDriveのオンラインのみ: {len(online)} 件 {online[:20]}")
    ok &= not bad and len(att) == EXPECTED_ATTACHMENTS and not online
    print("preflight:", "OK" if ok else "要確認(人間に報告)")
    return 0 if ok else 1


def cmd_download(a):
    """人間が実行する唯一の通信。固定リビジョンの公式ファイルを .hf-home に取得する。"""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=REVISION)
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    print(f"取得先: {path}\n合計: {size / 1e9:.2f} GB")


def cmd_verify(a):
    import huggingface_hub
    import numpy as np
    import sentence_transformers
    import torch
    import transformers

    rep = {"python": sys.version.split()[0], "sentence_transformers": sentence_transformers.__version__,
           "transformers": transformers.__version__, "torch": torch.__version__,
           "offline": huggingface_hub.constants.HF_HUB_OFFLINE}
    if not rep["offline"]:
        raise UserError("オフラインになっていません")
    snap = Path(huggingface_hub.snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    rep["snapshot"], rep["download_bytes"] = str(snap), sum(p.stat().st_size for p in snap.rglob("*") if p.is_file())
    if snap.name != REVISION:
        raise UserError(f"キャッシュのリビジョンが違います: {snap.name}")

    t = time.perf_counter()
    m = load_model("text")
    rep["text_load_s"] = time.perf_counter() - t
    rep["text_params"] = sum(p.numel() for p in m.parameters())
    rep["text_weights_GB_estimate"] = rep["text_params"] * 4 / 1e9  # float32 の重みだけの見積もり
    rep["search_query_prompt"] = m.prompts.get(QUERY_PROMPT)
    if rep["search_query_prompt"] != "task: search result | query: ":
        raise UserError(f"SearchQuery の接頭辞が想定と違います: {rep['search_query_prompt']!r}")

    texts = [doc_text("猫の世話", "猫には毎日新鮮な水と決まった量のごはんを与える。"), doc_text("株価", "日経平均は前日比で300円下落した。")]
    q = encode(m, ["猫の飼い方"], QUERY_PROMPT)
    d = encode(m, texts)
    rep["dim"], rep["cos_related"], rep["cos_unrelated"] = int(d.shape[1]), float(q[0] @ d[0]), float(q[0] @ d[1])
    rep["japanese_order_ok"] = rep["cos_related"] > rep["cos_unrelated"]
    # 自前の文書形式 == Document プロンプト(title: none)で、余計な接頭辞が付いていないこと
    same = encode(m, [doc_text("", "猫")])[0] @ encode(m, ["猫"], "Document")[0]
    rep["doc_format_matches_Document_prompt"] = float(same)

    # トークン数: 自前の数え方 == 実際の処理経路、上限超えは実際の経路で黙って切られること
    count = token_counter(m)
    s = doc_text("上限", ("漢字とひらがなとカタカナ、English words、数字123を混ぜた文章。" * 40)[:MAX_CHARS])
    long_s = s * 8
    real = lambda x: int(m.tokenize([x])["input_ids"].shape[1])
    rep["tokens_1200chars"], rep["tokens_1200chars_real"] = count(s), real(s)
    rep["long_count"], rep["long_real"] = count(long_s), real(long_s)
    rep["counter_matches_real_path"] = rep["tokens_1200chars"] == rep["tokens_1200chars_real"]
    rep["real_path_truncates"] = rep["long_real"] <= MAX_TOKENS < rep["long_count"]

    batch = [s] * BATCH
    t = time.perf_counter()
    encode(m, batch)
    rep["chunks_per_s_1200chars"] = BATCH / (time.perf_counter() - t)

    if a.images:
        for p in a.images:
            if not inside(p, VAULT / ATTACH_DIR):
                raise UserError(f"attachments 外の画像は使わない: {p}")
        t = time.perf_counter()
        mv = load_model("vision")
        rep["vision_load_s"] = time.perf_counter() - t
        rep["vision_params"] = sum(p.numel() for p in mv.parameters())
        probe = [*texts, "プロスペクト理論", "こどものスマホの手続き"]
        rep["text_vision_min_cos"] = float(np.min(np.sum(encode(m, probe) * encode(mv, probe), axis=1)))
        rep["text_vision_compatible"] = rep["text_vision_min_cos"] >= 0.9999
        v = encode(mv, [{"image": str(a.images[0])}, {"image": str(a.images[1])}, {"image": str(a.images[0])}])
        rep["image_cos_1_2"], rep["image_cos_1_1"] = float(v[0] @ v[1]), float(v[0] @ v[2])
        rep["image_cos_1_text"] = float(v[0] @ encode(mv, [doc_text("", "画像")])[0])
        rep["image_content_reflected"] = rep["image_cos_1_2"] < 0.999 and rep["image_cos_1_1"] > 0.9999 and rep["image_cos_1_text"] < 0.999

    write_text(DATA / "verify.json", json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    must = ["japanese_order_ok", "counter_matches_real_path", "real_path_truncates"] + (["image_content_reflected"] if a.images else [])
    failed = [k for k in must if not rep[k]] + ([] if rep["dim"] == DIM else ["dim"]) + ([] if rep["doc_format_matches_Document_prompt"] > 0.9999 else ["doc_format"])
    print("verify:", "OK" if not failed else f"要確認 {failed}")
    return 0 if not failed else 1


def read_chunks(files):
    chunks, problems = [], []
    for f in files:
        try:
            md = f["abs"].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            problems.append({"file": f["rel"], "error": repr(e)})
            continue
        title = note_title(f["rel"])
        sections = split_markdown(md) or [{"headings": [], "line": 1, "text": title}]
        for s in sections:
            parts = chunk_text(s["text"])
            for k, text in enumerate(parts, 1):
                chunks.append({"file": f["rel"], "title": title, "headings": s["headings"], "line": s["line"], "part": k,
                               "parts": len(parts), "text": text, "birth": f["birth"], "mtime": f["mtime"]})
    for f in files:  # 読み取り中に変わったファイル
        try:
            st = f["abs"].stat()
            if (st.st_size, st.st_mtime) != (f["size"], f["mtime"]):
                problems.append({"file": f["rel"], "error": "読み取り中に変更された"})
        except OSError as e:
            problems.append({"file": f["rel"], "error": repr(e)})
    return chunks, problems


def run_build(name, build):
    """build() が (meta, chunks, vectors) を返したら保存。BuildError なら原因を記録して停止し、既存の索引は残す。"""
    fail = DATA / "logs" / f"build-failure-{name}.json"
    try:
        meta, chunks, vectors = build()
    except BuildError as e:
        rec = {"at": dt.datetime.now().isoformat(), "name": name, "message": str(e), "details": e.details}
        write_text(fail, json.dumps(rec, ensure_ascii=False, indent=2))
        append_jsonl(DATA / "logs" / "build.jsonl", {"result": "failed", **rec})
        raise UserError(f"{e}。索引は保存していません(既存の索引はそのまま)。記録: {fail}")
    save_index(name, meta, chunks, vectors)
    if fail.exists():
        fail.unlink()
    append_jsonl(DATA / "logs" / "build.jsonl", {"result": "ok", "name": name, **{k: v for k, v in meta.items() if k != "splits"}})
    print(json.dumps({k: v for k, v in meta.items() if k != "splits"}, ensure_ascii=False, indent=2))


def cmd_build(a):
    def build():
        t0 = time.perf_counter()
        files = [f for d in SOURCE_DIRS for f in list_files(d, {".md"})]
        if a.sample:
            files = pick_sample(files)
        chunks, problems = read_chunks(files)
        if problems:
            raise BuildError(f"読めない・UTF-8でない・読み取り中に変更されたファイルが {len(problems)} 件", problems)
        t1 = time.perf_counter()
        m = load_model("text")
        t2 = time.perf_counter()
        chunks, splits = fit_tokens(chunks, token_counter(m))
        vectors = encode(m, [doc_text(chunk_title(c), c["text"]) for c in chunks])
        t3 = time.perf_counter()
        if a.sample:
            write_text(DATA / "index" / f"{a.name}-files.txt", "\n".join(f["rel"] for f in files))
        meta = {**model_meta("text"), "created": dt.datetime.now().isoformat(), "sample": SAMPLE_PLAN if a.sample else None,
                "files_by_dir": {d: sum(f["rel"].startswith(d + "/") for f in files) for d in SOURCE_DIRS},
                "max_chunk_chars": max(len(c["text"]) for c in chunks), "max_chunk_tokens": max(c["tokens"] for c in chunks),
                "extra_splits": len(splits), "splits": splits,
                "seconds": {"read_split": t1 - t0, "model_load": t2 - t1, "embed": t3 - t2, "total": t3 - t0},
                "chunks_per_s": len(chunks) / (t3 - t2)}
        return meta, chunks, vectors

    run_build(a.name, build)


def load_links():
    p = DATA / "images" / "links.json"
    if not p.exists():
        raise UserError("先に links を実行してください")
    return {r["image"]: r["from"] for r in json.loads(p.read_text(encoding="utf-8"))["resolved"]}


def image_inputs(links, all_images):
    if all_images:
        images = sorted(links)
        if len(images) != EXPECTED_LINKS["total"]:
            raise BuildError(f"リンク表の画像数が既知値 {EXPECTED_LINKS['total']} 件と異なります: {len(images)} 件", images)
        return images
    return [s.strip() for s in (DATA / "images" / "sample.txt").read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def image_chunks(image, sources, kind, text):
    """(画像, リンク元ノート) ごとに1チャンク。file はリンク元ノートなので、結果にはノートが出る。"""
    seen, chunks = set(), []
    for s in sources:
        if s["file"] in seen:
            continue
        seen.add(s["file"])
        st = (VAULT / s["file"]).stat()
        chunks.append({"kind": kind, "image": image, "file": s["file"], "title": note_title(s["file"]),
                       "headings": [f"画像 {Path(image).name}"], "line": s["line"], "part": 1, "parts": 1, "text": text,
                       "birth": getattr(st, "st_birthtime", st.st_ctime), "mtime": st.st_mtime})
    return chunks


def cmd_images(a):
    if a.all and a.mode != "vec":
        raise UserError("--all は --mode vec でのみ使えます")
    links = load_links()

    def build():
        import numpy as np

        if a.mode == "ocr":  # ocr.ps1 の出力。文字なしは失敗ではないが、エラーは止める
            rows = [json.loads(line) for line in (DATA / "images" / "ocr.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]
            errors = [r for r in rows if r.get("error")]
            if errors:
                raise BuildError(f"OCRエラー {len(errors)} 件", errors)
            jp = r"\u3040-\u30ff\u4e00-\u9fff\u30fc"
            chunks = []
            for r in rows:
                text = re.sub(rf"(?<=[{jp}])[ \t]+(?=[{jp}])", "", r["text"]).strip()  # Windows OCR の文字間スペース
                if text:
                    chunks += image_chunks(r["image"], links[r["image"]], "image-ocr", text)
            m = load_model("text")
            chunks, splits = fit_tokens(chunks, token_counter(m))
            vectors = encode(m, [doc_text(chunk_title(c), c["text"]) for c in chunks])
            meta = {**model_meta("text"), "source": "image-ocr", "images": len(rows), "empty": [r["image"] for r in rows if not r["text"].strip()]}
        else:  # 画像そのもの。テキスト用接頭辞は付けない
            images = image_inputs(links, a.all)
            missing = [i for i in images if i not in links or not (VAULT / i).is_file()]
            if missing:
                raise BuildError(f"リンク表に無い、または存在しない画像: {missing}", missing)
            m = load_model("vision")
            v = encode(m, [{"image": str(VAULT / i)} for i in images])
            chunks, rows = [], []
            for i, img in enumerate(images):
                cs = image_chunks(img, links[img], "image-vec", "")
                chunks += cs
                rows += [v[i]] * len(cs)
            vectors = np.stack(rows)
            meta = {**model_meta("vision"), "source": "image-vec", "images": len(images)}
        return {**meta, "created": dt.datetime.now().isoformat()}, chunks, vectors

    run_build(a.name, build)


def cmd_links(a):
    notes = [f for d in SOURCE_DIRS for f in list_files(d, {".md"})]
    attach = [f["rel"] for f in list_files(ATTACH_DIR, IMAGE_EXTS)]
    resolved, unresolved, external = {}, [], []
    for n in notes:
        for r in parse_image_refs(n["abs"].read_text(encoding="utf-8")):
            where = {"file": n["rel"], "group": group_of(n["rel"]), "line": r["line"], "syntax": r["syntax"], "raw": r["raw"]}
            if r["external"]:
                external.append({**where, "url": r["target"]})
                continue
            img, reason = resolve_image(r["target"], n["rel"], attach)
            if img:
                resolved.setdefault(img, []).append(where)
            else:
                unresolved.append({**where, "target": r["target"], "reason": reason})
    groups = {img: {w["group"] for w in ws} for img, ws in resolved.items()}
    actual = {"notesNotion": sum("notesNotion" in g for g in groups.values()),
              "dailynotes": sum("dailynotes" in g for g in groups.values()),
              "overlap": sum(len(g) > 1 for g in groups.values()), "total": len(groups),
              "notesNotion_unresolved_embeds": sum(u["group"] == "notesNotion" and u["syntax"].endswith("embed") for u in unresolved),
              "notesNotion_external": sum(e["group"] == "notesNotion" for e in external)}
    summary = {"actual": actual, "expected": EXPECTED_LINKS, "mismatch": [k for k in actual if actual[k] != EXPECTED_LINKS[k]],
               "dailynotes_unresolved": sum(u["group"] == "dailynotes" for u in unresolved),
               "dailynotes_external": sum(e["group"] == "dailynotes" for e in external)}
    data = {"summary": summary, "resolved": [{"image": k, "from": v} for k, v in sorted(resolved.items())],
            "unresolved": unresolved, "external": external}
    write_text(DATA / "images" / "links.json", json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["mismatch"] else 0


def cmd_search(a):
    import numpy as np

    if a.candidates < a.top:
        raise UserError("--candidates は --top 以上にしてください")
    t0 = time.perf_counter()
    loaded = [load_index(n) for n in a.index.split(",")]
    metas = [x[0] for x in loaded]
    config = check_same_model(metas)
    chunks = [c for x in loaded for c in x[1]]
    vectors = np.concatenate([x[2] for x in loaded])
    t1 = time.perf_counter()
    m = load_model(config)
    count = token_counter(m)
    t2 = time.perf_counter()
    startup = {"python_to_main_s": t0 - PROCESS_START, "index_load_s": t1 - t0, "model_load_s": t2 - t1, "ready_s": t2 - PROCESS_START}
    print(f"準備完了 {startup['ready_s']:.1f} 秒(索引 {startup['index_load_s']:.1f} / モデル {startup['model_load_s']:.1f})。構成: {config}、チャンク {len(chunks)}")

    def ask(line):
        recent = line.startswith("@recent ")
        q = line.removeprefix("@recent ").strip()
        if count(q) > MAX_TOKENS:
            raise UserError("検索語が長すぎます")
        t = time.perf_counter()
        qv = encode(m, [q], QUERY_PROMPT)[0]
        t_emb = time.perf_counter()
        scores = (vectors @ qv).tolist()
        normal = rank_notes(chunks, scores, a.top)
        rec = recent_daily(chunks, scores, a.top, a.candidates) if recent else None
        t_end = time.perf_counter()
        return {"q": q, "recent": recent, "normal": normal, "recent_hits": rec,
                "embed_s": t_emb - t, "rank_s": t_end - t_emb, "total_s": t_end - t}

    timings = []
    if a.eval:
        lines = [s.strip() for s in Path(a.eval).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        rows = []
        for rep in range(a.repeat):
            for line in lines:
                r = ask(line)
                timings.append({k: r[k] for k in ("q", "embed_s", "rank_s", "total_s")})
                if rep == 0:
                    rows.append(r)
        md = f"# 検索評価 {dt.datetime.now():%Y-%m-%d %H:%M}\n\n索引: {a.index} / 構成: {config} / チャンク数: {len(chunks)}\n\n"
        md += "- 判定: 「役に立つ」「少し関連する」「関係ない」\n- メモ有無: 「あり」「なし(確認済み)」「未確定」。**スコアだけで「なし」にしない**\n"
        md += "- 先にキーワード検索の欄を記入してから、表を判定する\n- 日記の日付はファイル名、ファイル日時はOSの記録。どちらも本文の出来事の日付とは限らない\n\n"
        for r in rows:
            md += f"## {r['q']}\n\nメモ有無(人間記入): 未確定\n\n" + KEYWORD_FORM
            md += f"### Semantic Search: 関連度順(異なる{a.top}ノート)\n\n" + hit_table(r["normal"], False)
            if r["recent"]:
                md += f"### Semantic Search: 直近モード(関連度上位{a.candidates}件の日記 → 日付の新しい順)\n\n"
                md += RECENT_NOTE.format(n=a.candidates) + "「関係ない」の行が混ざった件数も判定する。\n\n" + hit_table(r["recent_hits"], True)
        path = DATA / "results" / f"eval-{a.index.replace(',', '+')}-{dt.datetime.now():%Y%m%d-%H%M%S}.md"
        write_text(path, md)
        print(f"評価用ファイル: {path}")
    else:
        print("検索語を入力(行頭に @recent で直近モード。空行で終了)")
        while True:
            try:
                line = input("検索> ").strip()
            except EOFError:
                break
            if not line:
                break
            try:
                r = ask(line)
            except UserError as e:
                print(f"エラー: {e}")
                continue
            print_hits(r["q"], r["recent_hits"] if r["recent"] else r["normal"], r["recent"], a.candidates)
            print(f"\n検索 {r['total_s']:.2f} 秒(埋め込み {r['embed_s']:.2f} / 順位付け {r['rank_s']:.2f})")
            timings.append({k: r[k] for k in ("q", "embed_s", "rank_s", "total_s")})
    for i, t in enumerate(timings, 1):
        print(f"  {i:2d}. {t['total_s']:.2f} 秒  {t['q'][:30]}")
    append_jsonl(DATA / "logs" / "search.jsonl", {"at": dt.datetime.now().isoformat(), "index": a.index, "config": config,
                                                  "startup": startup, "queries": timings})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    v = sub.add_parser("verify")
    v.add_argument("--images", nargs=2, type=Path)
    v.set_defaults(fn=cmd_verify)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--sample", action="store_true")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("search")
    s.add_argument("--index", required=True)
    s.add_argument("--top", type=int, default=TOP)
    s.add_argument("--candidates", type=int, default=RECENT_CANDIDATES)
    s.add_argument("--eval")
    s.add_argument("--repeat", type=int, default=1)
    s.set_defaults(fn=cmd_search)
    sub.add_parser("links").set_defaults(fn=cmd_links)
    im = sub.add_parser("images")
    im.add_argument("--mode", choices=["ocr", "vec"], required=True)
    im.add_argument("--name", required=True)
    im.add_argument("--all", action="store_true", help="links.json の全画像を索引化（vec専用）")
    im.set_defaults(fn=cmd_images)
    a = p.parse_args()
    if getattr(a, "name", None) and not re.fullmatch(r"[\w-]+", a.name):
        sys.exit("--name は英数字・_・- のみ")
    try:
        sys.exit(a.fn(a) or 0)
    except (UserError, BuildError) as e:
        sys.exit(f"エラー: {e}")
    except ModuleNotFoundError as e:
        sys.exit(f"エラー: {e.name} がありません。人間が uv sync を実行してください(preflight だけは不要)")


if __name__ == "__main__":
    main()
