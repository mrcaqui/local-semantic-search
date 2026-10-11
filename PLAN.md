# Obsidian Vault ローカル Semantic Search 段階的実装計画書

第3版 2026-10-11 / 実装担当: GPT-6 Luna
進め方は4章、Lunaへの指示文は13章。本書の作成時点では、パッケージのインストール、モデル取得、実装ファイルの作成、Vaultの本文・画像の読み取り、索引作成は行っていない。

---

## 0. 目的と範囲

| 項目 | 決定内容 |
|---|---|
| 目的 | 自然な日本語の質問から、関連する**ノートの該当箇所の一覧**を表示する。複数ノートに散らばっていれば複数表示する |
| 検索対象 | `C:\Users\yomor\OneDrive\Desktop\vault` の `Notes` 72件、`Notion` 2,389件、`dailynotes` 110件(すべて `yyyy-MM-dd.md`)。計2,571件 |
| 画像(段階3) | `attachments` の画像367件のうち、対象ノートから参照される実在画像357件(Notes・Notion 108件、dailynotes 249件、重複なし)。まず約10件で試す |
| モデル | `google/embeddinggemma-2` の Google 公式チェックポイントを、固定リビジョンで `sentence-transformers` から直接使う。CPU、`float32`、768次元。テキストの試験は text 構成(270M)、画像の試験は vision 構成(440M) |
| 実行環境 | uv が管理するプロジェクト専用の `.venv`。Python はまずインストール済みの 3.14 を使い、問題が出た場合だけ 3.12 に切り替える(5.5)。CLIもPythonで書く |
| 表示 | 原則として異なる5ノート(`--top` で増やせる)。各ノートは代表箇所1件と、同じノートの別の箇所を最大2件。見出し、抜粋、パスと行、Obsidianリンク、OSが記録した作成・更新日時(**本文の出来事の日付ではない**)を出す。日記はファイル名の日付も出す(**本文の出来事の日付とは限らない**) |
| 直近モード | 関連度上位10件の日記を候補に選び、ファイル名の日付が新しい順に最大5件出す |
| 処理場所 | 初回のパッケージとモデルの取得が済んだ後は、PC内だけで処理する(オフラインに固定)。外部の画像URLは取得しない |
| Vault | 読み取り専用。コード・索引・ログ・結果・モデルキャッシュは `C:\Users\yomor\dev\local-semantic-search` に置く |
| 索引の完全性 | 全対象が成功したときだけ完成版として保存する。失敗したら原因を記録して停止し、既存の正常な索引は壊さない。失敗したファイルを自動で除外しない |
| 速度 | 起動したままの連続検索で3秒以内を目標にする。初回(モデル読み込みを含む)とは分けて測る。超えても精度評価は続ける |
| **範囲外** | 回答文を生成するLLM、クラウド処理、Obsidianプラグイン、ベクトルDB製品、GGUF・量子化版・Ollama配布版、差分更新、新しいキーワード検索エンジン |

---

## 1. 経緯(第2版からの変更理由)

Ollama 0.40.2(Windows)で `ollama pull embeddinggemma-2:740m` と `:740m-bf16` を実行したところ、どちらも `this model requires MLX support, but the MLX runtime is not available` で失敗した。このため、Ollamaをやめて、Google公式チェックポイントを `sentence-transformers` で直接動かす構成に変えた。Node.js+Ollamaのコードは未実装だったため、すべて削除した。

---

## 2. 事実の区分

### 2.1 公式資料で確認済み(2026-10-11 閲覧)

| # | 事実 | 出典 |
|---|---|---|
| F1 | sentence-transformers で `SentenceTransformer("google/embeddinggemma-2")` として読み込める。使わない部分は `config_kwargs` で読み込みから外せる。text だけなら `{"vision_config": None, "audio_config": None}`(270M)、text と画像なら `{"audio_config": None}`(440M) | [開発者ガイド](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/)、[モデルカード](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) |
| F2 | どの構成も同じチェックポイントから読み込むので、ベクトル空間は1つ。例として、270Mで埋め込んだ検索語を全体構成の文書と照合できる、とある | 開発者ガイド |
| F3 | 検索語の形式は `SearchQuery` = `task: search result \| query: `。文書の形式は `title: {title} \| text: {content}` で、タイトルが無ければ `title: none`。`prompt_name="Document"` は `title: none` を付けるので、タイトル付きの文書には使わない。接頭辞はテキストにだけ付ける | モデルカード、公式リポジトリの `config_sentence_transformers.json` |
| F4 | 画像は `model.encode({"image": パス})` のように、モダリティ名をキーにした辞書で渡す。プロンプトは付けない | 開発者ガイド |
| F5 | 推論は `bfloat16` か `float32` で行い、`float16` は使わない(NaNや劣化したベクトルを、エラーを出さずに返すため)。大半のCPUを含め、bfloat16が安全な環境以外では `float32` を使う。公式例: `model_kwargs={"torch_dtype": dtype}` | モデルカード、公式チェックポイント |
| F6 | 出力は768次元(Matryoshkaで512/256/128にもできる)。`normalize_embeddings=True` で単位ベクトルになる。モジュール構成に `Normalize` が含まれている | 開発者ガイド、`modules.json` |
| F7 | 文脈長は全モダリティ共通で8,192トークン。画像は既定で1枚280トークン | モデルカード |
| F8 | sentence-transformers は **6.1.0以上**が必要(それ未満は複数モダリティ入力の順序を無視する)。チェックポイントは transformers 5.18.0.dev0 / sentence-transformers 6.1.0 で保存されている | `config_sentence_transformers.json` |
| F9 | リポジトリはゲートなし(Apache-2.0)。2026-10-06時点のコミットは `914f7f89142e33e77833254d9c9b90c3cef7303b`。ファイルは `model.safetensors` 1,488,915,288バイト(BF16、740M全体)、`tokenizer.json` 32,170,510バイト、`tokenizer.model` 4,689,013バイト、ほかに小さい設定ファイルで、**初回のダウンロードは合計約1.53GB** | [Hub API](https://huggingface.co/api/models/google/embeddinggemma-2) |
| F10 | `HF_HOME` でキャッシュの場所を変えられる。`HF_HUB_OFFLINE=1` にすると Hub へのHTTP通信を一切せず、キャッシュに無いファイルはエラーになる。環境変数は `huggingface_hub` を import した時点で読まれる | [huggingface_hub 環境変数](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) |
| F11 | `uv sync` でプロジェクトの `.venv` が作られ、`uv.lock` に正確な版が固定される。`.python-version` はプロジェクトの既定のPythonになる。`uv run` は実行前にロックファイルと環境を確認する | [uv プロジェクト](https://docs.astral.sh/uv/guides/projects/) |
| F12 | PyPI の最新版は sentence-transformers 6.1.0(`transformers>=5,<6`、`torch>=2.2` を要求。`image` extra は `transformers[vision]`)、transformers 5.19.0、torch 2.14.1 | PyPI |

### 2.2 ユーザーが確認済み

| # | 事実 |
|---|---|
| U1 | Markdownは Notes 72件、Notion 2,389件、dailynotes 110件で計2,571件。dailynotesはすべて有効な日付の `yyyy-MM-dd.md` |
| U2 | attachmentsの画像は367件。対象ノートから参照される実在画像は Notes・Notion 108件、dailynotes 249件で、重複なしの計357件 |
| U3 | Notes・Notionには、未解決の画像埋め込み参照517件と、外部URLの画像参照10件がある(dailynotesを含む3フォルダ全体の値ではない) |
| U4 | Ollama配布版は、1章のとおりMLXエラーで取得できなかった |

### 2.3 このPCで確認済み(パッケージやモデルは取得していない)

| # | 事実 |
|---|---|
| P1 | uv 0.12.19。uv管理の CPython 3.14.7 がインストール済み(3.13.15もある)。uv のパッケージキャッシュは `C:\Users\yomor\AppData\Local\uv\cache` |
| P2 | Windows標準OCR(`Windows.Media.Ocr`)の日本語エンジンが追加インストールなしで使える。ただし文字ごとに半角スペースを入れる。PowerShell 5.1 では `.ps1` をBOM付きUTF-8で保存する必要がある(第2版で自作画像を使って確認) |
| P3 | 10章のコードのうち、依存パッケージが要らない部分を確認した。Python 3.14.7(`uv run --no-project`)で構文チェックと `test_lss.py` を実行し、11件成功、numpyが必要な1件はスキップ。ほかに `--help`、名前の検査、パッケージが無いときのエラー表示を確かめた。**モデル・torch・numpy を使う部分は一度も動かしていない** |
| P4 | 出力をリダイレクトすると、Windowsの既定(cp932)では `—` などの文字を表示できずに落ちる。これをテストで見つけたので、`lss.py` で標準出力をUTF-8にした |

### 2.4 実行時に検証する(未確認なので断定しない)

| # | 検証項目 | どこで | 判定 |
|---|---|---|---|
| V1 | Python 3.14 で依存関係が解決でき、torch の Windows 用wheelが入るか | 段階0(人間が `uv sync`) | 成功すれば3.14のまま。失敗したら5.5 |
| V2 | リリース版の transformers 5.19 が `embedding_gemma2` を読み込めるか(チェックポイントは 5.18.0.dev0 で保存) | 段階0 `verify` | 読み込みの成否 |
| V3 | `model_kwargs={"torch_dtype": torch.float32}` で、全パラメータが float32 になるか | `verify`(`load_model` が検査) | dtype の集合が `{float32}` |
| V4 | `local_files_only=True` と `HF_HUB_OFFLINE=1` で、通信せずに読み込めるか | `verify`、Wi-Fiを切った実行 | 成功（2026-10-11、ユーザー確認） |
| V5 | `m.tokenizer`(プロセッサの場合は中のトークナイザー)で数えたトークン数が、実際の処理経路(`m.tokenize`)と一致するか。上限を超えると実際の経路で黙って切られるか | `verify` | `counter_matches_real_path`、`real_path_truncates` |
| V6 | 自前の文書形式が、`Document` プロンプトの結果と一致するか(余計な接頭辞が付いていないこと) | `verify` | cos > 0.9999 |
| V7 | 768次元、有限、ノルム1。日本語の簡単な組で、関連する文の類似度が高いか | `verify` | 各項目 |
| V8 | text構成とvision構成で、同じテキストのベクトルが一致するか(F2の実測) | `verify --images` | 最小cos ≥ 0.9999 なら「互換」 |
| V9 | 画像の内容がベクトルに反映されるか(API・動作の確認) | `verify --images` | 別の2画像はcos < 0.999、同じ画像はcos > 0.9999、画像と文字のみもcos < 0.999 |
| V10 | 日本語の質問で、目的の画像のリンク元ノートが見つかるか(実用性。V9とは別) | 段階3 | 8.6の比較表 |
| V11 | CPUでの読み込み時間、索引作成の速度、連続検索の時間、メモリ使用量 | 段階0〜3 | 実測値を記録する |
| V12 | 日本語での検索精度、直近モードの使い勝手、キーワード検索との差 | 段階2 | Semantic Searchはユーザー受入済み（2026-10-11）。キーワード比較は省略 |
| V13 | OneDriveの「オンラインのみ」ファイルの有無 | 段階0 `preflight` | 0件であること |
| V14 | Obsidian上のVault名が `vault` か | 段階1 | `vault` とリンク先をユーザー確認済み（2026-10-11） |
| V15 | `torchvision` などの追加パッケージが、画像処理に必要か | 段階3 | 足りなければ停止し、人間が追加 |

---

## 3. 構成

### 3.1 ファイル(すべてVaultの外)

```
C:\Users\yomor\dev\local-semantic-search\
  PLAN.md           本書
  pyproject.toml    依存は sentence-transformers[image] と transformers の2つだけ
  .python-version   3.14(問題時のみ 3.12)
  uv.lock           人間の uv sync で作られる。手で編集しない
  .gitignore        .venv / .hf-home / data / __pycache__
  lss.py            すべての処理(事前確認・取得・検証・索引・検索・画像)
  test_lss.py       モデルとVaultを使わないテスト(unittest)
  ocr.ps1           段階3のOCR案(Windows PowerShell 5.1、BOM付きUTF-8)
  questions.txt     段階2の5問
  .venv\            uv が作る
  .hf-home\         Hugging Face のキャッシュ(モデル約1.53GB)
  data\             索引・ログ・評価結果。ノート本文を含むので外部に送らない
    index\<名前>\meta.json, chunks.json, vectors.npy
    images\links.json, candidates.txt, sample.txt, queries.txt, expected.txt, ocr.jsonl
    logs\build.jsonl, search.jsonl, build-failure-<名前>.json
    results\eval-*.md, stage*.md
    verify.json
```

ユーザー領域に残るもの: uv のパッケージキャッシュ(`C:\Users\yomor\AppData\Local\uv\cache`)と、uv が管理するPython本体。プロジェクトを消しても、これらは残る(`uv cache clean` で消せる)。

### 3.2 通信とオフライン
- 通信するのは、人間が行う `uv sync`(パッケージの取得)と `uv run --frozen lss.py download`(モデルの取得)の2つだけ。
- それ以外はすべて `uv run --frozen --offline lss.py …` で実行する。`--offline` はuvの通信を止め、`--frozen` はロックファイルを更新しない。`lss.py` は `download` 以外では `HF_HUB_OFFLINE=1` にしてから Hugging Face のライブラリを読み込み、`local_files_only=True` でモデルを開く。キャッシュに無いファイルを暗黙に取りに行くことはなく、エラーになる。
- 段階0で一度、人間がWi-Fiを切った状態で `verify` を実行し、本当に通信なしで動くことを確かめる。

---

## 4. 実行手順書(上から順に1行ずつ進める)

**人** = 人間、**AI** = Luna。AIの行は、Lunaに13章の指示文を渡すと、次の人間の行まで進んで止まる。

| # | 担当 | やること | 終わった目安 |
|---|---|---|---|
| 1 | 人 | Lunaに**指示文13.1**を渡す | — |
| 2 | AI | 10章のファイルを作成する。依存なしで動く確認(`preflight`、テストの一部)を行って停止する | `data/results/stage0a.md` |
| 3 | 人 | `stage0a.md` を読む。オンラインのみのファイルがあれば扱いを決める | 決めた |
| 4 | 人 | パッケージを取得する: `uv sync`(作業ディレクトリで実行) | `.venv` と `uv.lock` ができる。失敗したら出力を保存して5.5へ |
| 5 | 人 | モデルを取得する(約1.53GB): `uv run --frozen lss.py download` | 「取得先」と「合計 1.5x GB」が表示される |
| 6 | 人 | Lunaに**指示文13.2**を渡す | — |
| 7 | AI | テストと `verify` を実行する(オフライン)。条件を満たせば段階1(50件の速度試験)まで進んで停止する | `stage0b.md`、`stage1.md` |
| 8 | 人 | Wi-Fiを切って `verify` を実行し、確認後にWi-Fiを戻す | `verify: OK`（2026-10-11、ユーザー確認済み） |
| 9 | 人 | `stage1.md` のObsidianリンクを1つ開いて確認する | Vault名 `vault` とリンク先を確認済み（2026-10-11） |
| 10 | 人 | Lunaに**指示文13.3**を渡す | — |
| 11 | AI | 全件の索引を作り、評価ファイルを出して停止する | `eval-full-*.md` |
| 12 | 人 | Semantic Searchの使い勝手を確認する。必要な場合だけキーワード検索と比較する | Semantic Searchは受入済み。キーワード比較はユーザー判断で省略（2026-10-11） |
| 13 | 人 | Lunaに**指示文13.4**を渡す | — |
| 14 | AI | 集計(`stage2.md`)と、画像のリンク表・候補15件の作成を行って停止する | `stage2.md`、`candidates.txt` |
| 15 | 人 | 段階3に進むなら、候補から約10件を選んで `sample.txt` に書き、検索語を `queries.txt` と `expected.txt` に書く | 3ファイルを書いた |
| 16 | 人 | Lunaに**指示文13.5**を渡す | — |
| 17 | AI | 画像の検証(V9)、OCRと画像埋め込みの索引作成、実用性の比較(V10)を行って停止する | `stage3.md` |
| 18 | 人 | 357件すべてを処理するか決める | 画像埋め込み索引を357件作成済み（2026-10-11） |

AIが**しないこと**: `uv sync`・`uv add`・`uv lock`・`download` などのパッケージやモデルの取得、Pythonのインストール、Vault内のファイルの作成・変更、外部URLへのアクセス、`--offline` を外した実行、失敗したファイルの自動除外、類似度だけで「メモなし」とすること。

---

## 5. 段階0: 環境とモデル

### 5.1 段階0a(AI、パッケージ不要)

```powershell
cd C:\Users\yomor\dev\local-semantic-search
[Console]::OutputEncoding = [Text.Encoding]::UTF8
uv run --no-project --no-python-downloads --python 3.14 lss.py preflight
uv run --no-project --no-python-downloads --python 3.14 python -m unittest -v
```

- `preflight` は、Vaultの件数(72 / 2,389 / 110、画像367)、日付でない日記の数、OneDriveの「オンラインのみ」ファイルの数を出す。読むのはファイル一覧と属性だけで、本文は読まない。
- テストはnumpyが要る1件がスキップされ、残りがすべて成功すること。
- 件数が違う場合や、オンラインのみのファイルがある場合は、報告して止まる。

### 5.2 人間による取得(手順4・5)

```powershell
cd C:\Users\yomor\dev\local-semantic-search
uv sync
uv run --frozen lss.py download
```

### 5.3 段階0b(AI、オフライン)

```powershell
uv run --frozen --offline python -m unittest -v
uv run --frozen --offline lss.py verify
```

`verify` は次を確認し、`data/verify.json` に書く。
- Python・sentence-transformers・transformers・torch の版。オフラインになっていること。
- キャッシュのリビジョンが `REVISION` と一致すること。取得したファイルの合計サイズ(ダウンロード量)。
- text構成の読み込み時間、パラメータ数、float32の重みの見積もり(パラメータ数 × 4バイト。**ダウンロード量とは別物**)。
- `SearchQuery` の接頭辞、768次元、有限、ノルム1。日本語の簡単な組での類似度の順序。
- 自前の文書形式と `Document` プロンプトが一致すること(V6)。
- 自前のトークン数の数え方が実際の処理経路と一致すること。上限を超えると実際の経路では切られること(V5)。1,200字の文書のトークン数。
- 1,200字の文書16件を埋め込む速度。

### 5.4 次へ進む判断基準
- 必須: `verify: OK`(`japanese_order_ok`、`counter_matches_real_path`、`real_path_truncates`、dim、文書形式の一致)。
- 必須: 手順8のWi-Fiなしの `verify` が成功する(段階2に進む前までに確認できればよい)。
- 記録のみ: 読み込み時間、速度、メモリの見積もり。

### 5.5 Python 3.14 で問題が出た場合
1. 失敗したコマンドと出力全文を `data/results/py314-failure.md` に保存する(`uv sync` の失敗なら人間が、`verify` の失敗ならAIが保存する)。
2. 原因が3.14に固有か(例: wheelが無い、`requires-python` の不一致、3.14でだけ起きる例外)を、AIが判断して報告する。3.14に固有でない問題(例: transformersがモデルを知らない)なら、3.12に変えても直らないので変えない。
3. 3.14に固有なら、人間が次を実行し、手順4からやり直す。
   ```powershell
   uv python install 3.12
   uv python pin 3.12
   uv sync
   ```

---

## 6. 段階1: 50件の速度試験

**速度と動作の試験であり、精度は判定しない。** 50件の結果をVault全体の精度とみなさない。

対象(`SAMPLE_PLAN`): 各フォルダをサイズ順に並べ、最短から最長まで等間隔に選ぶ。Notes 10件、Notion 30件、dailynotes 10件。選んだファイルは `data/index/sample50-files.txt` に残る。

```powershell
uv run --frozen --offline lss.py build --name sample50 --sample
uv run --frozen --offline lss.py search --index sample50 --eval questions.txt --repeat 2
```

`--repeat 2` で、1つのプロセスの中で5問を2周し、10回の検索時間を測る。評価ファイルは1周目の結果から作る。続けて対話モードで2〜3問を試し、表示を確認する(`uv run --frozen --offline lss.py search --index sample50`。行頭に `@recent` を付けると直近モード、空行で終了)。

計測項目:
- 索引作成: フォルダ別の件数、チャンク数、最大文字数、最大トークン数、追加分割の数、読み込み・分割の時間、モデル読み込み時間、埋め込み時間、chunks/秒。
- 検索: 準備完了までの時間(Python起動、索引、モデル読み込み)、10回それぞれの時間と内訳(埋め込み、順位付け)。
- 推定: 全件の時間 = フォルダ別の平均チャンク数 × 件数 ÷ chunks/秒。

次へ進む判断基準:
- 50件すべてで索引作成に成功している。全チャンクが `MAX_TOKENS` 以下(`max_chunk_tokens`)。
- 検索がエラーなく動き、表示項目が揃っている。人間がObsidianリンクを確認済み。
- 速度: 10回すべてが3秒以内なら合格。超えた場合も**段階2(精度評価)は止めない**。内訳(検索語の埋め込みか、順位付けか)と、高速化の候補(スレッド数、`truncate_dim`、索引の分割)を報告する。高速化に取り組むかどうかは、段階2の結果を見て人間が決める。
- 人間が全件の推定時間を見て承認している。

---

## 7. 段階2: 全件の精度評価（キーワード比較は任意）

```powershell
uv run --frozen --offline lss.py build --name full
uv run --frozen --offline lss.py search --index full --eval questions.txt --repeat 2
```

`questions.txt`:
```
@recent こどものスマホで何か作業したという直近の履歴を出して
ロシアがウクライナに戦争をしかけたのはなぜ？
中国の半導体に関する最近の動向は？
プライム効果って覆ったんだっけ？
プロスペクト理論って結局どういうことなんだっけ？
```

1問目は通常モードと直近モードの両方を出す。評価ファイルには、質問ごとに次の4つが入る。「メモ有無」(初期値は未確定)、キーワード検索の比較欄、通常モードの表、1問目だけは直近モードの表。

人間による評価は任意。キーワード比較を希望する場合は、普段どおりObsidianで検索してから使った語、試行回数と所要時間、見つけた関連ノートを記入する。Semantic Searchの判定とメモ有無を記入する場合も、**スコアだけで「なし」にしない**。直近モードの表では、関連の薄い日記が混ざった件数も見る。評価しなかった欄は空欄・未評価のままにする。

2026-10-11: ユーザーはSemantic Searchがよく機能すると判断し、5問のキーワード検索との比較を省略した。`stage2.md` と評価ファイルの未記入欄はそのまま保持している。

AIの集計(`stage2.md`):

| 質問 | メモ有無 | 役に立つ | 少し関連 | 関係ない | 最初の「役に立つ」 | 直近: 役に立つ/関係ない | KW: 回数/分 | Semのみ | KWのみ |
|---|---|---|---|---|---|---|---|---|---|

「メモあり」なのに上位に無い、または「キーワード検索だけで見つかった」ノートについて、次の手順で調べる。そのノートが `chunks.json` にあるか確認する。`search --index full --top 50` で順位とスコアを調べる。原因の候補を報告する。調整や索引の作り直しはしない。「未確定」の質問は上位候補とスコアをそのまま示す。実測前に閾値は設けない。

判断基準:
- 全件の索引作成に成功している。速度は段階1と同じ扱い(超えても評価は続け、内訳を報告する)。
- 「メモあり」の質問で上位に「役に立つ」があるか、無い場合はその原因が特定されている。
- キーワード検索との比較はユーザーが希望する場合に実施する。
- 最終判断は、人間が「日常の調べ物に使えるか」で決める。

---

## 8. 段階3: 画像の小規模試験(約10件)

### 8.1 リンク表(AI)
```powershell
uv run --frozen --offline lss.py links
```
対象は3フォルダの全Markdown。構文は `![[x]]`、`[[x]]`、`![](x)`、`[](x)` の4つ。`|サイズ` と `#見出し` は外し、表の中の `\|` に対応し、Markdownリンクはパーセントエンコードを戻す。画像拡張子が付いたものだけを対象にする。パス付きの参照は完全一致、パスなしはファイル名がちょうど1件一致したときだけ解決し、それ以外は未解決として記録する。外部URLは記録するだけで取得しない。期待値との照合は次のとおり(一致しなくても規則は緩めず、内訳を報告する)。

| 項目 | 期待値 |
|---|---|
| 解決した画像: Notes・Notion / dailynotes / 両方 / 計 | 108 / 249 / 0 / 357 |
| Notes・Notionの未解決の埋め込み参照 / 外部URL | 517 / 10 |
| dailynotesの未解決・外部URL | 既知の値なし(記録のみ) |

### 8.2 候補と選定
1. AIが、Notes・Notion・dailynotes の各リンク元から最大5件、計15件の候補を選ぶ(png/jpgでサイズの大きいものを優先)。`data/images/candidates.txt` に `画像<TAB>リンク元フォルダ<TAB>リンク元ノート` で書く。
2. 人間が、文字が読めるものを約10件(各フォルダから最低2件。足りなければあるだけ)選んで `sample.txt` に書く。
3. 人間が、画像内の文字を手掛かりにした検索語を `queries.txt` に、対応を `expected.txt`(`検索語<TAB>画像`)に書く。

### 8.3 画像がベクトルに反映されるかの確認(V9。実用性とは別)
```powershell
uv run --frozen --offline lss.py verify --images "C:\Users\yomor\OneDrive\Desktop\vault\attachments\<sample1>" "C:\Users\yomor\OneDrive\Desktop\vault\attachments\<sample2>"
```
`image_content_reflected` と、text構成とvision構成の互換性(`text_vision_compatible`、V8)を記録する。

### 8.4 2つの方法で索引を作る
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ocr.ps1 -ListFile data\images\sample.txt -OutFile data\images\ocr.jsonl
uv run --frozen --offline lss.py images --mode ocr --name images-ocr
uv run --frozen --offline lss.py images --mode vec --name images-vec
```
- OCR: 文字間のスペースを除き、`title: {リンク元ノート} > 画像 {名前} | text: {OCR文字列}` として text構成で埋め込む。OCRエラーが1件でもあれば停止する。文字の無い画像は `empty` に記録する。
- 画像埋め込み: vision構成で `{"image": パス}` を接頭辞なしで埋め込む。
- どちらの方法でも、(画像, リンク元ノート)ごとに1チャンクを作る。`file` はリンク元ノートなので、結果にはノートが出て「画像経由: …」と表示される。
- text構成の索引(`full`)と vision構成の索引(`images-vec`)を一緒に検索できるのは、`text_vision_compatible` が true の場合だけ。それ以外はコードが拒否する。

### 8.5 実用性の評価(V10)
```powershell
uv run --frozen --offline lss.py search --index full --eval data\images\queries.txt
uv run --frozen --offline lss.py search --index full,images-ocr --eval data\images\queries.txt
uv run --frozen --offline lss.py search --index full,images-vec --eval data\images\queries.txt
```

### 8.6 比較表と判断
| 検索語 | 期待する画像 | リンク元ノート | テキストのみ | +OCR | +画像埋め込み |
|---|---|---|---|---|---|

各セルには期待ノートの順位を書く(5位までに無ければ「圏外」)。テキストのみでは圏外で、どちらかの方法で5位以内に入る検索語が複数あれば、その方法で357件に広げることを提案する。最終判断は人間が行う。

---

## 9. Luna向け実装仕様

### 9.1 定数(`lss.py` の先頭)

| 定数 | 値 | 意味 |
|---|---|---|
| `MODEL_ID` / `REVISION` | `google/embeddinggemma-2` / `914f7f89…7303b` | 公式リポジトリと、変わらないコミット |
| `CONFIGS` | text: vision・audio なし / vision: audio なし | `config_kwargs` |
| `QUERY_PROMPT` | `SearchQuery` | 検索語の接頭辞 |
| `PROMPT_VERSION` | `q:SearchQuery\|d:title-text\|v1` | 入力形式を変えたら上げる |
| `DIM` | 768 | 出力次元 |
| `MAX_CHARS` / `MAX_TOKENS` / `MIN_SPLIT_CHARS` | 1200 / 2048 / 100 | 分割の上限 |
| `BATCH` | 16 | encode のバッチサイズ |
| `SAMPLE_PLAN` | Notes 10 / Notion 30 / dailynotes 10 | 段階1 |
| `TOP` / `OTHERS` / `RECENT_CANDIDATES` | 5 / 2 / 10 | 表示の上限 |
| `HF_HOME` | `.hf-home` | モデルのキャッシュ |

### 9.2 CLI
4章の各コマンドと `lss.py` 冒頭の docstring のとおり。`search` は起動したまま入力を待つ(空行で終了)。`--eval ファイル --repeat N` で、評価ファイルの作成と、N周分の計時を行う。`--top N` で表示ノート数を変え、`--candidates M`(M ≥ N)で直近モードの候補数を変える。

### 9.3 索引の形式
- `meta.json`: `format`(3)、`complete`(true)、`model`、`revision`、`config`(text/vision)、`dtype`、`prompt_version`、`max_tokens`、`count`、`dim`、作成時の情報と計時。
- `chunks.json`: `file`、`title`、`headings`、`line`、`part`、`parts`、`text`、`tokens`、`birth`、`mtime`。画像チャンクだけ `kind`(image-ocr / image-vec)と `image` を持つ。
- `vectors.npy`: float32 の `(count, 768)`、各行は正規化済み。
- 検索時は、モデル、リビジョン、入力形式、次元を照合し、1つでも違えば拒否する。text構成とvision構成を混ぜるのは、互換性の確認が済んでいる場合だけ。

### 9.4 分割と切り捨ての防止
1. 見出しで分割する(frontmatterは1節にし、コードブロック内の `#` は見出しにせず、`C#` を壊さない)。
2. 1,200字以下に分ける(段落 → 文 → 強制分割。文字は捨てない)。
3. `title: {ノート名 > 見出し} | text: {本文}` の**トークン数を、モデルのトークナイザーで数える**。2,048を超える文書は、本文を半分以下に分けて数え直す(有限回で終わる)。`max_seq_length` も2,048に設定するが、そこに頼らず、上限を超える文書をモデルに渡さない。

### 9.5 失敗時の扱い
- 読めない・UTF-8でない・読み取り中に変更されたファイルがあれば、モデルを読み込む前に、全件の一覧を記録して停止する。
- 埋め込み中の例外、NaN・Inf、次元の違い、ノルムが1でないベクトルが出たら停止する。
- 停止したときは、`data/logs/build-failure-<名前>.json` と `build.jsonl` に原因を記録し、索引は保存しない。保存は `.tmp` に書いてから入れ替える(既存の索引は `.old` に退避し、入れ替えに成功してから消す)。
- 再実行の規則: 原因を直さずに同じ条件で再実行しない。コードや設定が原因ならAIが直す(9章の定数を変えたら `PROMPT_VERSION` を上げる)。Vault側が原因なら人間に報告する。一時的な原因(メモリ不足で他のアプリを閉じた、など)の場合だけ、同じ条件で1回まで再実行できる。2回目も失敗したら人間に報告する。

### 9.6 順位付け
- 通常: 内積(=コサイン類似度)の降順に並べる。同点はパスの降順(日記では新しい日付が先)。異なるノートを `--top` 件、各ノートは代表箇所1件と別の箇所を最大2件。
- 直近: 日付が有効な日記だけを通常と同じ規則で `--candidates` 件選び、日付の新しい順に `--top` 件。候補内の関連度の順位を表示し、「関連する日記が候補数より少なければ、関連の薄い日記も入る」と表示する。

### 9.7 受け入れ条件(Lunaが守ること)
1. 10章のコードを基準にする。変えた箇所は理由を `CHANGES.md` に書く。
2. 書き込みは `out()` を通す(`write_text`、`append_jsonl`、`remove_tree`、`save_index`)。確認方法: `Select-String -Path lss.py -Pattern 'write_text\(|open\(|np.save|rename|unlink|rmtree'` の該当箇所が、すべて `out()` を通った経路であること。
3. 依存パッケージ、設定項目、抽象化を足さない。追加のパッケージが必要になったら止まって人間に頼む。
4. `uv run --frozen --offline python -m unittest` が成功すること。
5. 未確認の項目(2.4)を「確認済み」と書かない。実測した値をそのまま報告する。

---

## 10. コード全文

10.1〜10.7 をそのまま保存する(`uv.lock` は人間の `uv sync` で作られる)。確認した範囲は P3 のとおりで、モデル・torch・numpy を使う部分は未実行。

### 10.1 pyproject.toml
```toml
[project]
name = "local-semantic-search"
version = "0.2.0"
requires-python = ">=3.12"
dependencies = [
    # 6.1.0 未満は複数モダリティ入力の順序を無視する(公式 config_sentence_transformers.json の requirements)
    "sentence-transformers[image]>=6.1.0",
    # 公式チェックポイントは transformers 5.18.0.dev0 で保存されている。リリース版での読み込みは段階0で確認
    "transformers>=5.19.0",
]

[tool.uv]
package = false
```

### 10.2 .python-version
```
3.14
```

### 10.3 .gitignore
```
.venv/
.hf-home/
data/
__pycache__/
```

### 10.4 lss.py
```python
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
            images = [s.strip() for s in (DATA / "images" / "sample.txt").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
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
```

### 10.5 test_lss.py
```python
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
```

### 10.6 ocr.ps1(BOM付きUTF-8で保存)
```powershell
# ocr.ps1 — 段階3(代替案): Windows標準OCR(Windows.Media.Ocr)で画像内の文字を読む。追加インストール不要。
# Windows PowerShell 5.1 で実行する(PowerShell 7 では WinRT 型を直接読めない)。
#   powershell -ExecutionPolicy Bypass -File ocr.ps1 -ListFile data\images\sample.txt -OutFile data\images\ocr.jsonl
param(
  [Parameter(Mandatory)][string]$ListFile,   # 1行1画像。例: attachments/foo.png
  [Parameter(Mandatory)][string]$OutFile,
  [string]$VaultRoot = 'C:\Users\yomor\OneDrive\Desktop\vault',
  [string]$Lang = 'ja'
)
$ErrorActionPreference = 'Stop'
$outFull = [IO.Path]::GetFullPath($OutFile)
$vaultFull = [IO.Path]::GetFullPath($VaultRoot).TrimEnd('\') + '\'
if ($outFull.ToLower().StartsWith($vaultFull.ToLower())) { throw "出力先がVault内です: $outFull" }

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Await($op, [Type]$type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $null = $t.Wait(-1); $t.Result }

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Lang))
if ($null -eq $engine) { throw "OCR言語 '$Lang' が使えません。人間が日本語のOCR機能を追加してください。" }
$max = [Windows.Media.Ocr.OcrEngine]::MaxImageDimension

$out = New-Object System.Collections.Generic.List[string]
foreach ($rel in Get-Content -Encoding UTF8 $ListFile) {
  $rel = $rel.Trim().TrimStart([char]0xFEFF)
  if (-not $rel) { continue }
  $abs = [IO.Path]::GetFullPath((Join-Path $VaultRoot ($rel -replace '/', '\')))
  $stream = $null
  try {
    if (-not (Test-Path -LiteralPath $abs)) { throw "ファイルがありません: $abs" }
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($abs)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    if ($decoder.PixelWidth -gt $max -or $decoder.PixelHeight -gt $max) { throw "画像が大きすぎます($($decoder.PixelWidth)x$($decoder.PixelHeight) > $max)" }
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $text = ($result.Lines | ForEach-Object { $_.Text }) -join "`n"
    $out.Add((@{ image = $rel; text = $text; width = $decoder.PixelWidth; height = $decoder.PixelHeight } | ConvertTo-Json -Compress))
  } catch {
    $ex = $_.Exception; while ($ex.InnerException) { $ex = $ex.InnerException }
    $out.Add((@{ image = $rel; text = ''; error = $ex.Message } | ConvertTo-Json -Compress))
  } finally {
    if ($stream) { $stream.Dispose() }
  }
}
[IO.File]::WriteAllLines($outFull, $out, (New-Object Text.UTF8Encoding $false))
Write-Output "OCR完了: $($out.Count) 件 → $outFull"
```

### 10.7 questions.txt
```
@recent こどものスマホで何か作業したという直近の履歴を出して
ロシアがウクライナに戦争をしかけたのはなぜ？
中国の半導体に関する最近の動向は？
プライム効果って覆ったんだっけ？
プロスペクト理論って結局どういうことなんだっけ？
```

---

## 11. リスクと対処

| リスク | 対処 |
|---|---|
| Python 3.14 用の torch などが無い | 5.5の手順で、3.14に固有の問題の場合だけ3.12にする |
| リリース版の transformers がモデルを読めない | `verify` で止まって報告する。dev版を入れるかどうかは人間が判断する(今回の計画ではしない) |
| CPUでの全件作成が長い・検索が3秒を超える | 段階1で実測して推定する。超えても精度評価は続け、内訳を見てから高速化を判断する |
| メモリ不足 | `verify` の見積もりと実測を報告する。他のアプリを閉じて1回だけ再実行できる |
| 黙った切り捨て | 自前でトークン数を数えて分割する。`verify` で、数え方が実際の経路と一致することを確かめる |
| OneDriveのオンラインのみファイル | `preflight` で検出して止まる |
| Windowsでキャッシュのシンボリックリンクが作れない | huggingface_hub が警告を出し、ファイルを実体として置く(容量が増えることがある)。動作には影響しない |

---

## 12. 未解決の判断事項

| 事項 | 誰が・いつ |
|---|---|
| Vault側の原因で索引作成が失敗した場合に、直すか対象を改訂するか | 人間・発生時 |
| 3秒を超えた場合に高速化するか | 人間・段階2の後 |
| 直近モードの候補数10が適切か | 人間・段階2の後 |
| 画像をOCRと画像埋め込みのどちらで扱うか、357件すべてを処理するか | 人間・段階3の後 |

---

## 13. Lunaへの指示文

### 13.1 ファイル作成と段階0a(手順1)
```text
あなたは C:\Users\yomor\dev\local-semantic-search で作業する実装担当です。PLAN.md を全文読み、書かれていない設計判断を足さないでください。
1. PLAN.md 10章の pyproject.toml、.python-version、.gitignore、lss.py、test_lss.py、ocr.ps1(BOM付きUTF-8)、questions.txt を、そのまま作業ディレクトリに保存する。
2. 5.1のコマンドを実行する(uv run --no-project --no-python-downloads --python 3.14 …)。パッケージのインストール、uv sync、モデルの取得はしない。
3. 結果を data/results/stage0a.md にまとめて停止する。次の3つを書くこと。preflight の各件数と既知値の比較、オンラインのみのファイル数、テストの結果(成功・スキップ件数)。
件数が違う場合や、オンラインのみのファイルがある場合は、内容を報告して人間の判断を待ってください。
```

### 13.2 段階0bと段階1(手順6。人間が uv sync と download を済ませた後)
```text
人間が uv sync と uv run --frozen lss.py download を済ませました。PLAN.md 5.3〜6章を実行してください。コマンドはすべて uv run --frozen --offline で実行します。
- 5.3: テストと verify を実行する。5.4の必須条件を満たさなければ、data/verify.json と出力をそのまま報告して停止する。Python 3.14に固有の問題なら5.5に従い、data/results/py314-failure.md を書いて停止する。
- 6章: sample50 の索引を作り、search --eval --repeat 2 で10回計時する。対話モードで2〜3問(1つは @recent)を試す。
- 索引作成が失敗したら、9.5の規則に従う。
- data/results/stage0b.md と stage1.md を作って停止する。stage1.md には6章の計測項目と、判断基準を1つずつ満たしたかどうかを書く。3秒を超えた場合は、内訳と高速化の候補も書く。
- 人間に頼むことを最後に書く。手順8・9はユーザーが2026-10-11に確認済み。
守ること: Vaultは読み取り専用。--offline を外さない。パッケージやモデルを取得しない。未確認のことを確認済みと書かない。
```

### 13.3 段階2の索引と評価ファイル(手順10)
```text
PLAN.md 7章の build --name full と search --eval --repeat 2 を実行してください。
索引作成が成功するまで評価に進まず、失敗したら9.5に従ってください。
評価ファイルのパス、索引作成の記録(件数・チャンク数・追加分割・時間)、10回の検索時間と内訳を報告して停止してください。
最後に、未実施の人間判断だけを書いてください。キーワード比較はユーザーが希望した場合だけ行います。
```

### 13.4 段階2の集計と段階3の準備(手順13)
```text
評価結果をPLAN.md 7章に従って集計し、未記入欄はそのままで data/results/stage2.md を作ってください。キーワード比較はユーザーが希望した場合だけ扱ってください。
ユーザーがSemantic Searchを受け入れて比較を省略した場合は、その判断を記録し、個別の判定を作らないでください。
続けて links を実行し、8.1の期待値と照合したうえで、8.2の手順1の候補15件を data/images/candidates.txt に書いて停止してください。
```

### 13.5 段階3(手順16)
```text
sample.txt、queries.txt、expected.txt を書きました。PLAN.md 8.3〜8.6 を実行してください。
画像が反映されるかの確認(8.3)と、日本語の質問で見つかるかの評価(8.5)は、別々に報告してください。
OCRエラー、画像処理に必要なパッケージの不足、text構成とvision構成の非互換で止まった場合は、そのまま報告して人間の判断を待ってください。
8.6の比較表を含む data/results/stage3.md を作って停止してください。
```

---

## 14. 変更履歴

| 版 | 内容 |
|---|---|
| 第1版 | Node.js+Ollama で計画 |
| 第2版 | dailynotes の追加、直近モード、全件成功時だけ保存、キーワード検索との比較、画像357件 |
| 第3版 | Ollama配布版がMLXエラーで使えなかったため、Google公式チェックポイント+sentence-transformers+uv(Python)に変更した。Node・Ollama関連のコードと手順(`/api/embed`、`truncate`、`keep_alive`、digest、模擬サーバーなど)はすべて削除した。CLIを、起動したまま検索できる形にした。トークン数をモデルのトークナイザーで数え、エラーを待たずに分割するようにした。そのため、エラー文の判定と、同じ条件の再実行をコードで拒否する仕組み(`conditionsKey` / `--retry-same`)は削除し、9.5の運用規則にした。画像は、画像埋め込みとOCRを比較することにした |
