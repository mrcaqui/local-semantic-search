# 変更記録

- `lss.py images` に `--all` を追加。既存の `sample.txt` と `images-vec` の動作を保ったまま、`--mode vec --all` はリンク表全体を対象にする。OCRとの誤用を拒否し、全リンク画像が357件でなければ索引作成を中止する。全件モードは `sample.txt` を読まない。`images-vec-all` など別名で保存でき、小規模評価索引と既存 `full` を置き換えない。
- 理由: ユーザー承認に基づき、段階3で評価した画像埋め込みをリンク済み357画像へ拡張するため。
- 検証: 全357画像の読み取りとPillowデコード、全件選択・件数異常停止・既存sample動作のユニットテスト。

- EXPECTED_FILES.Notes と計画上の現在件数を、ユーザー確認に合わせて69から72へ更新（Markdown合計2,571件）。full索引は変更していない。PLAN.mdとstage0a/0b/1/2記録に2026-10-11のユーザー確認（Wi-Fiなしverify、Obsidianリンク、Semantic Search受入、キーワード比較省略）を追記。
