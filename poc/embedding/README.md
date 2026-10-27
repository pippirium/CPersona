# ローカル埋め込み比較PoC

Jina v5-text-nanoとEmbeddingGemma 2のdense検索を、同じ日本語データで比較する。
本番の記憶DB・MCP設定は変更せず、上流へのPR・pushは行わない。
既存CPersonaのハイブリッド検索全体や品質ゲートの比較は、この段階の対象外。

## 実行

リポジトリ直下から実行する。

```sh
poc/embedding/setup.sh  # 環境を再作成する場合。現在のPCでは準備済み
poc/embedding/run.sh --model both
```

既定はCPU・float32。M1 Proでの互換性を優先する。
`--device mps`は試験用で、CPUと別に評価すること。
モデルは初回にHugging Faceからダウンロードしてキャッシュする。
モデルのリビジョンはコード内で固定。Jinaは発行元のカスタムコードを利用する。
結果は`results/`のJSONに保存。Recall@1、Recall@5、MRR、処理時間と検索順位を含む。
時間には初回のウォームアップが含まれ、厳密な性能ベンチマークではない。

付属の10件は実行確認用の合成データで、モデルの採用判断には使わない。
実データを評価するときは、次の形式のJSONを作り、指定する。

```sh
poc/embedding/run.sh --dataset /absolute/path/evaluation.json --model both
```

```json
{
  "records": [{"id": "decision-1", "text": "保存した決定と理由"}],
  "queries": [{"text": "その決定を思い出す質問", "relevant": ["decision-1"]}]
}
```

入力はローカル推論し、評価データを外部APIへ送らない。
該当記憶なしの拒否判定・訂正の解釈・最終回答精度はこのdense検索指標では測れない。

## 画像の動作確認（明示指定時のみ）

```sh
poc/embedding/run.sh --model gemma --image /absolute/path/example.png \
  --image-query '赤い警告のある設定画面'
```

指定した画像だけを埋め込み、質問との類似度を数値で出力する。
画像やBase64は出力しない。画像未指定時はvision encoderを読み込まない。
1画像の確認は画像検索精度の評価ではない。

## CPersona本体との関係

`.venv-poc`にはこのcheckoutのCPersonaもeditableでインストールする。
ただし本PoCはモデル単体の比較であり、MCPへの自動接続や本番モデル切り替えはしない。
次の段階で、複製した評価DBと専用embedding adapterを使い、
Node/Block・品質ゲート・reconstructを含む比較を行う。

## このPCでの動作確認（2026-10-25）

Apple M1 Pro / 32GB、Python 3.12、CPU / float32。
合成10問で両モデルの検索と結果JSON保存が完了した。

| モデル | Recall@1 | Recall@5 | MRR | 質問処理時間の中央値 |
|---|---:|---:|---:|---:|
| Jina v5-text-nano | 0.70 | 1.00 | 0.850 | 約57ms |
| EmbeddingGemma 2 | 0.80 | 1.00 | 0.867 | 約67ms |

10問のみの実行確認であり、Gemmaの優位性を示す結果ではない。
PDFからレンダリングした1画像のGemma埋め込みも768次元・有限値で完了。
画像テストはコマンドへの明示指定時のみ実行し、出力は数値と形状のみ。
CPersona本体のimportも確認した。本番MCP設定や記憶DBの変更はしていない。

## Git

originは個人fork、upstreamは原本。ローカルでupstreamのpush URLを無効化済み。
仮想環境・結果は`.git/info/exclude`で除外する。
PR作成や自動push処理は含まない。

## CP-02: 3モデルの固定条件比較

Jina/BGE-M3/EmbeddingGemma 2の独立dense比較は `compare.py` と
[COMPARISON-PROTOCOL.md](COMPARISON-PROTOCOL.md) を使用する。
このCLIはMCPサーバーへ登録せず、本番記憶DB・embedding backend設定を変更しない。
既存の10問は日本語smokeとして残し、公開LMEB_SciFactの固定revisionの全test
（1,748 passages / 188 queries）を別集計する。全LMEBや一般日本語品質の代用にはしない。

比較用の隔離環境には `requirements-compare.txt` をインストールする
（既存lockの依存＋pyarrow 26.0.0）。モデル取得は推論と分離し、各固定SHAの
ローカルsnapshotを `--model-root` で明示する。Jinaは発行元の固定custom codeを使う。

```sh
python -m poc.embedding.compare prepare \
  --lmeb-files <downloaded-parquet-path-map.json> \
  --japanese poc/embedding/sample.json --out <isolated-input-dir>
python -m poc.embedding.compare run --model jina \
  --model-root <snapshot-at-preregistered-sha> \
  --dataset <isolated-input-dir>/japanese_smoke.json \
  --dataset <isolated-input-dir>/lmeb_scifact.json --out <result-dir>/jina.json
# 同じ条件で --model bge / gemma を別プロセスで実行
python -m poc.embedding.compare compare \
  --result <result-dir>/jina.json --result <result-dir>/bge.json \
  --result <result-dir>/gemma.json --out <result-dir>/paired.json
```

parquet path-mapは `mteb/LMEB_SciFact` @
`432cf48c879c2de792720d5c0af6900312ebe7b6` の
`LMEB_SciFact-{corpus,queries,qrels}/test-00000-of-00001.parquet` をキー、
ダウンロード先の絶対パスを値とするJSON。path-mapと原文データはコミットしない。
結果はID・順位・指標・条件・hash・資源量のみで、query/corpus本文やパスを含めない。
通常の `evaluate.py` / run.sh の動作は変更しない。

### 全LMEB 22データセットの再現

`full_lmeb.py` は上記 SciFact/japanese smoke の別条件で、公開 `KaLM-Embedding/LMEB` の固定snapshot全体を評価する。実行前に評価用隔離環境へ `requirements-compare.txt` を入れ、Hugging Faceから次のrevisionを取得する。入力JSONは専用scratchに作り、本文はGitへ追加しない。

```sh
hf download KaLM-Embedding/LMEB task_instructions.json --repo-type dataset \
  --revision f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0
python -m poc.embedding.full_lmeb prepare \
  --snapshot <KaLM-Embedding-LMEB-snapshot>/f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0 \
  --task-instructions <pinned-task_instructions.json> \
  --out <isolated-input-dir>
```

Jina/BGE/Gemmaをそれぞれ別プロセスで実行する。モデルrootのbasenameは固定SHAと一致させ、結果とチェックポイントは隔離ディレクトリに保存する。
各推論呼出しは最大1,024件の外側チャンクに分け、チャンクごとにMPS同期して進捗を記録する。同じMacで報告された大規模コーパスのMPS待ちを抑える実行上の調整で、内側batch 8、入力・モデル・ランキング条件は維持する。結果にはこの実行条件とコードrevisionを記録する。

```sh
python -m poc.embedding.full_lmeb run --model jina \
  --model-root <snapshot>/8a7f00aac812071b69403df470f1038ec85f8925 \
  --input-dir <isolated-input-dir> --task-instructions <pinned-task_instructions.json> \
  --checkpoint-dir <checkpoint-dir> \
  --out <result-dir>/jina.json.gz
# bgeとgemmaも各固定SHAのsnapshotを指定して同じ条件で実行
python -m poc.embedding.full_lmeb compare \
  --result <result-dir>/jina.json.gz --result <result-dir>/bge.json.gz \
  --result <result-dir>/gemma.json.gz --out <result-dir>/paired.json
```

`compare` は`paired.json`と同じbasenameのMarkdown reportも書き出す。生成物には公開ID・順位・指標・条件だけを含め、正規化したquery/corpus本文は含めない。

準備時にQASPERの同一ID・同一本文の重複だけを統合する。コーパスにない正解IDと公開candidate scene外の正解IDはRecall/NDCGの分母に残す。Jinaには公式LMEBの指示文を付け、BGE/Gemmaは公式main設定どおり指示文を付けない。レポートはsource欠落数、candidateの不足数、512-token truncation、全193 query taskの集計方法を記載し、SciFact結果やLoCoMo最終回答精度とは混ぜない。workerはオフライン推論・MPS/bf16・最大512 tokens・corpus/query quality batch 8を使い、1 taskあたり最大20問のbatch-one latency sampleを別に測る。チェックポイントはモデル・入力・コード・実行条件が一致する場合だけ再利用する。
### 2026-10-28 全LMEB実行の途中経過

Jinaは全103グループを完了し、BGE-M3はLongMemEval group 8のチェックポイント保存後に依頼どおり停止しました。Gemmaの全件評価と3モデルの対応比較は未実施です。数値、fingerprint、停止後の再開手順は[チェックポイント報告](../../docs/research/cp02-full-lmeb-checkpoint-2026-10-28.ja.md)を参照してください。入力本文・モデル・チェックポイント・クエリ別結果はこのGitブランチへ含めていません。
