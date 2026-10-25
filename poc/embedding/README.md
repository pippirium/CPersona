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
