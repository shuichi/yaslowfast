# yaslowfast

PyTorch 2.14.1で、SlowFast-R50をスクラッチ学習する動画の二値分類パッケージです。動画を事前にJPEGフレームへ変換してWebDatasetのtar shardに保存し、単一GPUまたは`torchrun`によるDDPで学習します。各動画を、互いに排他的な2クラスのいずれかに分類します。

モデルは、PyTorchVideoの`slowfast_r50`のモデル定義を移植・再構成したPyTorch実装です。モデル本体はこのリポジトリの`src/yaslowfast/models.py`に実装されており、PyTorchVideoをインストールしたり、実行時に呼び出したりする必要はありません。PyTorchVideo、torchvision、fvcoreへの依存は、実行・開発・テスト・エクスポートのいずれにもありません。学習用の重みはランダム初期化します。

移植したコードの由来と適用ライセンスを明示するため、元の著作権表示、Apache License 2.0の本文、変更内容の記録を保持しています。詳細は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)に記載し、ライセンス本文とともに配布パッケージにも同梱しています。

## ライセンス

このプロジェクトは **Apache License 2.0** の下で公開しています。全文は[LICENSE](LICENSE)を参照してください。

PyTorchVideo由来部分の著作権表示と変更内容は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、Apache License 2.0本文は[LICENSE](LICENSE)に保持しています。これらはwheel・sdist・配布用ソースZIPに同梱します。

## 最短手順（H100 × 8、1ノード）

下記の「環境の準備」を済ませ、仮想環境を有効にした状態で、リポジトリ直下から実行します。positive・negativeそれぞれの動画を格納したtarを用意し、入力パスを置き換えてください。出力先の`datasets/20261026`は未作成または空にします。

```sh
yaslowfast-prepare \
  --positive data/positive_sample.tar --negative data/negative_sample.tar \
  --output datasets/20261026

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  -m yaslowfast.train --config configs/slowfast_r50.yaml \
  --dataset datasets/20261026/dataset.json \
  --output-dir runs/h100x8 --device cuda
```

splitと入力動画のmanifestは前処理で自動生成します。`data/`のサンプルtarは各クラス1動画のため、生成結果はtrainのみです。このサンプルで前処理・検査は実行できますが、上の学習には空でないvalも必要です。既定の分割では各クラス2動画以上、testも用意する場合は各クラス3動画以上を用意してください。

mean/stdは前処理で自動計算され、学習時に自動適用されます。標準設定はGPUあたりバッチ8・worker 8（8GPU全体ではバッチ64・worker 64）、精度は自動選択でBF16、196 epochです。チェックポイントと学習ログは`runs/h100x8`に保存されます。

## 環境の準備

Python 3.12を基準にしています。パッケージの対応範囲はPython 3.11〜3.14です。動画の前処理と動画ファイルからの推論には、PATH上のFFmpegが必要です。学習時はJPEGを読み込むため、動画デコーダーを起動しません。

```sh
uv sync --python 3.12 --extra dev
source .venv/bin/activate
ffmpeg -version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

CUDA環境では、GPU・ドライバーに合うPyTorch 2.14.1のwheelを[公式インストール案内](https://pytorch.org/get-started/locally/)に従って選んでください。専用indexを使う場合はuvのindex設定も管理し、再度`uv sync`したときに意図しないwheelへ戻らないようにします。GPU学習前には上記コマンドでCUDAが利用可能なことを確認してください。

ONNX書き出しとONNX Runtimeによる一致検証も使う場合は、Python 3.12の環境にexport extraを追加します。ONNXは1.17.0、対象アプリに合わせてONNX Runtimeは1.20.1に固定しています。ONNX 1.17.0の公式wheelはPython 3.12までなので、学習側のPython対応範囲とは別に、ONNX書き出しにはPython 3.12を使用してください。`torch.export`形式だけなら追加依存は不要です。

```sh
uv sync --locked --python 3.12 --extra dev --extra export
```

以下のコマンドは、リポジトリをカレントディレクトリとし、仮想環境を有効にした状態で実行します。YAML中の相対パスは、YAMLファイルの場所ではなくコマンドのカレントディレクトリを基準に解決します。

## 配布用ZIPの作成

uvでPythonを選択・管理して、配布用のソースZIPを作成できます。スクリプトには[uvが対応するインラインメタデータ](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)でPython 3.11〜3.14と追加依存なしを宣言しています。事前の`uv sync`は不要で、ZIP作成のためにPyTorchなどをインストールすることもありません。Gitも不要です。

```sh
# 同梱するファイルを確認（ZIPは作成しない）
uv run tools/make_dist.py --dry-run

# dist/yaslowfast-0.1.0-source.zip を作成（バージョンはpyproject.tomlから取得）
uv run tools/make_dist.py

# 同名のZIPを作り直す場合
uv run tools/make_dist.py --force

# 出力先を変更する場合（相対パスはカレントディレクトリ基準）
uv run tools/make_dist.py --output /tmp/yaslowfast-source.zip
```

ZIPには`yaslowfast-<バージョン>/`フォルダを作り、以下を同梱します。

- `src/yaslowfast/`内のPythonソース・型定義、`configs/`内のYAML、`tests/`内のPythonテスト
- `README.md`、`pyproject.toml`、`uv.lock`、`.python-version`、`.gitignore`
- `LICENSE`、`THIRD_PARTY_NOTICES.md`、`tools/make_dist.py`

同梱対象を明示する許可リスト方式です。`.venv/`やuvのキャッシュ、データセット、動画、学習結果、モデルの重み、既存ZIP、`.env`、Gitの履歴などは含めません。対象フォルダ内でもデータ・実行結果・仮想環境のディレクトリ、隠しファイル、キャッシュ、シンボリックリンクを除外します。`.gitignore`を変更してもZIPの対象は変わりません。新しい配布ファイルが必要になったら、[tools/make_dist.py](tools/make_dist.py)の`REQUIRED_FILES`／`SOURCE_TREES`を更新し、`--dry-run`で確認してください。

`uv.lock`は依存ライブラリのバージョンを記録したファイルで、ライブラリ本体ではないため同梱します。受け取った側はZIPを展開してそのフォルダに移動し、次のコマンドで環境を再作成します。依存ライブラリの取得にはネットワーク接続が必要です。

```sh
uv sync --locked --python 3.12 --extra dev
source .venv/bin/activate
```

必要に応じて`--extra export`も追加してください。FFmpegの準備やGPU環境の設定は「環境の準備」を参照してください。データセットと学習済みモデルは別途用意します。

## 入力tarと自動split

`--positive`と`--negative`に、それぞれのクラスの動画を格納したtarを指定します。tar直下・サブディレクトリ内の動画を収集し、クラスIDは`positive=0, negative=1`に固定します。`group_id`の指定は不要で、新しい出力にも含めません。入力tarは変更しません。

動画拡張子は`.mp4`、`.avi`、`.mov`、`.mkv`、`.webm`、`.m4v`、`.mpg`、`.mpeg`、`.mts`、`.m2ts`、`.ts`、`.wmv`、`.flv`に対応します。ディレクトリ、macOSの補助ファイル、動画以外のファイルは読み飛ばします。各tarに動画が1本以上必要です。不正なパス、リンク・特殊ファイル、同じメンバー名の重複は拒否します。

フレーム抽出・窓生成の前に、positive・negativeそれぞれを元動画単位でtrain／val／testへ割り当てます。既定の比率は80/10/10です。同じ動画から作る窓はすべて同じsplitに入ります。動画内容のhashを使うため、同じ入力動画と`--seed`なら、tar内の格納順やtarの保存先に依存せず同じsplitになります。必要なら`--split-ratios 0.8 0.1 0.1`で比率を変更できます。

既定比率で動画が少ない場合はtrainを優先し、次のように割り当てます。3動画以上でも、少数時には80/10/10と厳密には一致しません。

| 各クラスの動画数 | そのクラスの割り当て |
|---|---|
| 1本 | trainのみ |
| 2本 | train・valに1本ずつ |
| 3本以上 | train・val・testに最低1本ずつ確保し、比率に近づける |

valが空でも前処理は完了し、現在の学習処理にはvalが必要な旨を表示します。学習にはtrainが両クラスを含み、valが空でないことが必要です。各splitの件数とクラス別件数は`dataset.json`に保存します。

tarをまたぐ場合を含め、内容hashが同じ動画は重複として拒否します。別の内容の動画に登場する同一人物や同一撮影セッションのまとまりは、自動では識別しません。

## 動画をWebDatasetへ変換

```sh
yaslowfast-prepare \
  --positive data/positive_sample.tar --negative data/negative_sample.tar \
  --output datasets/20261026 \
  --fps 15 \
  --short-side 320 \
  --window-frames 64 \
  --window-stride 32 \
  --jpeg-quality 95 \
  --shard-size-mb 512 \
  --seed 42

yaslowfast-prepare --verify datasets/20261026/dataset.json
```

`--output`で指定したディレクトリをそのまま使用し、日付ディレクトリを追加で作ることはありません。既存のファイルがある出力ディレクトリは上書きしません。読めない動画は既定で処理を中断し、`errors.json`へ記録します。デコードできない動画を除外して続ける場合に限り`--skip-errors`を指定します。tar内の不正な構造や内容が重複する動画は、この指定でも拒否します。除外の結果、データセットに1クラスしか残らない場合は完成データとして公開しません。shardへの書き込み失敗は、部分的な動画を公開しないよう常に中断します。

FFmpegで時刻を0起点に揃えてからfps変換するため、可変フレームレートの入力も秒単位でサンプリングします。fps変換で失った細かな動きは学習時に復元できません。素早い動きを扱う場合は保存fpsと入力フレーム数を合わせて検討してください。保存画像は縦横比を維持して短辺を320pxへresizeします。`jpeg-quality`は1〜100をFFmpeg MJPEGのqscale 31〜2へ対応付ける値です。

既定では、15 fpsの64フレームを1窓とし、32フレームずつずらして保存します。動画末尾は最後のフレームで終わる窓を追加し、同じ末尾窓を二重生成しません。短い動画は保存時に末尾を反復しますが、`valid_frames`に実フレーム数を記録します。学習・評価時は実フレーム部分から時間切り出しを行ってから不足分を反復し、短い動きが静止フレームだけにならないようにします。

```text
datasets/20261026/
  manifest.jsonl
  dataset.json
  errors.json
  train-000000.tar
  val-000000.tar
  test-000000.tar

# 1つのtarに、以下のまとまりを多数格納
<video-id>_<window-id>.json
<video-id>_<window-id>.000000.jpg
<video-id>_<window-id>.000001.jpg
...
```

`manifest.jsonl`は動画ごとの出典と割り当てを記録する自動生成ファイルです。`source_archive`（入力tarの絶対パス）、`source_member`（tar内のパス）、`sha256`、`video_id`、`label`、`class_id`、`split`を保存し、一時展開先のパスは残しません。これは処理の追跡用で、従来の`--manifest`への入力形式とは異なります。

`dataset.json`には`task: "binary_classification"`、`num_classes: 2`、クラス対応、前処理設定、動画と窓の件数、shardサイズ、SHA-256を保存します。train splitがある場合は、学習用フレームから計算したRGBのmean/stdも`normalization`に保存します。上の例のクラス対応は`{"positive": 0, "negative": 1}`です。以前に作成した2クラスのschema version 1データセットも読み込めます。

空のsplitにはtarを生成しません。tarは非圧縮で、1窓を途中で分割しません。512 MiBは目安で、tarのpaddingや1窓のサイズによって超える場合があります。重なる窓ではJPEGを重複保存するため、strideを小さくすると保存量も増えます。tarは一括展開せず、動画1本ずつ処理します。動画1本分の元ファイルと抽出JPEGを置く一時ディスク領域が必要です。これらの一時ファイルは各動画の処理後に削除します。

完成前には`.incomplete`マーカーを残し、全shardの作成とRGB統計の計算を終えてから`dataset.json`を公開します。マーカーがあるデータセットは学習できません。`--verify`は全shardのサイズ・hash・サンプル件数・クラス対応・全JPEGのデコードを確認します。現在の読み込み対象はローカルまたは共有ファイルシステム上のtarです。

## 従来のmanifest入力（互換用）

個別の動画ファイルを指定する従来の入力方式も使用できます。tar入力とは同時指定できません。`--video-root`と`--class-names`はこの方式専用です。

```sh
yaslowfast-prepare \
  --manifest manifests/videos.jsonl \
  --class-names positive negative \
  --output datasets/legacy
```

JSON配列、JSONL、タブ区切りの`.tsv`／`.txt`に対応します。必須項目は`video`と`label`です。クラス名は任意の文字列ですが、manifest全体で厳密に2種類必要です。既定ではクラス名のソート順でIDを0・1に割り当てます。`--class-names CLASS_0 CLASS_1`を指定すると、その順序を使用します。対応表は`dataset.json`とチェックポイントに保存されます。

```jsonl
{"video":"videos/person01_negative.mp4","label":"negative","split":"train","group_id":"person01"}
{"video":"videos/person02_positive.mp4","label":"positive","split":"train","group_id":"person02"}
{"video":"videos/person03_negative.mp4","label":"negative","split":"val","group_id":"person03"}
{"video":"videos/person04_positive.mp4","label":"positive","split":"val","group_id":"person04"}
```

JSON形式なら、このレコードを配列にします。TSVは`video`, `label`, `split`, `group_id`の順で、ヘッダーあり・なしの両方を受け付けます。区切りには実際のタブを使用してください。パス中の空白はそのまま扱えます。

| フィールド | 内容 |
|---|---|
| `video` | 動画ファイルのパス。相対パスはmanifestの親ディレクトリ基準。`--video-root`で基準を変更可能 |
| `label` | 動画全体のクラス名。指定した2種類のいずれか |
| `split` | 任意。`train`、`val`、`test`のいずれか |
| `group_id` | 任意。同じ人物・撮影セッションなど、分割を跨がせたくない単位 |

指定済みのsplitは維持します。未指定の動画は、動画／group単位でラベル集合を考慮して分割します。既定の比率は80/10/10で、小さいグループ集合では比率が厳密に一致しない場合があります。同じgroupの一部にsplitがあれば、残りにもそのsplitを適用します。

この方式に限り、既存の`group_id`指定を引き続き受け付けます。分割はフレーム抽出・窓生成より先に行います。同一パスの重複、同じgroupのsplit跨ぎ、内容hashが同じ動画の重複は拒否します。内容が異なる同一人物の動画は自動では判別できないため、必要な`group_id`を入力してください。学習には厳密に2クラスと、空でないtrain・val splitが必要です。両クラスの学習例がtrainに含まれることも検査します。val／testの個々のsplitは1クラスだけでも評価できますが、データセット全体のクラス対応は必ず2クラスです。

互換モードで既存アプリへ渡す用途では、ID 0を本物、ID 1を偽物に対応させて学習してください。上の例は`positive`が本物、`negative`が偽物を表す前提です。任意のクラス名を使う場合も、`--class-names`には「本物のラベル、偽物のラベル」の順に指定します。省略時は名前のソート順となり、`negative=0, positive=1`になります。1クラスまたは3クラス以上のmanifest、重複するクラス名、manifestにない名前の指定は拒否します。

## RGB正規化

`yaslowfast-prepare`は、train splitだけからRGBのmean/stdを自動計算し、`dataset.json`の`normalization`に保存します。学習時は既定の`data.normalization: auto`により、この値を自動で読み込みます。通常はデータ変換後にそのまま学習を開始でき、統計計算コマンドの追加実行やYAMLへの値の転記は不要です。

統計はRGB `[0,1]`の保存JPEGの全画素を対象とし、空間crop前に計算します。窓の重なりと短尺paddingで同じフレームを重複集計せず、val／test splitは使いません。評価専用などtrain splitが空のデータセットでは、統計の計算・保存を省略します。実際に学習へ適用したmean/stdはチェックポイントに保存され、評価・推論でも共用されます。学習再開時も値の一致を確認します。

統計情報がない既存データセットを`auto`で使う場合は、警告を出してYAMLの`data.mean`／`data.std`を使用します。これらも未指定ならmean=`[0.45,0.45,0.45]`、std=`[0.225,0.225,0.225]`を使います。既存データでも独自統計を使いたい場合は、新しい出力先へデータ変換をやり直すか、次のコマンドで統計を計算してください。

```sh
yaslowfast-stats --dataset datasets/20261026/dataset.json --output normalization.json
```

`yaslowfast-stats`は統計確認や手動設定用の独立したコマンドです。`dataset.json`や学習YAMLは変更しません。結果を手動で適用する場合は、YAMLの`data.normalization`を`config`にして、`data.mean`／`data.std`に計算結果を設定します。`auto`ではデータセットに保存された統計がYAMLのmean/stdより優先されます。例えば、固定値を明示的に使う設定は次のとおりです。

```yaml
data:
  normalization: config
  mean: [0.45, 0.45, 0.45]
  std: [0.225, 0.225, 0.225]
```

`yaslowfast-stats`に`--max-frames 10000`を指定すると、決まった順序の先頭10000フレームを使います。ランダムな全体標本ではありません。

## モデルと標準学習設定

[configs/slowfast_r50.yaml](configs/slowfast_r50.yaml)が標準レシピです。

| 項目 | 設定 |
|---|---|
| Backbone | Slow/FastのResNet-50、各段`[3,4,6,3]` |
| Fast入力 | `B × 3 × 32 × 224 × 224` |
| Slow入力 | `B × 3 × 8 × 224 × 224` |
| 時間解像度比 | α=4 |
| Fastチャネル幅 | Slowの1/8 |
| 分類特徴・出力層 | 2048+256=2304次元 → `Linear(2304, 2)` |
| パラメータ数 | 二値分類の標準R50は33,649,098 |
| 出力 | `[batch, 2]`のlogits。列0・1は保存したクラスIDに対応 |
| 損失・推論 | ラベル0・1のCrossEntropyLoss、推論はsoftmaxで2クラスの確率を計算 |
| 初期化 | Kaiming Conv、残差最終BN scale=0、分類headの重みstd=0.01 |
| Optimizer | SGD、momentum 0.9、weight decay `1e-4`、BatchNormのweight decayは0 |
| Schedule | 196 epoch相当、34 epoch相当のwarmup、cosine減衰 |

Slow入力はFast入力から`linspace(0,T-1,T/4).long()`で選択します。32フレームでは`[0,4,8,13,17,22,26,31]`で、単純な`::4`とは異なります。モデルのfusionも公式と同じ4か所です。

出力層は2クラスに固定しており、Python APIやCLIでも2以外のクラス数を拒否します。学習にはlogitsをそのままCrossEntropyLossへ渡し、推論時にsoftmaxを適用します。例えば上のクラス対応では`softmax(logits, dim=1)[:, 0]`が`positive`の確率です。動画単位では各窓の確率を平均し、確率が大きいクラスを選びます。同率の場合はID 0を選びます。

既定の`pooling: adaptive`は標準32/8フレーム・224px入力で公式poolingと一致し、異なる入力サイズも扱います。公式の固定poolingを使う場合は`pooling: reference`とし、32フレーム・224px以上を使用してください。224px以外では両poolingの平均化範囲が異なるため、同じ出力を保証しません。

学習時は保存窓から32フレームをランダム抽出し、短辺256〜320pxへresizeして224pxをrandom cropします。空間crop、左右反転、色変換のパラメータはクリップ内で共通です。評価では時間中心の32フレーム、短辺256px、center cropを使用します。15 fps保存時のFast入力は約2.13秒です。横反転や色変換はクラスの意味を保てる場合だけ設定で有効にします。

## 単一GPU・DDP学習

設定ファイルの既定パスは`data/dataset.json`です。以下では`--dataset`で今回の出力先に変更します。

単一GPU:

```sh
yaslowfast-train \
  --config configs/slowfast_r50.yaml \
  --dataset datasets/20261026/dataset.json \
  --output-dir runs/experiment01 \
  --device cuda
```

1ノード・4 GPU:

```sh
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m yaslowfast.train \
  --config configs/slowfast_r50.yaml \
  --dataset datasets/20261026/dataset.json \
  --output-dir runs/experiment01 \
  --device cuda
```

2ノード・各4 GPUでは、両ノードで同じコマンドを実行し、`YASF_NODE_RANK`だけを0と1に変えます。`192.0.2.10`は例示用なので、ノード0の到達可能なアドレスに置き換えてください。

```sh
YASF_NODE_RANK=0
torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank="$YASF_NODE_RANK" \
  --master-addr=192.0.2.10 --master-port=29500 \
  -m yaslowfast.train \
  --config configs/slowfast_r50.yaml \
  --dataset datasets/20261026/dataset.json \
  --output-dir runs/experiment01 \
  --device cuda
```

全ノードで同じコード・設定・データセットを使用し、同じパスからshardを読めるようにしてください。再開時のチェックポイントも全rankから読める必要があります。CUDAはNCCL、CPUの分散確認はGlooを使います。MPSは`--device mps`で単一プロセスの明示指定に対応しますが、このDDP構成では使いません。`auto`はCUDAが使えなければCPUを選びます。

### バッチサイズと学習率

`data.batch_size`は**GPUあたり・勾配蓄積前**のバッチサイズです。

```text
実効グローバルバッチ = data.batch_size × world_size × train.accumulation_steps
学習率 = train.learning_rate × 実効グローバルバッチ / train.reference_batch_size
```

後者は`scale_learning_rate: true`の場合です。既定では実効バッチ64で学習率0.1になります。例えば4 GPU、各4サンプル、4回の勾配蓄積なら実効バッチ64です。VRAMに合わせてバッチ数を調整し、大きく条件を変えた場合は収束も確認してください。自動スケーリングを止める場合は`scale_learning_rate: false`を指定します。

勾配蓄積はBatchNormが一度に見るバッチを増やしません。GPUあたりのバッチが小さい場合は`sync_batchnorm: true`を検討できますが、CUDA DDPが必要で、通信も増加します。SyncBatchNormでも蓄積ステップを跨いだ統計の同期は行いません。

`precision: auto`は対応CUDA GPUでBF16、それ以外のCUDAでFP16、CPUでFP32を選びます。FP16ではGradScalerを使用し、overflow時は全rankが同時に更新を見送ります。更新に成功したステップだけ学習率scheduleを進めます。`grad_clip_norm`、`label_smoothing`、`mixup_alpha`は任意設定です。

`weight_decay`は通常のパラメータに、`norm_weight_decay`はBatchNorm／SyncBatchNormのパラメータに適用します。既定値はそれぞれ`1e-4`と`0.0`です。

`compile`と`channels_last_3d`は既定で無効です。実際のGPU・入力形状で効果を比較して有効にしてください。`num_workers`、`prefetch_factor`、`shuffle_buffer`はrankごとの設定です。shuffle bufferには圧縮JPEGを保持しますが、多数のrank・workerで大きくするとCPUメモリ使用量も増えます。

### epochとデータサンプリング

学習はrank・worker・epochごとのseedで、shardを復元抽出する無限ストリームです。各shardを選んだら中の全サンプルを読み、圧縮状態のbufferでshuffleしてから必要なフレームだけをdecodeします。サンプルの再訪問は仕様です。

epochの規定update数に達するとworkerへ停止を通知し、先読み済みのbatchを学習に使わず読み切ってからworkerを終了します。学習中のサンプル順・epochごとのseed・FP16 overflow時の追加読み取りは変わりません。この終了処理の変更は設定やcheckpoint形式を変更しないため、従来のcheckpointから同じ設定で再開できます。

**1 epochは全データの重複なし一巡ではなく、一定回数のoptimizer更新です。** `steps_per_epoch: null`なら、`ceil(trainの窓数 / 実効グローバルバッチ)`を更新回数にします。明示的に`steps_per_epoch`を設定することもできます。全rankが同じ更新回数を実行するため、データ終了時刻の違いでDDPが待ち続けることを避けられます。

既定の`video_balanced: true`では、窓の多い長尺動画が学習を支配しないように、窓のlossへ`全窓数 / (全動画数 × その動画の窓数)`を掛けます。これは動画の寄与を揃えるもので、クラスの件数を揃える設定ではありません。`false`では窓を均等に扱います。

## チェックポイントと再開

学習ディレクトリには次を保存します。

| ファイル | 内容 |
|---|---|
| `latest.pt` | 最後に完了したepochのモデル・optimizer・scheduler・scaler・各rankのRNGなど |
| `best.pt` | 検証指標が最良のチェックポイント。既定の指標は動画Accuracy |
| `config.json` | 実行設定 |
| `environment.json` | PyTorch、device、precision、world size、実効バッチ、データfingerprintなど |
| `metrics.jsonl` | epochごとの学習・検証指標 |
| `validation.json` | 最後の検証結果と動画別予測 |

epoch途中で停止した場合は、最後に保存されたepochの続きから再開します。shuffle buffer、worker先読み、途中のmicrostepをその場で復元する機能はありません。既存の`latest.pt`がある出力先へ新規学習を開始するとエラーになります。

```sh
yaslowfast-train \
  --config configs/slowfast_r50.yaml \
  --dataset datasets/20261026/dataset.json \
  --output-dir runs/experiment01 \
  --device cuda \
  --resume runs/experiment01/latest.pt
```

再現性のあるepoch境界再開では、モデル・augmentation・worker数・batch・勾配蓄積・optimizer・schedule・precision・world size・PyTorchバージョンを揃えてください。異なる設定はチェックして拒否します。出力先やログ頻度は変更できます。同じ内容のデータセットを別の場所へコピーした場合は`--dataset`で変更できます。

途中まで実行して再開を確認する場合は、**最終予定の`epochs`を変更せず**に`--stop-after-epoch`を使います。例えば196 epochのscheduleを維持して10 epoch完了時点で停止するには次のようにします。

```sh
yaslowfast-train --config configs/slowfast_r50.yaml \
  --dataset datasets/20261026/dataset.json \
  --output-dir runs/experiment01 --device cuda --stop-after-epoch 10
```

再開時は同じコマンドから`--stop-after-epoch`を外し、`--resume runs/experiment01/latest.pt`を追加します。再現性の比較には`deterministic: true`も使用し、同じハードウェア・実行環境を維持してください。高速化設定や異なるデバイス間でbit単位の一致は保証しません。

データfingerprintは、shard hashを含む`dataset.json`を正規化して計算します。学習開始時に毎回全tarを再hashする方式ではありません。データのコピー後や破損が疑われる場合は`yaslowfast-prepare --verify`を実行してください。

## 評価と動画推論

### 分類済みの動画tarから混同行列を算出

positive・negativeに分類済みの動画tarを直接指定して、任意のチェックポイントを評価できます。リポジトリ直下から実行します。

```sh
yaslowfast-evaluate \
  --checkpoint runs/h100x8/best.pt \
  --positive data/eval_positive.tar \
  --negative data/eval_negative.tar \
  --device cuda \
  --batch-size 8 \
  --output results/best-evaluation.json \
  --report reports/best-evaluation
```

`--positive`と`--negative`は両方指定します。各tarに含まれる対応形式の動画をすべて評価し、train／val／testへの再分割は行いません。macOSの`._*`・`__MACOSX`やCSVなどの動画以外のファイルは読み飛ばします。`data/eval_positive.tar`と`data/eval_negative.tar`は各100動画です。各1動画の`positive_sample.tar`／`negative_sample.tar`は動作確認向けです。

チェックポイントのモデル構成、fps、JPEG品質、窓長・stride、評価時のリサイズ・中央クロップ、学習時のmean/stdをそのまま使用します。元の学習データセットが手元になくても実行できます。クラス名は`positive`と`negative`が必要ですが、IDはチェックポイントの`class_to_idx`から取得するため、IDの並びが逆の既存モデルも評価できます。

各時間窓のsoftmax確率を動画ごとに平均し、最大確率のクラスを予測とします。同率ならID 0を選びます。**動画1本を1件**として集計するため、長い動画も混同行列に二重加算されません。Accuracy、Balanced Accuracy、Macro F1、各クラスのPrecision／Recall／F1と件数、混同行列をコンソールへ表示します。positiveを陽性とし、表示順はpositive、negativeです。

| 正解 ＼ 予測 | positive | negative |
|---|---|---|
| positive | TP | FN |
| negative | FP | TN |

行が正解、列が予測です。JSONの`confusion_matrix`と確率配列は従来どおり**クラスID順**なので、読み取る際は`class_to_idx`を参照してください。分母が0になるPrecision／Recall／F1は0として扱います。`loss`／`window_loss`は実際の各窓のlogitsから計算したcross entropy、`video_loss`は動画平均確率から計算します。コンソールの先頭行には従来の要約JSONも出力します。動画ごとの進捗は標準エラー出力です。

`--report DIR`を指定すると、次のファイルを生成します。

- `DIR/index.html`: 日本語のHTMLレポート。色付きの混同行列（件数・行ごとの割合）、主要指標、誤分類・全動画の詳細、入力・チェックポイント・前処理・実行条件を表示します。ブラウザで直接開くことができ、インターネット接続やWebサーバーは不要です。
- `DIR/metrics.json`: 指標と動画ごとの予測結果を含む評価データ。
- `DIR/predictions.csv`: 元tar、tar内の動画名、正解・予測・確率などの動画別一覧。

レポートディレクトリは自動作成し、再実行時は同名のレポートファイルを更新します。`--report`を省略してもコンソールへ結果を表示し、`--output`のJSON（既定は`evaluation.json`）を保存します。HTMLには動画本体を埋め込まず、元tarとメンバー名で対象動画を識別します。

tar直接評価は単一プロセスです。`--device auto`はCUDAが使えればCUDA、それ以外はCPUを選び、Apple SiliconのMPSは`--device mps`で明示指定します。既定の精度はFP32で、必要なら`--precision`を指定できます。`--batch-size`は1動画内の時間窓のバッチ数です。FFmpegが必要で、元動画と抽出JPEGは1動画ずつ一時保存・削除します。`--dataset`、`--split`、`--num-workers`、`--allow-preprocessing-mismatch`との併用や`torchrun`による分散実行はできません。

空のtar、不正なtar構造、内容が重複した動画、デコードできない動画、不正なモデル出力はエラーにし、動画を黙って除外した部分評価は出力しません。低い精度そのものは実行エラーにせず、指標として報告します。汎化性能の評価には、学習やチェックポイント選択に使用していない動画を用意してください。入力tar間の内容重複は検査しますが、過去の学習データとの重複は自動検査しません。

### 前処理済みデータセットの評価と動画推論

WebDatasetの評価:

```sh
yaslowfast-evaluate \
  --checkpoint runs/experiment01/best.pt \
  --dataset datasets/20261026/dataset.json \
  --split test \
  --device cuda \
  --output results/test.json
```

testが空の場合は`--split val`を使用します。`--batch-size`と`--num-workers`は評価時だけ変更できます。複数GPUでの評価も、`torchrun ... -m yaslowfast.evaluate`に同じ引数を渡して実行できます。

この方式でも混同行列をコンソールに表示し、`--report DIR`でHTML・JSON・CSVレポートを生成できます。分散実行時のファイル出力はrank 0のみです。

評価は全窓を有限・重複なしで分配し、末尾バッチを落としません。rankごとに窓数が異なる場合や、担当shardがないrankにも対応します。各窓のsoftmax確率を元動画ごとに平均し、動画1本を1件としてAccuracy、macro F1、balanced accuracy、混同行列を計算します。`loss`／`window_loss`は窓単位の無重みcross entropy、`video_loss`は動画ごとに確率を平均した後の負の対数尤度です。

学習済みcheckpointと評価データのクラスID対応は一致させてください。fps、保存画像サイズ、窓長・stride、JPEG品質も検査します。意図的な条件変更に限り`--allow-preprocessing-mismatch`で許可でき、その差は評価JSONに記録されます。比較可能な評価には、学習時と同じ前処理を使用してください。

動画ファイルからの推論:

```sh
yaslowfast-predict \
  --checkpoint runs/experiment01/best.pt \
  --video videos/example.mp4 videos/another.mp4 \
  --device cuda \
  --batch-size 4 \
  --include-windows \
  --output results/predictions.json
```

チェックポイントに保存したfps・JPEG設定・窓生成・正規化を共用します。JSONにはラベル、全クラス確率、窓数、処理時間を出力します。`--include-windows`で窓別の確率も保存できます。このコマンドは単一プロセスで実行してください。

## 書き出しと速度測定

`torch.export`形式:

```sh
yaslowfast-export --checkpoint runs/experiment01/best.pt \
  --format torch --batch-size 1 --output exports/slowfast.pt2
```

ONNX形式（export extraが必要）:

```sh
yaslowfast-export --checkpoint runs/experiment01/best.pt \
  --format onnx --batch-size 1 --output exports/slowfast.onnx
```

通常モードは固定shapeの`slow`と`fast`の2入力、float32・NCTHW形式で書き出します。入力はRGB正規化と時間／空間サンプリング済みのテンソル、出力`logits`は2クラスのlogitsです。動画デコードや前処理そのものはグラフに含めません。クラス対応と正確な前処理仕様を隣接する`.pt2.json`／`.onnx.json`に保存します。書き出したartifactを再読み込みし、PyTorch eager出力との数値一致を確認してから公開します。

ONNXの既定opsetは18です。`--opset-version`の対応範囲は18〜21とし、生成したファイルの実際のopsetとIRバージョン（上限10）を検査します。これは[ONNX Runtime 1.20の対応範囲](https://onnxruntime.ai/docs/reference/compatibility.html)に合わせた制限です。ONNX 1.17.0のcheckerとONNX Runtime 1.20.1でのロード・実推論・PyTorch出力との比較に成功したファイルだけを公開します。重みをONNX本体に埋め込み、外部の重みファイルを必要としません。

通常・互換モードともに、保存前にサブグラフとローカル関数を含む全ノードの名前を検査し、無名・重複名を一意の名前に補正します。これにより、ORT 1.20.1の`GatherSliceToSplitFusion`が無名ノードから同名のノードを作り、モデルの読み込みに失敗する問題を防ぎます。補正はノード名のみを変更し、演算・重み・入出力名は維持します。Runtimeの最適化は有効のまま検証します。

以前に書き出したONNXは自動更新されません。`uv sync --locked --python 3.12 --extra dev --extra export`で依存環境を更新し、同じcheckpointから再エクスポートしてください。再学習は不要です。旧設定のopset 22は受け付けないため、既定の18または対応範囲内を指定してください。

### 既存SlowFast入出力との互換モード

既存の2入力・二値分類インターフェースでONNXファイルを差し替える場合は、`--compat-mode slowfast`を指定します。このモードはONNX形式のみで利用でき、専用処理は`src/yaslowfast/compat_slowfast.py`にまとめています。

互換モードの出力順は、モデルの列0を出力列0、モデルの列1を出力列1へそのまま渡す固定仕様です。既存アプリは列0を本物、列1を偽物として読むため、学習時からこのID順にしてください。ラベル名の指定、特定の名前の判定、自動的な列の入れ替えは行いません。

```sh
yaslowfast-export --checkpoint runs/experiment01/best.pt \
  --format onnx --compat-mode slowfast \
  --output exports/slowfast-compat.onnx
```

このREADMEのデータ準備例では`--class-names positive negative`により、本物を表す`positive`をID 0、偽物を表す`negative`をID 1に割り当てています。`genuine`／`fake`など別の名前でも、同じID順で学習すれば同じコマンドでエクスポートできます。互換モードは名前からクラスの意味を推測せず、学習済みの出力列をそのまま使用します。

以前の例に従い`negative=0, positive=1`で学習したcheckpointなど、出力列の意味が逆の重みを使う場合は、学習側でIDと重みの対応を修正する必要があります。`dataset.json`やcheckpointのラベル名・対応表を書き換えるだけでは、学習済みの出力層の意味は変わりません。互換エクスポートはこの不一致を自動修正せず、ラベル名による検出も行いません。

| 項目 | 互換モードの仕様 |
|---|---|
| Slow入力 | `input_slow`、float32、`[N, 3, 8, 430, 295]` |
| Fast入力 | `input_fast`、float32、`[N, 3, 32, 430, 295]` |
| 入力画像 | RGB、NCTHW、値域`[0,1]`、mean/stdによる正規化前 |
| 出力 | `output`、float32、`[N, 2]`。モデルの列0・1を維持し、アプリは列0＝本物、列1＝偽物として扱う |
| 可変次元 | バッチ`N`のみ。時間・画像サイズは固定 |
| 必要ファイル | 重みと前処理を含む単一の`.onnx`。隣接JSONは説明用 |

グラフ内部では、Fast 32枚にcheckpointの評価設定と同じ短辺リサイズ・中央クロップ・mean/std正規化を行います。通常の224pxモデルでは、設定どおりにリサイズして224×224へ切り出します。正規化には新モデルの学習時の値を使用します。呼び出し側では、RGBの`/255`以外の正規化を追加しないでください。

Slowのフレーム選択は、学習時に合わせてFastの`[0,4,8,13,17,22,26,31]`から再生成します。従来の`[0,4,8,12,16,20,24,28]`を含む`input_slow`も同じ名前・型・形状で受け取れますが、計算には使用しません。Fastが推論内容を決める入力です。エクスポート後も両入力を保持し、従来の2入力の呼び出しでRuntimeが受理することを検証します。checkpointのモデルは`num_frames: 32`、`alpha: 4`である必要があり、8フレームのsmoke設定は対象外です。

互換モードでは動的バッチを検証します。Softmaxは含まず、出力0と出力1の大小をそのまま比較できます。隣接JSONの`class_to_idx`は、checkpointに保存された対応をそのまま記録します。これは学習元の情報であり、互換モードの処理分岐には使用しません。固定の出力列対応は`output_class_indices: [0, 1]`として記録します。通常の学習、WebDatasetの形式、通常モードの前処理済み入力の仕様は変わりません。

この互換性は、既存の呼び出しが同じ入出力で新モデルを実行できることを対象とします。入力画像を作る前のROI切り抜きや変形、フレーム抽出で失われた情報は復元できません。撮影・切り抜き・時間間隔をそろえた代表データで分類精度を確認してください。実際のAndroid／iOSアプリのRuntimeやExecutionProviderでの動作と速度は、Python版Runtime 1.20.1のCPU検証とは別に確認する必要があります。

### 速度測定

```sh
yaslowfast-benchmark \
  --checkpoint runs/experiment01/best.pt \
  --device cuda --precision bf16 --batch-size 1 \
  --warmup 10 --iterations 50 \
  --output results/benchmark.json
```

`--compile`と`--channels-last-3d`で最適化の有無を比較できます。学習前の二値分類モデルを測る場合は`--checkpoint`の代わりに`--config configs/slowfast_r50.yaml`を使います。CPU計測では`--device cpu --precision fp32 --threads 1`などを指定します。

計測対象はdevice上の合成テンソルに対するmodel forwardです。動画デコード、JPEGデコード、前処理、device転送を含みません。p50／p95 latency、clips/sec、CUDAの場合はpeak memoryを保存します。`predict`の処理時間とは計測範囲が異なります。

SlowFast-R50はConv3dを使う約3,365万パラメータの二値分類モデルです。エクスポートの成功は、モバイル端末のバックエンドが高速に実行できることを意味しません。ExecuTorch、Core ML、ONNX Runtime等での演算対応、精度、実機latencyは対象端末ごとに確認してください。量子化・蒸留・端末別の変換やアプリ組み込みは、この実装には含めていません。

## CPUでの接続確認とテスト

[configs/smoke.yaml](configs/smoke.yaml)は、8フレーム・32px・細いチャネル・各段1ブロックの小さいモデルです。前処理から学習までの接続確認専用で、SlowFast-R50の学習レシピや性能測定の代わりにはなりません。標準設定で作成したデータも読み込めます。以下の学習例には、trainに両クラス、valに1動画以上を含むデータセットを使用してください。

```sh
yaslowfast-train --config configs/smoke.yaml \
  --dataset datasets/20261026/dataset.json --output-dir runs/smoke --num-threads 1

torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29501 -m yaslowfast.train \
  --config configs/smoke.yaml --dataset datasets/20261026/dataset.json \
  --output-dir runs/smoke-ddp --device cpu --num-threads 1

pytest -q
ruff check src tests tools
ruff format --check src tests tools
```

テストは、tar入力とmanifest自動生成、splitの再現性と少数動画の割り当て、不正tar・重複動画の拒否、実FFmpeg動画の前処理、shard検査、短尺padding、複数workerの評価分配、rank別サンプリング、モデル初期化と勾配、動画単位の指標、分散評価、checkpoint／schedule／RNGの復元、exportなどを対象にしています。ONNXテストにはexport extra、動画テストにはFFmpegが必要です。プロセスや通信を制限したsandboxでは、DataLoader共有メモリやGlooのローカル通信に実行許可が必要な場合があります。

macOSでは`torchrun --standalone`がホスト名の解決によって起動待ちになる場合があるため、上のCPU例ではloopbackアドレスを明示しています。Glooのインターフェース選択で問題が出る場合は、macOS上でコマンドの前に`GLOO_SOCKET_IFNAME=lo0`を指定してください。Linuxでは実際のインターフェース名を使用します。

2026-10-26に、macOS arm64・Python 3.12.13・PyTorch 2.14.1で以下を確認しました。

| 検証 | 結果 |
|---|---|
| 全自動テスト | ラベル名に依存しない出力列固定、互換モード・ONNX制限・非有限出力時の既存ファイル保護を含む153件成功 |
| 互換モードの出力列固定 | 任意の日本語ラベル、positive／negative、genuine／fakeの名前やID対応を変えても、既知のlogits `[2, -3]` がONNXから同じ列順で返ることを確認 |
| Ruff lint / format | 成功 |
| 元動画→JPEG→WebDataset→学習→評価・直接動画推論 | 成功、評価と推論の確率が一致 |
| epoch境界の中断・再開 | 旧形式の2クラスチェックポイントからの再開を含め、単一プロセスでモデル・optimizer状態が連続学習とbit単位で一致 |
| 2プロセスGloo＋各rank 1 DataLoader worker | 学習・再開成功、連続学習とモデルがbit単位で一致 |
| 分散評価 | 空rank、rank間の窓数差、buffer同期、重複窓の拒否を確認 |
| 小型モデルの合成2クラス動画へのfit | 60更新後、検証2動画のAccuracy 1.0・video loss約0.00013。実データの精度評価ではありません |
| 初回実装時の公式R50との数値比較 | 二値専用化前の400クラス版で、32/8フレーム・224pxの全5段の両経路とlogitsの最大絶対誤差0.0 |
| R50 backward | 小さい時空間入力（8/2フレーム・32px）で全パラメータの勾配が有限 |
| 通常モードの二値R50 ONNX（依存固定前の記録） | 未学習2クラス・32/8フレーム・224pxで書き出し成功、出力は`[1, 2]`、ONNX Runtimeとeagerの最大絶対誤差9.31e-8 |
| 互換モードの二値R50 ONNX | ランダム初期化の33,649,098パラメータ、入力430×295・内部crop224。任意ラベル`positive=0, negative=1`で出力列`[0, 1]`を維持。単一ファイル135,143,551 bytes、バッチ1・2・4をバイト列からロードして推論成功。PyTorchとの最大絶対誤差1.0804e-7 |
| 上記R50互換モードの検証環境（ORT 1.20.1対応前の記録） | ONNX 1.17.0、ONNX Runtime 1.22.0、ONNX Script 0.7.2、onnx-ir 1.1.0、opset 18、IR 10 |
| 上限opsetとCLI（ORT 1.20.1対応前の記録） | 任意ラベル`class_zero`／`class_one`の小型モデルを、ラベル指定なしのCLIでopset 22に書き出し。出力列`[0, 1]`を維持し、バッチ1・2・4で推論成功。最大絶対誤差7.4506e-9。現在の上限は21 |
| ORT 1.20.1対応 | export関連74テスト成功。小型SlowFastの通常・互換モードをopset 18／21で書き出し、互換モードはバッチ1・2・4でPyTorchとの数値一致を確認。無名Gather群の最適化時の名前衝突を再現し、命名補正後のロード・推論成功、名前以外のprotobuf不変、サブグラフ・ローカル関数内の命名を確認 |
| wheel / sdist | ビルド成功、wheel内の第三者ライセンス同梱を確認 |

CPUテスト中に、利用可能なアクセラレーターによって`pin_memory`が無効になる旨のPyTorch警告が出る場合があります。CUDA／NCCL、CUDA AMP、複数GPU・複数ノード、MPS、およびモバイル端末での実機検証は別途必要です。CPUでのテスト成功を、それらの性能や動作の検証済みという意味では扱いません。

上表の公式R50との数値比較は初回実装時に行った検証の記録です。現在のテストは、このリポジトリのモデルの形状・パラメータ数・初期化・勾配・pooling・エクスポートを検証し、参照元パッケージのインストールを必要としません。

互換モードの実サイズR50検証は、ランダム初期化した重みでの入出力・数値一致の確認です。学習済みモデルの分類精度や実機速度の測定結果ではありません。

## 実装と参照元

| ファイル | 役割 |
|---|---|
| `src/yaslowfast/prepare.py` | tar入力・manifest自動生成、動画分割、JPEG化、tar作成、RGB統計の自動計算、検査 |
| `src/yaslowfast/data.py` | WebDatasetストリーム、augmentation、DataLoader |
| `src/yaslowfast/models.py` | 純PyTorchのSlowFast-R50とpathway生成 |
| `src/yaslowfast/train.py` | DDP、AMP、勾配蓄積、学習・検証・checkpoint |
| `src/yaslowfast/evaluate.py`, `metrics.py` | 分散評価、動画単位集約、指標 |
| `src/yaslowfast/predict.py` | 動画ファイルからの推論 |
| `src/yaslowfast/export.py`, `benchmark.py` | 通常／互換モードの出力、数値比較、速度測定 |
| `src/yaslowfast/onnx_export.py` | ONNXのバージョン・opset・IR制限、単一ファイルの検査 |
| `src/yaslowfast/compat_slowfast.py` | 既存SlowFast入出力への変換、前処理、クラス順、署名の検証 |
| `src/yaslowfast/stats.py` | trainフレームだけのRGB統計 |
| `src/yaslowfast/config.py` | 未知のキーを拒否する設定読み込み・検証 |
| `src/yaslowfast/labels.py` | 二値分類のクラス数・クラスID対応の共通検証 |

- [PyTorchVideo SlowFastモデル](https://github.com/facebookresearch/pytorchvideo/blob/2ac6ed361309cb25144636ad82dd8bd6a4970e0c/pytorchvideo/models/slowfast.py)
- [公式SlowFast-R50学習設定](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/SLOWFAST_8x8_R50.yaml)
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [WebDatasetの形式と実装](https://github.com/webdataset/webdataset)
- [FFmpeg fps filter](https://ffmpeg.org/ffmpeg-filters.html#fps)

PyTorchVideo由来部分の著作権表示と変更内容は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、Apache License 2.0本文は[LICENSE](LICENSE)に保存し、wheelにも同梱しています。
