# toy problem: Satellite-USV 環境低減制御

これはROSノードではなく、スライド方針を小さく試すための独立した簡易プロトタイプです。主に見るのは `run_control_demo.py` です。

## 目的

簡単化・改変したモデルで実装を試行します。詳細分布 `x_k` の平均化で得られる粗マップ `y_k = A x_k` を、固定の目標粗マップ `\bar{y}` に近づけます。

ただし、「初期分布をただ下げればよい」という意味ではありません。デフォルトでは、初期場は右半分が広く高い分布、上位targetは右上だけを相対的に高く残す分布にしています。スライド方針で見たいのは、上層が作る粗い目標出力 `\bar{y}` に対して、局所的にしか作用できないUSVが `y_k = A x_k` をどれだけ近づけられるかです。

上位plannerは人工的な粗い目標 `\bar{y}_{k+1}` を作り、下位plannerは局所移動・局所制御候補を選びます。

## モデル

```tex
y_k = A x_k
```

```tex
x_{k+1}
= x_k + \Delta t\{D\nabla^2 x_k - \lambda x_k + q_k - \beta u_k\}
```

```tex
\|A x_{k+1} - \bar{y}_{k+1}\|
```

デフォルトのtargetは、現在値を少し下げるだけではなく、固定の空間目標です。

```tex
\bar{y}_{j}
= \alpha \max_i y_{0,i}\, r_j,
\qquad
\max_j r_j = 1
```

ここで `r_j` は右上の粗セル付近が高くなるパターンです。

制御則はCLF/CBFではなく、MPCでもありません。追従誤差、入力コスト、移動コスト、局所的に下げられる量の報酬を使った一歩先評価です。すでに削り切ったセルにUSVが停留し続けないように、局所減少量の報酬を入れています。

## USV運動

USVの運動は連続ダイナミクスではなく、fine grid上の離散移動です。

```tex
p_{r,k+1}\in\mathcal{N}(p_{r,k})
```

```tex
\mathcal{N}(p)
=
\{\mathrm{stay},\mathrm{up},\mathrm{down},\mathrm{left},\mathrm{right}\}
```

デフォルトではstayに小さいペナルティを入れているため、同程度の候補なら隣接セルへ動きます。また複数USVが同じfine cellに重ならないよう、制御候補セルを予約しながら1台ずつ決めます。

## デフォルト設定

- USV数: `3`
- horizon: `400` steps
- movie stride: `4`
- 粗グリッド: `4x4`
- fine grid: `16x16`
- 状態: 1種類のスカラー環境変数 `x`
- 初期分布: 右半分が広く高い滑らかな場
- 上位target: 右上を高めに残し、それ以外を低くする固定粗マップ
- 下位planner: 局所USV制御の一歩先候補評価
- USV運動: fine grid上の `stay/up/down/left/right`

## スライド方針との比較

| 観点・要素 | メイン方針 | デモ |
| --- | --- | --- |
| 上位目標 | MPC等でplannerから出力 | 固定の粗マップ |
| 環境モデル | NPZ、移流、etc. | 1変数、拡散 + 制御減少 |
| 制御入力 | ガウス関数などで低減 | USVが選んだ格子点で下げる |
| USV運動 | 連続制御 | 上下左右のみ |
| 最適化 | MPC、CLF/CBF | N/A、ヒューリスティック |
| 複数USV | 協調最適化 | 3台を貪欲に順番決定 |
| 評価指標 | `\|A x-\bar{y}\|`, `\sum_i x_i`, `\max_i x_i` | `control_metrics.png` に保存 |

## v1とv2の整理

現在の実装は **v1** です。固定targetを使って、構造が見えるかを確認するtoy problemとして位置づけます。次の **v2** では、上位plannerらしさと局所制御らしさを少し増やします。

| 観点・要素 | v1: 現在 | v2: 次に試す案 |
| --- | --- | --- |
| 上位目標 | 固定の粗マップ `\bar{y}` | desired mapから毎stepの実現可能target `\bar{y}_{k+1}` を生成 |
| desired map | targetそのものとして固定 | `y^{\mathrm{des}}` として別に持つ |
| target生成 | `\bar{y}_j=\alpha\max_i y_{0,i}r_j` | `y_k` から `y^{\mathrm{des}}` へ少し近づくfeasible target |
| 制御入力 | 選択した1格子点だけ下げる | 近傍/ガウスカーネルで局所的に下げる |
| USV運動 | 離散移動 | まず離散のまま。速度制約のtoy表現として扱う |
| planner | 一歩先ヒューリスティック | 一歩先ヒューリスティックを維持しつつtarget追従を明確化 |
| 可視化 | `x`, `A x`, `\bar{y}`, `x_0-x_k` | `y^{\mathrm{des}}`, `\bar{y}_{k+1}`, `A x`, residualも見せる |

v2の上位plannerは、例えば以下のようにします。

```tex
y^{\mathrm{raw}}_{k+1}
=
y_k + \eta\left(y^{\mathrm{des}}-y_k\right)
```

ただしUSVは環境値を増やせないので、増加方向の要求は切ります。

```tex
\bar{y}_{k+1}
=
\min\left(y_k,\ y^{\mathrm{raw}}_{k+1}\right)
```

これにより、固定targetを直接追うのではなく、現在の粗出力 `y_k` から見て次に実現可能な粗目標 `\bar{y}_{k+1}` を上層plannerが出す、という形に近づけます。

制御入力も点ではなく局所カーネルにします。

```tex
x_{i,k+1}
=
x^{\mathrm{natural}}_{i,k+1}
-
\beta a_{r,k}K(i,p_{r,k})
```

ここで `K(i,p)` は現在セルと近傍に効くカーネルです。最初は3x3近傍、次にガウスカーネルを試すのがよいです。

## v2の実装タスク

1. `desired coarse map y_des` を追加する
2. `build_upper_target()` を固定targetから毎stepのfeasible target生成へ変更する
3. 点制御を近傍/ガウスカーネルへ変更する
4. 図を `desired map`, `current target`, `coarse output`, `residual` が分かる構成へ変える
5. metricsに `\|A x_k-\bar{y}_k\|` と `\|A x_k-y^{\mathrm{des}}\|` を分けて出す

## 実行例

```sh
python3 examples/satellite_usv/run_control_demo.py --steps 400 --usv-count 3 --movie-stride 4 --comparison --output-dir satellite_usv_control_demo
```

`--movie-stride` はGIF/MP4に保存するフレームの間引きです。例えば `--steps 400 --movie-stride 4` なら、計算は400 steps行い、動画は約100フレームだけ保存します。より長く試すなら:

```sh
python3 examples/satellite_usv/run_control_demo.py --steps 800 --usv-count 3 --movie-stride 8
```

## 出力

- `control_evolution.gif`
- `control_fields.png`
- `control_metrics.png`
- `control_comparison.png`
