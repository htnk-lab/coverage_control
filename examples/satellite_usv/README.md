# toy problem: Satellite-USV 環境低減制御

これはROSノードではなく、スライド方針を小さく試すための独立した簡易プロトタイプです。

現在は2種類あります。

- `run_cbf_demo.py`: 粗4セル・各粗セルあたり細16セル、合計64細セルの一次CBF版。まず見るならこちら
- `run_hocbf_demo.py`: 連続位置 `p_i` と介入強度 `u_i` を持つHOCBF版
- `run_control_demo.py`: 4x4粗マップ・16x16細マップのヒューリスティック版

## 目的

簡単化・改変したモデルで実装を試行します。詳細分布 `x_k` の平均化で得られる粗マップ `y_k = A x_k` を、固定の目標粗マップ `\bar{y}` に近づけます。

ただし、「初期分布をただ下げればよい」という意味ではありません。デフォルトでは、初期場は右半分が広く高い分布、上位targetは右上だけを相対的に高く残す分布にしています。スライド方針で見たいのは、上層が作る粗い目標出力 `\bar{y}` に対して、局所的にしか作用できないUSVが `y_k = A x_k` をどれだけ近づけられるかです。

上位plannerは人工的な粗い目標 `\bar{y}_{k+1}` を作り、下位plannerは局所移動・局所制御候補を選びます。

## 一次CBFデモ

`run_cbf_demo.py` は、CBFの構造だけを見やすくするためにさらに小さくした版です。

モデルと結果の簡単なレポートは [CBF_REPORT.md](CBF_REPORT.md) にまとめています。

- 粗セル: `2x2`、合計4セル
- 細セル: 各粗セルあたり `4x4`、合計 `8x8 = 64` セル
- ロボット: 4台。各粗セルに1台ずつ固定割当
- ロボット運動: 割り当て粗セル内の細セルだけを移動
- CBFで決めるもの: 全ロボットの介入強度ベクトル `u`
- CBFで決めないもの: ロボット速度、ロボットが粗セル外へ出ない制約

ロボットが粗セルから出ないことはCBF制約ではありません。このtoy problemでは、そもそも各ロボットの候補位置集合を割り当て粗セル内に限定し、制御対象のシステムから「粗セル外への移動」を除いています。

```tex
y_k
=
(A x)_k
```

```tex
\dot{x}
=
f(x)-\beta\operatorname{diag}(x)\kappa(p)u
```

```tex
\dot{y}
=
Af(x)
-
\beta S(x,p)u
```

```tex
S(x,p)=A\operatorname{diag}(x)\kappa(p)
```

ここで `\kappa(p)` はロボット位置まわりの介入カーネルです。カーネルは粗セル境界で打ち切らないため、入力は粗セル間で小さく結合します。

上位plannerは滑らかな粗参照を出すものとして置いています。

```tex
\bar{y}_k(t)
=
y^{\mathrm{des}}_k
+
\left(y_k(0)-y^{\mathrm{des}}_k\right)
\exp(-\rho t)
```

一次CBFは各粗セルごとに次で定義します。

```tex
e_k
=
y_k-\bar{y}_k
```

```tex
b_k
=
\varepsilon^2-e_k^2
```

```tex
\dot{b}_k+\gamma b_k\ge 0
```

この条件を入力ベクトル `u` について書くと線形不等式になります。

```tex
-2e_k
\left(
(A f(x))_k-\dot{\bar{y}}_k
-
\beta S_{k:}(x,p)u
\right)
+
\gamma b_k
\ge 0
```

実装では、全ロボット入力をまとめたスラック付きQPを解きます。ノミナル入力は0なので、CBF条件を満たすために必要な最小介入強度を見る設定です。

```tex
\min_{u,\delta}
\frac12u^{\mathsf T}u+\frac{M}{2}\delta^{\mathsf T}\delta
```

```tex
\mathrm{s.t.}\quad
0\le u_i\le u_{\max},
\qquad
\dot b_k+\gamma b_k\ge-\delta_k,
\qquad
\delta_k\ge0
```

実行例:

```sh
python3 examples/satellite_usv/run_cbf_demo.py --steps 80 --output-dir satellite_usv_cbf_demo
```

出力:

- `cbf_evolution.gif`: `x`, `y_k=A x`, `\bar{y}_k`, `y^{\mathrm{des}}`, 局所除去場, `b_k` の時間変化
- `cbf_fields.png`: 最終時刻の場
- `cbf_metrics.png`: `max_i |e_i|`、CBF barrier、連続CBF残差、スラック、Euler離散残差、各入力

## HOCBFデモ

`run_hocbf_demo.py` は、一次CBFの次の最小版です。モデルと結果の簡単なレポートは [HOCBF_REPORT.md](HOCBF_REPORT.md) にまとめています。

```tex
\dot x
=
D\nabla^2x-\lambda x
+
q(t)
-
\beta\operatorname{diag}(x)\kappa(p)u
```

```tex
\dot p_i=v_i,
\qquad
\dot u_i=\nu_i
```

上側barrierだけを使います。

```tex
b_k
=
\bar y_k+\varepsilon_k-y_k
```

HOCBF条件は

```tex
\psi_{k,1}
=
\dot b_k+c_1b_k
```

```tex
\psi_{k,2}
=
\dot\psi_{k,1}+c_2\psi_{k,1}
\ge-\delta_k
```

です。QPで決める入力は `v_i,\nu_i` です。位置制約と介入強度上下限制約も同じQPに一次CBFとして入れています。位置制約は各辺ごとのbox制約ではなく、各ロボット1本の6次ノルム型バリアです。

```tex
h_{p,i}(p_i)
=
1
-
\left|
\frac{p_{i,x}-c_{i,x}}{d_{i,x}}
\right|^6
-
\left|
\frac{p_{i,y}-c_{i,y}}{d_{i,y}}
\right|^6
```

```tex
\nabla h_{p,i}(p_i)^{\mathsf T}v_i
+
k_p h_{p,i}(p_i)
\ge0
```

したがって4台なら位置CBFは4本です。この制約集合は担当粗セルboxに内接するため、`h_{p,i}\ge0` なら粗セル外には出ません。

HOCBFだけでは移動目的を作らないため、ノミナル速度は担当粗セル内の介入効率が高い位置へ向けています。また、介入強度を上げっぱなしにしないため、

```tex
\nu_i^{\mathrm{nom}}=-k_u u_i
```

を入れています。動作確認用に、初期場は粗セル内の端側にhotspotを置き、標準設定では同じ付近に固定された小さい発生項 `q_j^k` を入れています。ガウスカーネル幅は狭めです。移動する発生項は `--source-mode moving` を指定したときだけ使うstress testです。

実行例:

```sh
python3 examples/satellite_usv/run_hocbf_demo.py --steps 240 --output-dir satellite_usv_hocbf_demo
```

出力:

- `hocbf_evolution.gif`: 軌跡つきで `x`, `q`, `y_k=A x`, `\bar{y}_k`, removal rate, `b_k`, `||v_i||_\infty` を表示
- `hocbf_fields.png`: 最終時刻の場
- `hocbf_metrics.png`: `b_k`, `psi_1`, `psi_2`, スラック、位置barrier、`u_i`, `||v_i||_\infty`, 累積移動距離, `J_u,J_v,L`

## ヒューリスティック制御デモ

`run_control_demo.py` は、粗目標を追う構図をもう少し大きい格子で見せる版です。こちらはCBF/CLF/MPCではなく、一歩先候補評価です。

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

| 観点・要素 | メイン方針 | 一次CBFデモ | HOCBFデモ | ヒューリスティックデモ |
| --- | --- | --- | --- | --- |
| 上位目標 | MPC等でplannerから出力 | 滑らかな粗参照 | 滑らかな粗参照 | 固定の粗マップ |
| 環境モデル | NPZ、移流、etc. | 1変数、拡散 + 乗算型低減 | 1変数、拡散 + 乗算型低減 | 1変数、拡散 + 制御減少 |
| 制御入力 | ガウス関数などで低減 | 結合CBF-QPで `u_i` を決定 | HOCBF-QPで `v_i,\nu_i` を決定 | USVが選んだ格子点で下げる |
| USV運動 | 連続制御 | 粗セル内に限定した外部更新 | 連続位置 `\dot p=v` | 上下左右のみ |
| 最適化 | MPC、CLF/CBF | 一次CBF-QP | HOCBF-QP | N/A、ヒューリスティック |
| 複数USV | 協調最適化 | 4台、各粗セルに1台固定 | 4台、各粗セルに1台固定 | 3台を貪欲に順番決定 |
| 評価指標 | `A x-\bar{y}`, CBF条件, 入力 | `cbf_metrics.png` | `hocbf_metrics.png` | `control_metrics.png` |

## ヒューリスティック版のv1とv2の整理

`run_control_demo.py` は **v1** です。固定targetを使って、構造が見えるかを確認するtoy problemとして位置づけます。次の **v2** では、上位plannerらしさと局所制御らしさを少し増やします。

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
