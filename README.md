# coverage_control

## 動作環境

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12

## インストール

```sh
cd ~/ros2_ws/src/coverage_control
python3 -m pip install -r requirements/lib.txt
rosdep install -i -y --from-paths .
```

```sh
sudo python -m pip install -r requirements/tools.txt
```

Python 3.12で `pip install` が失敗する場合は、必要に応じて `--break-system-packages` を付けてください。

## 使い方

### simple coverage control

launch引数で以下を選べます。

- 場の次元: `{1,2,3}`
- エージェント数: `{1,2,3,4,5}`
- 初期密度関数 `phi` の種類: `{1: Uniform, 2: Gaussian, 3: Disk}`

```sh
ros2 launch coverage_control scc.launch dim:={1,2,3} num:={1,2,3,4,5} phi:={1,2,3}
```

#### rviz

launch fileとrviz panelのチェックボックスを調整すると、密度マップ `phi` と各エージェントのセンシング領域 `sr` を、それぞれ `Marker` と `Pointcloud2` として表示できます。

<img src=assets/scc_1d_3.png width=50%><img src=assets/scc_1d_5_phi.png width=50%>
<img src=assets/scc_2d_5.png width=50%><img src=assets/scc_2d_5_phi.png width=50%>
<img src=assets/scc_3d_5.png width=50%><img src=assets/scc_3d_4_phi.png width=50%>

#### rqt_graph

<img src=assets/scc_2d_rosgraph.png width=100%>

### persistent coverage control

`pcc-devel` branch向けの起動例です。

```sh
ros2 launch coverage_control pcc.launch dim:={1,2,3} agent_num:={1,2,3,4,5}
```

#### rviz

<img src=assets/pcc_2d_5.gif width=70%>
<img src=assets/pcc_3d_5.gif width=70%>

## 研究用プロトタイプ

### toy problem: Satellite-USV 環境低減制御

#### 一次CBFデモ

粗セル4個、各粗セルあたり細セル16個、合計64細セルを使う最小版です。CBFで決めるのは全ロボットの介入強度ベクトル `u` だけです。ロボットは各粗セルに1台ずつ固定割当し、割り当て粗セル内の細セルだけを動きます。これはCBF制約ではなく、粗セル外へ出る運動を制御対象のシステムに含めない、という扱いです。

モデルと結果の短いレポートは [examples/satellite_usv/CBF_REPORT.md](examples/satellite_usv/CBF_REPORT.md) にあります。

```tex
y_k=(Ax)_k
```

```tex
\dot{x}
=
f(x)-\beta\operatorname{diag}(x)\kappa(p)u
```

```tex
b_k
=
\varepsilon^2-\left(y_k-\bar{y}_k\right)^2
```

```tex
\dot{b}_k+\gamma b_k\ge-\delta_k
```

上の条件は `u` に関する線形不等式になるため、実装では全ロボット入力をまとめたスラック付きQPを解いています。ノミナル入力は0で、CBF条件を満たすために必要な最小介入強度を見る設定です。

```sh
cd ~/ros2_ws/src/coverage_control
python3 examples/satellite_usv/run_cbf_demo.py --steps 80 --output-dir satellite_usv_cbf_demo
```

主な出力:

- `cbf_evolution.gif`: `x`、粗出力 `y_k=A x`、上位参照 `\bar{y}_k`、desired `y^{\mathrm{des}}`、局所除去場、barrier `b_k`
- `cbf_fields.png`: 最終時刻の場
- `cbf_metrics.png`: `max_i |e_i|`、CBF barrier、連続CBF残差、スラック、Euler離散残差、各入力

#### HOCBFデモ

一次CBFの次の最小版です。ロボット位置 `p_i` と介入強度 `u_i` を状態にし、速度 `v_i` と介入変化率 `\nu_i` をHOCBF-QPで決めます。環境モデルはまだ1変数ですが、局所的な供給がある状況を見るため、拡散・減衰に小さい固定発生項を加えています。移動発生項は `--source-mode moving` のstress testとして分けています。

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

上側barrier

```tex
b_k=\bar y_k+\varepsilon_k-y_k
```

に対してHOCBF条件を課します。位置制約と介入強度上下限制約も同じQPに一次CBFとして入れています。位置制約は各辺ごとのbox制約ではなく、各ロボット1本の6次ノルム型バリアです。名目速度は担当粗セル内の介入効率が高い位置へ向け、介入強度を上げっぱなしにしないため `\nu_i^{\mathrm{nom}}=-k_u u_i` を入れています。標準設定の発生項 `q_j^k` は固定された局所sourceで、移動sourceは `--source-mode moving` のstress testとして分けています。

```tex
h_{p,i}
=
1
-
\left|
\frac{p_{i,x}-c_{i,x}}{d_{i,x}}
\right|^6
-
\left|
\frac{p_{i,y}-c_{i,y}}{d_{i,y}}
\right|^6,
\qquad
\nabla h_{p,i}^{\mathsf T}v_i+k_ph_{p,i}\ge0
```

```sh
cd ~/ros2_ws/src/coverage_control
python3 examples/satellite_usv/run_hocbf_demo.py --steps 240 --output-dir satellite_usv_hocbf_demo
```

モデルと結果の短いレポートは [examples/satellite_usv/HOCBF_REPORT.md](examples/satellite_usv/HOCBF_REPORT.md) にあります。

#### ヒューリスティック制御デモ

簡単化・改変したモデルで実装を試行します。詳細分布 `x_k` の平均化で得られる粗マップ `y_k = A x_k` を、固定の目標粗マップ `\bar{y}` に近づけます。

ただし、「初期分布をただ下げればよい」という意味ではありません。デフォルトでは、初期場は右半分が広く高い分布、上位targetは右上だけを相対的に高く残す分布にしています。見たいのは、上層が作る粗い目標出力 `\bar{y}` に対して、局所的にしか作用できないUSVが `y_k = A x_k` をどれだけ近づけられるかです。

```tex
y_k = A x_k
```

```tex
x_{k+1}
= x_k + \Delta t\{D\nabla^2 x_k - \lambda x_k + q_k - \beta u_k\}
```

```tex
\bar{y}_{j}
= \alpha \max_i y_{0,i}\, r_j,
\qquad
\max_j r_j = 1
```

ここで `r_j` は右上の粗セル付近が高くなる固定の空間パターンです。

下位plannerは、一歩先の移動候補と局所制御候補を評価します。

```tex
J^{(m)}
= \|A x^{(m)}_{k+1} - \bar{y}_{k+1}\|^2
+ c_u\|a^{(m)}_k\|^2
+ c_p d(p^{(m)}_{k+1},p_k)
- c_\ell \Delta x^{(m)}_{\mathrm{local}}.
```

これはMPCでもCLF/CBF制御でもありません。すでに削り切ったセルにUSVが停留し続けないように、局所減少量の報酬を入れています。

USV運動はfine grid上の離散移動です。

```tex
p_{r,k+1}\in\mathcal{N}(p_{r,k})
```

```tex
\mathcal{N}(p)
=
\{\mathrm{stay},\mathrm{up},\mathrm{down},\mathrm{left},\mathrm{right}\}
```

デフォルトではstayに小さいペナルティを入れているため、同程度の候補なら隣接セルへ動きます。また複数USVが同じfine cellに重ならないよう、制御候補セルを予約しながら1台ずつ決めます。

実行例:

```sh
cd ~/ros2_ws/src/coverage_control
python3 examples/satellite_usv/run_control_demo.py --steps 400 --usv-count 3 --movie-stride 4 --comparison --output-dir satellite_usv_control_demo
```

デフォルト設定:

- USV数: `3`。`--usv-count` で変更可能
- horizon: `400` steps。広い場の変化が見えるように長め
- movie stride: `4`。400 stepsを計算しても、GIFには約100フレームだけ保存する
- 粗グリッド: `4x4`
- fine grid: `16x16`
- 状態: 1種類のスカラー環境変数 `x`
- 初期分布: 右半分が広く高い滑らかな場
- 上位目標: 右上を高めに残し、それ以外を低くする固定粗マップ
- 下位planner: 局所USV制御の一歩先候補評価
- USV運動: fine grid上の `stay/up/down/left/right`

スライド方針との比較:

| 観点・要素 | メイン方針 | 一次CBFデモ | HOCBFデモ | ヒューリスティックデモ |
| --- | --- | --- | --- | --- |
| 上位目標 | MPC等でplannerから出力 | 滑らかな粗参照 | 滑らかな粗参照 | 固定の粗マップ |
| 環境モデル | NPZ、移流、etc. | 1変数、拡散 + 乗算型低減 | 1変数、拡散 + 乗算型低減 | 1変数、拡散 + 制御減少 |
| 制御入力 | ガウス関数などで低減 | 結合CBF-QPで `u_i` を決定 | HOCBF-QPで `v_i,\nu_i` を決定 | USVが選んだ格子点で下げる |
| USV運動 | 連続制御 | 粗セル内に限定した外部更新 | 連続位置 `\dot p=v` | 上下左右のみ |
| 最適化 | MPC、CLF/CBF | 一次CBF-QP | HOCBF-QP | N/A、ヒューリスティック |
| 複数USV | 協調最適化 | 4台、各粗セルに1台固定 | 4台、各粗セルに1台固定 | 3台を貪欲に順番決定 |
| 評価指標 | `A x-\bar{y}`, CBF条件, 入力 | `cbf_metrics.png` | `hocbf_metrics.png` | `control_metrics.png` |

v2では、固定targetを直接追うのではなく、desired mapから毎stepの実現可能target `\bar{y}_{k+1}` を作る方針です。また、点制御を近傍/ガウスカーネルへ拡張します。

| 観点・要素 | v1: 現在 | v2: 次に試す案 |
| --- | --- | --- |
| 上位目標 | 固定の粗マップ | desired mapから毎step生成 |
| 制御入力 | 選択格子点で下げる | 近傍/ガウスカーネルで下げる |
| 可視化 | `x`, `A x`, `\bar{y}`, `x_0-x_k` | desired map, current target, residualも表示 |

詳細は [examples/satellite_usv/README.md](examples/satellite_usv/README.md) を参照してください。

比較モード:

- `no_control`: 自然ダイナミクスのみ
- `random_control`: ランダムなUSV移動と局所減少
- `greedy_fine`: fine-cell値が大きい方向へ移動
- `hierarchical`: `\|A x_{k+1} - \bar{y}_{k+1}\|`、入力コスト、移動コストを使って候補評価

主な出力:

- `control_evolution.gif`: `x_k`、粗出力 `y_k = A x_k`、目標 `\bar{y}_{k+1}`、減少量 `x_0-x_k` の時間変化
- `control_fields.png`: 最終時刻の場
- `control_metrics.png`: 追従誤差、総量、最大値、制御コスト
- `control_comparison.png`: baseline比較。`--comparison` を付けた場合のみ出力

長く計算しつつ動画生成を軽くしたい場合:

```sh
python3 examples/satellite_usv/run_control_demo.py --steps 800 --usv-count 3 --movie-stride 8
```

## 開発用コマンド

### フォーマット

- isort
- black

```sh
task fmt
```

### lint

- black
- ruff

```sh
task lint
```

### mypy

- mypy

```sh
task mypy
```
