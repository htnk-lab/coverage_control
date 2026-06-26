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

| 観点・要素 | メイン方針 | デモ |
| --- | --- | --- |
| 上位目標 | MPC等でplannerから出力 | 固定の粗マップ |
| 環境モデル | NPZ、移流、etc. | 1変数、拡散 + 制御減少 |
| 制御入力 | ガウス関数などで低減 | USVが選んだ格子点で下げる |
| USV運動 | 連続制御 | 上下左右のみ |
| 最適化 | MPC、CLF/CBF | N/A、ヒューリスティック |
| 複数USV | 協調最適化 | 3台を貪欲に順番決定 |
| 評価指標 | `\|A x-\bar{y}\|`, `\sum_i x_i`, `\max_i x_i` | `control_metrics.png` に保存 |

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
