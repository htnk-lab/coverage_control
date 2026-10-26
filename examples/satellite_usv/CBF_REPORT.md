# 一次CBF toy problem レポート

## 目的

このデモは、上位plannerが与える粗い参照出力と、細かい環境場に作用するUSV制御の関係を、最小構成で確認するためのものです。

今回は観測問題ではなく、環境低減制御だけを扱います。粗出力

```tex
y=Ax
```

を上位参照

```tex
\bar{y}
```

の近くに保つため、一次Control Barrier Functionでロボットの介入量ベクトル `a` を決めます。

## 格子とロボット

| 項目 | 設定 |
| --- | --- |
| 粗セル | `2x2 = 4` |
| 細セル | 各粗セルあたり `4x4 = 16` |
| 細セル全体 | `8x8 = 64` |
| ロボット | 4台 |
| 割当 | 各粗セルに1台ずつ固定 |

ロボットは割り当てられた粗セル内の細セルだけを移動します。これはCBF制約ではありません。今回のtoy problemでは、粗セル外へ出る運動を制御対象のシステムに含めない、という扱いです。

## 環境モデル

細かい環境状態を

```tex
x\in\mathbb{R}^{64}
```

とします。連続時間モデルは

```tex
\dot{x}
=
f(x)
-
\beta\operatorname{diag}(x)K(p)a
```

です。ここで

```tex
f(x)=D\nabla^2x-\lambda x
```

です。

| 記号 | 意味 |
| --- | --- |
| `x` | 細セル上の環境状態 |
| `A` | 細セルから粗セルへの平均化行列 |
| `y=Ax` | 粗セル平均出力 |
| `p_r` | ロボット `r` の位置 |
| `a_r` | ロボット `r` の介入量 |
| `K(p)` | 各ロボット位置から作る局所カーネル行列 |
| `\beta` | 介入ゲイン |

`K(p)` の `r` 列が、ロボット `r` のガウス型カーネルです。ロボット位置は担当粗セル内に限定しますが、カーネル自体は粗セル境界で打ち切っていません。そのため、隣の粗セル平均にも小さく影響し、入力は結合します。

離散時間シミュレーションではEuler近似で

```tex
x_{k+1}
=
x_k
+
\Delta t
\left[
D\nabla^2x_k-\lambda x_k-\beta u_k
\right]
```

を計算します。ここで

```tex
u_k=\operatorname{diag}(x_k)K(p_k)a_k
```

です。濃度 `x_j` が0に近い細セルでは除去量も0に近くなるため、一定量を引き続けるモデルより非負性と整合しやすい形です。

## 上位参照

粗出力は、各粗セル内の16細セルの平均です。

```tex
y_i=(Ax)_i
```

上位plannerは、今回は滑らかな参照軌道を直接出すものとして簡単化しています。

```tex
\bar{y}_i(t)
=
y_i^{\mathrm{des}}
+
\left(
y_i(0)-y_i^{\mathrm{des}}
\right)
\exp(-\rho t)
```

したがって

```tex
\bar{y}_i(0)=y_i(0)
```

となり、初期時刻ではCBF集合内から始まります。

## 一次CBF

粗セルごとの誤差を

```tex
e_i=y_i-\bar{y}_i
```

とし、許容誤差幅を `\varepsilon` として

```tex
b_i=\varepsilon^2-e_i^2
```

を定義します。

`b_i\ge0` は

```tex
|y_i-\bar{y}_i|\le\varepsilon
```

を意味します。この集合を保つため、一次CBF条件

```tex
\dot{b}_i+\gamma b_i\ge0
```

を課します。

粗出力の微分は

```tex
\dot{y}
=
Af(x)
-
\beta S(x,p)a
```

```tex
S(x,p)
=
A\operatorname{diag}(x)K(p)
```

です。したがって、セル `i` のCBF条件は

```tex
-2e_i
\left[
(Af(x))_i
-
\dot{\bar{y}}_i
-
\beta S_{i:}(x,p)a
\right]
+
\gamma
\left(
\varepsilon^2-e_i^2
\right)
\ge0
```

となります。これは入力ベクトル `a` に関する線形不等式です。

## QP

実装では、4台分の入力をまとめた結合QPとして解きます。名目入力は0です。

```tex
\min_{a,\delta}
\frac12 a^{\mathsf T}a
+
\frac{M}{2}\delta^{\mathsf T}\delta
```

```tex
\mathrm{s.t.}
\quad
0\le a_r\le a_{\max}
```

```tex
\dot{b}_i+\gamma b_i\ge-\delta_i
```

```tex
\delta_i\ge0
```

これにより、「CBF条件を満たすために必要な最小介入量」を求めます。スラック `\delta_i` は、入力上限、一方向入力、ロボット位置、離散化などにより条件が厳密には満たせない場合の診断量です。

## 実装上の単純化

| 要素 | 今回の扱い |
| --- | --- |
| 環境モデル | 1変数の拡散 + 乗算型局所低減 |
| NPZモデル | 未実装 |
| 移流 | 未実装 |
| 上位planner | 滑らかな粗参照を直接生成 |
| ロボット運動 | 粗セル内に限定した離散移動 |
| CBF対象 | 粗出力誤差 `y_i-\bar{y}_i` |
| CBF入力 | 全ロボットの介入量ベクトル `a` |
| ロボット速度 | CBFでは決めない |

今回の実装は、速度までCBFで決めるHOCBFではありません。ロボット位置 `p_r` は既知の時変パラメータのように扱い、その時点の位置で有効な介入カーネルを使って `a` だけを決めています。

## 可視化

実行例:

```sh
python3 examples/satellite_usv/run_cbf_demo.py --steps 80 --output-dir satellite_usv_cbf_demo
```

出力:

| ファイル | 内容 |
| --- | --- |
| `cbf_evolution.gif` | 時間変化 |
| `cbf_fields.png` | 最終時刻の状態 |
| `cbf_metrics.png` | CBF診断指標と入力 |

`cbf_fields.png` では以下を確認します。

| パネル | 見るもの |
| --- | --- |
| `fine state x_k` | 64細セル上の環境状態 |
| `coarse output y_k=A x_k` | 粗セル平均 |
| `upper reference \bar{y}_k` | 上位planner由来の参照 |
| `desired output y^{des}` | 最終的に近づきたい粗出力 |
| `local control field u_k` | ロボット位置に応じた局所介入 |
| `CBF barrier b_i` | CBF条件で保ちたい余裕 |

`cbf_metrics.png` では、CBF成立性に近い量を優先して表示します。

| 指標 | 意味 |
| --- | --- |
| `\max_i |e_i|` | 成分ごとの最大追従誤差 |
| `\varepsilon` | 許容誤差の水平線 |
| `\min_i b_i` | barrierの最小値 |
| `\min_i(\dot b_i+\gamma b_i)` | 連続時間CBF残差 |
| `\max_i\delta_i` | CBFスラック |
| `\min_i \Delta b_i^{disc}` | Euler更新後の離散CBF残差 |
| `a_r` | 各ロボットの介入量 |

ここで離散CBF残差は

```tex
\Delta b_i^{disc}
=
b_{i,k+1}
-
(1-\gamma\Delta t)b_{i,k}
```

として表示しています。

## 結果の読み方

現在のデフォルト出力例では、主な値は以下です。

| 指標 | 値 |
| --- | --- |
| final tracking error | `0.1390` |
| final desired error | `0.1399` |
| maximum `\max_i |e_i|` | `0.0800` |
| `\varepsilon` | `0.0800` |
| minimum final barrier | `2.1758e-07` |
| maximum pre-slack continuous CBF violation | `9.0494e-06` |
| maximum slack | `9.0494e-06` |
| minimum relaxed continuous CBF residual | `-2.5892e-11` |
| minimum Euler discrete CBF residual | `1.8242e-07` |
| maximum input | `0.4000` |

`tracking error` は2ノルムなので、成分ごとのCBF条件を見るには `\max_i |e_i|` を見る方が直接的です。今回のデフォルトでは、`max_i |e_i|` がほぼ `\varepsilon` に張り付きます。これは名目入力を0にした最小介入QPなので、必要になるまで介入を増やさないためです。

`minimum final barrier` が正で、緩和後の連続時間CBF残差がほぼ非負であれば、

```tex
|y_i-\bar{y}_i|\le\varepsilon
```

を保つ方向に入力が選ばれている、と解釈できます。さらに今回はEuler離散残差も正なので、この設定では離散シミュレーション上でも大きな破綻は見えていません。

一方で、pre-slackの連続CBF違反とスラックが完全な0ではない点は重要です。これは、一方向の低減入力、入力上限、ロボット位置、離散化の影響を受けるためです。したがって、この結果は「CBF条件をスラック付きQPとして満たし、必要な緩和量が十分小さい」と読むのが正確です。

## 今後の拡張

次に進めるなら、以下が自然です。

1. ロボット位置を連続化し、速度入力 `v_r` を持たせる
2. 介入量 `a_r` の変化率 `w_r=\dot a_r` を入力に持ち上げる
3. 速度と介入量を同時に扱うHOCBFへ拡張する
4. 環境モデルをNPZまたはNPZDに近づける
5. 上位plannerを固定の指数参照ではなくMPCで生成する
