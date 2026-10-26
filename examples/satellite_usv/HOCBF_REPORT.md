# HOCBF toy problem レポート

## 目的

一次CBF版では、ロボット位置 `p_i` を外から与え、介入強度だけをQPで決めました。HOCBF版では、ロボット位置と介入強度を状態に持ち上げます。

```tex
\dot p_i=v_i,
\qquad
\dot u_i=\nu_i
```

したがって、QPが直接決める入力はロボット速度 `v_i` と介入変化率 `\nu_i` です。ここで `u_i` はロボット `i` の物理的な介入強度です。

## 添字と集合

| 対象 | 記号 | 現在の設定 |
| --- | --- | --- |
| 粗セル | `k\in\mathcal K` | `N_c=4` |
| 粗セル `k` 内の細セル | `j\in\mathcal J` | 各粗セルあたり `N_f=16` |
| 細セル中心 | `r_j^k\in\mathbb R^2` | fine gridのセル中心 |
| ロボット | `i\in\mathcal I` | `N_R=4` |
| 割当写像 | `\sigma:\mathcal I\to\mathcal K` | 現在は `N_R=N_c`, `\sigma(i)=i` |

実装上は粗セルとロボットが1対1対応なので、かなりの場所で `i=k` のように見えます。ただし、数式では粗セル添字 `k` とロボット添字 `i` を分けます。

## 細セルモデル

粗セル `k` 内の細セル `j` の状態を

```tex
x_j^k(t)\ge0
```

とします。toy problemでは1変数の環境状態だけを扱います。将来NPZへ拡張するなら、`x_j^k` を `\boldsymbol x_j^k=[N_j^k,P_j^k,Z_j^k]^{\mathsf T}` に置き換える想定です。

自然ダイナミクスは

```tex
f_j^k(x)
=
D
\sum_{(\ell,m)\in\mathcal N_j^k}
\left(
x_m^\ell-x_j^k
\right)
-
\lambda x_j^k
```

です。ここで `D` は拡散係数、`\lambda` は自然減衰率です。

HOCBFデモでは、制御後に状態が単調に消えるだけにならないよう、小さい局所発生項を入れています。標準設定では発生源中心は固定です。

```tex
q_j^k(\rho_k)
=
s_k
\exp
\left(
-
\frac{\|r_j^k-\rho_k\|^2}{2\sigma_q^2}
\right)
```

`\rho_k` は粗セル `k` 内の固定発生源中心です。これはNPZモデルの代わりではなく、局所的に状態が供給される場合でもHOCBF-QPが上側制約を保てるかを見るためのtoy forcingです。移動発生源は `--source-mode moving` を指定した場合だけ使うstress testであり、本筋の結果としては扱いません。

ロボット `i` の介入カーネルは

```tex
\kappa_{ji}^k(p_i)
=
\exp
\left(
-
\frac{\|r_j^k-p_i\|^2}{2\sigma_u^2}
\right)
```

です。これは観測関数ではなく、介入カーネルです。現在の実装では粗セル境界で打ち切っていないため、担当外の粗セルにも小さく影響します。

以上をまとめると、細セルモデルは

```tex
\dot x_j^k
=
f_j^k(x)
+
q_j^k(\rho_k)
-
\beta x_j^k
\sum_{i\in\mathcal I}
\kappa_{ji}^k(p_i)u_i
```

です。`x_j^k` を掛けているので、`x_j^k=0` なら除去量も0になります。

## 粗セル平均

粗セル出力は、細セル状態の平均です。

```tex
y_k
=
\frac{1}{N_f}
\sum_{j\in\mathcal J}x_j^k
```

ベクトル表記では `y=Ax` です。これは別の粗モデル状態ではなく、細セル状態から計算される出力です。

粗セル平均forcingと介入効果を

```tex
\bar q_k(\rho_k)
=
\frac{1}{N_f}
\sum_{j\in\mathcal J}
q_j^k(\rho_k)
```

```tex
S_{ki}(x,p_i)
=
\frac{1}{N_f}
\sum_{j\in\mathcal J}
x_j^k\kappa_{ji}^k(p_i)
```

と定義します。`S_{ki}` は、ロボット `i` の単位介入強度が粗セル `k` の平均をどれだけ下げられるかを表します。

すると

```tex
\dot y_k
=
f_k^{\mathrm{avg}}(x)
+
\bar q_k(t;\rho_k)
-
\beta
\sum_{i\in\mathcal I}
S_{ki}(x,p_i)u_i
```

です。ここで

```tex
f_k^{\mathrm{avg}}(x)
=
\frac{1}{N_f}
\sum_{j\in\mathcal J}
f_j^k(x)
```

です。

## 上位参照

上位plannerの代わりに、toy problemでは滑らかな粗参照を与えています。

```tex
\bar y_k(t)
=
y_k^{\mathrm{des}}
+
\left(
y_k(0)-y_k^{\mathrm{des}}
\right)
\exp(-\rho t)
```

したがって

```tex
\dot{\bar y}_k(t)
=
-\rho
\left(
y_k(0)-y_k^{\mathrm{des}}
\right)
\exp(-\rho t)
```

```tex
\ddot{\bar y}_k(t)
=
\rho^2
\left(
y_k(0)-y_k^{\mathrm{des}}
\right)
\exp(-\rho t)
```

もQPで使います。

## 出力バリア

現在は上側制約だけを扱います。

```tex
y_k-\bar y_k\le\varepsilon_k
```

したがって、粗セル `k` のバリアは

```tex
b_k
=
\bar y_k+\varepsilon_k-y_k
```

です。安全集合は

```tex
\mathcal C_y
=
\left\{
x:
b_k\ge0,\quad k\in\mathcal K
\right\}
```

です。

## HOCBF条件

1階微分は

```tex
\dot b_k
=
\dot{\bar y}_k
-
f_k^{\mathrm{avg}}(x)
-
\bar q_k(t;\rho_k)
+
\beta
\sum_{i\in\mathcal I}
S_{ki}(x,p_i)u_i
```

です。ここには状態 `p_i,u_i` は現れますが、QP入力 `v_i,\nu_i` はまだ現れません。したがって相対次数2として扱います。

```tex
\psi_{k,0}=b_k
```

```tex
\psi_{k,1}
=
\dot b_k+c_1b_k
```

```tex
\psi_{k,2}
=
\dot\psi_{k,1}
+
c_2\psi_{k,1}
\ge-\delta_k
```

を課します。線形class-`\mathcal K`関数を使っているので、

```tex
\psi_{k,2}
=
\ddot b_k
+
(c_1+c_2)\dot b_k
+
c_1c_2b_k
```

です。

2階微分では

```tex
\frac{d}{dt}
\left[
S_{ki}(x,p_i)u_i
\right]
=
\dot S_{ki}u_i
+
S_{ki}\nu_i
```

かつ

```tex
\dot S_{ki}
=
\frac{\partial S_{ki}}{\partial x}\dot x
+
\nabla_{p_i}S_{ki}^{\mathsf T}v_i
```

なので、HOCBF制約は最終的に

```tex
c_k(z,t)
+
\sum_{i\in\mathcal I}
C_{ki}^{v}(z)v_i
+
\sum_{i\in\mathcal I}
C_{ki}^{\nu}(z)\nu_i
\ge
-\delta_k
```

という、QP入力 `v_i,\nu_i` に関してアフィンな不等式になります。

実装では `source_dot` も同じ関数から返しています。標準の固定sourceでは `\dot q_j^k=0` です。`--source-mode moving` を使うstress testでは `\rho_k(t)` が動くため、`\dot q_j^k(t;\rho_k(t))` を `\ddot b_k` の定数項に `\dot{\bar q}_k` として含めます。つまり、どちらのsource modeでも、シミュレーションとHOCBF-QPは同じ自然ダイナミクスを見ています。

## QP

最適化変数は

```tex
\xi
=
\operatorname{col}
\left(
v_1,\ldots,v_{N_R},
\nu_1,\ldots,\nu_{N_R},
\delta_1,\ldots,\delta_{N_c}
\right)
```

です。QPは

```tex
\begin{aligned}
\min_{\{v_i,\nu_i\},\delta}\quad
&
\frac12
\sum_{i\in\mathcal I}
(v_i-v_i^{\mathrm{nom}})^{\mathsf T}
R_v
(v_i-v_i^{\mathrm{nom}})
\\
&+
\frac12
\sum_{i\in\mathcal I}
r_\nu
(\nu_i-\nu_i^{\mathrm{nom}})^2
+
\frac{M}{2}
\sum_{k\in\mathcal K}
\delta_k^2
\end{aligned}
```

subject to

```tex
\psi_{k,2}\ge-\delta_k
```

位置CBF、介入強度CBF、およびboundsです。

| 制約 | 本数 | スラック | 内容 |
| --- | ---: | --- | --- |
| 出力HOCBF | 4 | あり | `y_k-\bar y_k\le\varepsilon_k` |
| 位置CBF | 4 | なし | 各ロボット1本の6次ノルム型セル内制約 |
| 介入強度CBF | 8 | なし | `0\le u_i\le u_{\max}` |
| 速度bounds | 8 | なし | 4台 x 2成分の `v_i` 上下限 |
| 介入変化率bounds | 4 | なし | 各 `\nu_i` の上下限 |
| slack bounds | 4 | - | `\delta_k\ge0` |

## 位置制約

担当粗セル内にロボットを保持する滑らかなバリアを `h_{p,i}(p_i)` とします。現在は6次ノルム型です。

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
h_{p,i}(p_i)\ge0
```

を満たせば、ロボットは担当粗セルboxに内接する滑らかな領域内にいます。四隅までは使いませんが、粗セル外には出ません。

一次CBF条件は

```tex
\nabla h_{p,i}(p_i)^{\mathsf T}v_i
+
k_p h_{p,i}(p_i)
\ge0
```

です。したがって位置CBFは各ロボット1本、現在は4本です。

## 介入強度制約

介入強度は

```tex
0\le u_i\le u_{\max}
```

です。`u_i` は状態なので、`\nu_i` に対する一次CBFとして扱います。

```tex
h_{u,i}^{\min}=u_i,
\qquad
h_{u,i}^{\max}=u_{\max}-u_i
```

```tex
\nu_i+k_u^{\mathrm{cbf}}u_i\ge0
```

```tex
-\nu_i
+
k_u^{\mathrm{cbf}}
(u_{\max}-u_i)
\ge0
```

です。ノミナル入力の減衰係数 `k_u` と紛らわしいので、記号上は `k_u^{\mathrm{cbf}}` と分けています。

## ノミナル入力

HOCBFそのものは「制約を破らない」ための条件であり、ロボットを積極的に動かす目的関数ではありません。そのため、デモではノミナル入力を別に与え、QPがそれを制約内で補正する構造にしています。

### ノミナル介入変化率

介入強度は、必要がなければ自然に0へ戻します。

```tex
\nu_i^{\mathrm{nom}}
=
-k_u u_i
```

QPはこの `\nu_i^{\mathrm{nom}}` からのずれを罰しつつ、出力HOCBFが必要とする場合には正の `\nu_i` も選びます。

### ノミナル速度

ロボット `i` の担当粗セルを `\sigma(i)` とします。まず、粗セルごとの超過量を

```tex
\eta_k
=
\max(y_k-\bar y_k,0)
```

とします。担当粗セル内の細セル `j` に対して、実装では次のスコアを使っています。

```tex
s_j^{\sigma(i)}
=
\left(
0.25+\eta_{\sigma(i)}
\right)
\left(
x_j^{\sigma(i)}
+
\alpha_q q_j^{\sigma(i)}
\right)
```

ここで `\alpha_q` は `source_target_weight` です。標準設定では固定sourceを含めて介入効率の高い位置を選びます。`--source-mode moving` のstress testでは、この項により移動source側へノミナルターゲットが動きます。

実装では離散セルの最大値が切り替わってノミナル目標が跳ねるのを避けるため、スコアを二乗した重み付き重心を使います。

```tex
\omega_j^{\sigma(i)}
=
\left(
\max(s_j^{\sigma(i)},0)
\right)^2
```

```tex
p_i^{\mathrm{tar}}
=
\frac{
\sum_{j\in\mathcal J}
\omega_j^{\sigma(i)}r_j^{\sigma(i)}
}{
\sum_{j\in\mathcal J}
\omega_j^{\sigma(i)}
}
```

です。以前は最大スコア近傍だけから重心を作っていましたが、4x4の局所格子では最大セルの切り替わりで後半の速度が不自然に振れやすいため、この滑らかな形に変更しました。実装では粗セル境界に近づきすぎないよう、この `p_i^{\mathrm{tar}}` を担当粗セル内の少し内側にclipしています。

ノミナル速度は

```tex
v_i^{\mathrm{nom}}
=
k_v
\left(
p_i^{\mathrm{tar}}-p_i
\right)
```

です。最後に成分ごとに `[-v_{\max},v_{\max}]` にclipします。これはMPCではなく、現在の `x`, `q`, `y-\bar y` から一歩の目標方向を作るヒューリスティックです。したがって、今回のHOCBFデモは

```text
ノミナル移動 + HOCBF safety filter
```

として解釈します。

## 実装対応

最新版のスライド記号と、現在のPython実装名の対応は以下です。内部変数名には旧名が残っていますが、レポート上の意味は `u_i,\nu_i` にそろえています。

| 最新記号 | 実装名 | 意味 |
| --- | --- | --- |
| `k` | `coarse_index`, `coarse_flat` | 粗セル添字 |
| `j` | fine grid index | 細セル添字 |
| `i` | `robot_index` | ロボット添字 |
| `u_i` | `actions` | 介入強度 |
| `\nu_i` | `action_rates` | 介入変化率 |
| `p_i` | `positions` | ロボット位置 |
| `v_i` | `velocities` | ロボット速度 |
| `\kappa_{ji}^k` | `kernel_matrix(...)` | 介入カーネル |
| `S_{ki}` | `_coarse_sensitivity(...)` | 粗セル平均への介入効果 |
| `q_j^k` | `source` | 局所source |
| `\dot q_j^k` | `source_dot` | sourceの時間微分。固定sourceでは0 |
| `\bar y_k` | `target_output` | 上位参照 |
| `b_k` | `barrier` | 出力バリア |
| `\delta_k` | `slack` | 出力HOCBFスラック |

内部変数名には旧名の `actions/action_rates` が残っていますが、これはPython側の配列名です。数式・スライド・レポートでは `u_i/\nu_i` を使います。

## デフォルト設定

| パラメータ | 値 | 意味 |
| --- | ---: | --- |
| `dt` | `0.05` | 時間刻み |
| `steps` | `240` | シミュレーション長 |
| `D` | `0.015` | 拡散係数 |
| `lambda` | `0.0` | 自然減衰 |
| `beta` | `0.85` | 介入ゲイン |
| `sigma_u` | `0.85` | 介入カーネル幅 |
| `u_max` | `1.2` | 介入強度上限 |
| `v_max` | `1.4` | 速度成分上限 |
| `nu_max` | `2.5` | 介入変化率上限 |
| `epsilon` | `0.08` | 上側誤差許容幅 |
| `c_1,c_2` | `5.0, 5.0` | HOCBFゲイン |
| `k_p` | `3.0` | 位置CBFゲイン |
| `k_u^{cbf}` | `5.0` | 介入強度CBFゲイン |
| `source_mode` | `static` | 固定局所source。`moving` はstress test、`none` で無効 |
| `q_0` | `0.025` | source強度 |
| `sigma_q` | `0.65` | source幅 |
| `T_q` | `6.0` | moving stress testの周期 |
| `target_score_power` | `2.0` | ノミナル目標重心の重み指数 |

## 実行と出力

```sh
python3 examples/satellite_usv/run_hocbf_demo.py --steps 240 --output-dir satellite_usv_hocbf_demo
```

| ファイル | 内容 |
| --- | --- |
| `hocbf_evolution.gif` | 時間変化 |
| `hocbf_fields.png` | 最終時刻の状態 |
| `hocbf_metrics.png` | HOCBF診断指標 |

現在のデフォルト例の代表値は以下です。

| 指標 | 値 |
| --- | --- |
| final tracking error | `0.1530` |
| final desired error | `0.1590` |
| minimum barrier | `2.8897e-06` |
| minimum psi1 | `6.7255e-06` |
| maximum pre-slack HOCBF violation | `4.3173e-07` |
| maximum slack | `4.3173e-07` |
| minimum relaxed HOCBF residual | `-2.6648e-12` |
| minimum position barrier | `8.3593e-01` |
| minimum intervention margin | `0.0000e+00` |
| final path length | `7.4468` |
| final intervention energy | `0.9355` |
| final velocity energy | `8.8703` |

`minimum intervention margin` が0なのは、初期状態で `u_i=0` から始めているためです。下限制約自体は `\nu_i+k_u^{\mathrm{cbf}}u_i\ge0` としてQPに入っています。
`minimum psi1` やHOCBF残差はソルバ許容誤差の範囲で0近傍に保たれています。

## 次の課題

1. HOCBFなしのノミナル移動ケースを追加して比較する
2. 一次CBF、HOCBF、ノミナル移動のみの3ケースで総介入強度を比較する
3. 上位参照を指数参照ではなくMPCから生成する
4. 環境モデルをNPZまたはNPZDに拡張する
