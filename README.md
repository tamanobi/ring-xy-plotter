# リング式 XY プロッタ — PC 側ソフト

円環レールの上を走る 2 台の台車 A・B を 1 本の棒でつなぎ、棒に固定したペンで描くプロッタのためのソフトです。
線画の画像から G-code を作る 3 つのコマンドと、機械の較正とずれの補正に使う測定スクリプトが入っています。

プラレールの輪の上を走るペンプロッタのために書いたもので、この機械専用です。機構と経緯は記事にまとめてあります
（記事の URL: 公開後に追記）。同じ形の機械を作る人や、逆運動学、較正、ずれの補正の実例を見たい人の参考に置いています。

- ライセンスは MIT（`LICENSE`）
- 個人の工作の記録なので、プルリクエストは受け付けていません
- 文中の数字は、作者の機械（輪の半径 約 231 mm、A4 の紙）での実測です。ほかの機械では変わります

## 目次

1. [セットアップ](#セットアップ)
2. [絵を描くまで](#絵を描くまで)
3. [送る前の確認](#送る前の確認)
4. [img2svg: 画像から線を取り出す](#img2svg-画像から線を取り出す)
5. [refine-passes: 線を画像に合わせて整える](#refine-passes-線を画像に合わせて整える)
6. [較正と補正の手順](#較正と補正の手順)
7. [ずれのモデルと補正の仕組み](#ずれのモデルと補正の仕組み)
8. [測定の例](#測定の例)
9. [ファイルの一覧](#ファイルの一覧)

## セットアップ

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt     # 依存パッケージと、このパッケージ（編集可能）
pytest                              # テスト
```

`refine-passes` には PyTorch が要ります（`pip install -e .[refine]`。CUDA 版は pytorch.org の手順で）。

## 絵を描くまで

コマンドは 3 段です。1 段目と 2 段目は線画の画像から SVG を作る段で、手元に SVG があれば 3 段目だけで描けます。

```
img2svg drawing.png --fill none --pen-width 0.46 -o lines.svg          # 1. 画像から線を取り出す
refine-passes drawing.png lines.svg --pen-width 0.46 -o refined.svg    # 2. ペン幅で描いた像が画像に合うように整える
svg2ring refined.svg --params fit_params.json --paper 210x297          # 3. G-code（refined.gcode）とプレビューを作る
python scripts/grbl_send.py COM3 refined.gcode                         # 送る
```

`svg2ring` のよく使うオプション（全部は `svg2ring --help`）:

```
svg2ring drawing.svg --scale 0.8 --offset 0 10 --rotate 90
svg2ring drawing.svg --fit                              # 届く範囲に収まるよう縮小
svg2ring drawing.svg --params fit_params.json           # 較正の結果（下の手順 1）を使う
svg2ring drawing.svg --deflection deflect_params.json --lead-in 3 --lead-out 0.15   # ずれの補正（下の手順 2〜3）
svg2ring drawing.svg --dwell 0.2 --dwell-up 0.1         # ペン上げ下げの待ち（秒）
svg2ring drawing.svg --keep-direction                   # 折れ線を逆向きにしない
ring-preview drawing.gcode --svg drawing.svg            # 既存の G-code を検査して描く
```

**座標系**: SVG の中心を輪の中心に合わせます。SVG の下が α = 0° の方向（手前）です。単位は mm
（SVG の width/height か viewBox から換算）。塗りは無視し、線だけを描きます。
文字は Inkscape でパスに変換しておきます（text 要素は飛ばして警告します）。

**描ける範囲**: 作者の機械では半径 1.5〜172 mm です。A4 の四隅には届きません。
届かない点は捨てて線をそこで切ります（`--unreachable error` で止めることもできます）。

**G-code の約束事**: X = A の角度 α、Y = B の角度 β（度）、G90。ペン上げは `M5`、下げは `M3 S110`、
そのあとに `G4 P<待ち>`。ストッパー原点（X0 Y88）で始めて、終わりにそこへ戻ります。

**送り速度**: ペン先が `--speed`（既定 15 mm/s）で動くように、区間ごとに F を決めます。
中心に近いほど台車が速く回る必要があるので、F は上限（`--fmax`、既定 1500 度/分）で頭打ちになり、ペン先は自動的に遅くなります。
F は 2 軸の合成速度です。GRBL の軸ごとの上限（$110/$111）が 2000 なら、2 軸が同じだけ動く移動では F2828 まで通ります。
作者はふだん `--speed 30 --fmax 2250 --accel 200` で描いています。

## 送る前の確認

1. `svg2ring` の出力（プレビューの PNG と統計）を見る。届かない線、α・β の範囲、所要時間
2. 実機で `G92 X0 Y88`（ストッパー原点）を送ってから、まずペンを外すか上げたまま一通り流す
3. 問題なければペンを付けて描く

`svg2ring` と `ring-preview` は、できあがった G-code を読み直して次を検査し、問題があれば終了コード 1 で一覧を出します。

- 2 台の角度差 Δ が、`grbl_send.py` のガード範囲（88〜180°）の外に出ないか
- ケーブルのねじれ: 原点からの回転角が ±400° に収まっているか（`--twist` で変更）
- 軸ごとの速度が GRBL の最大速度（`--max-rate`、$110/$111）を超えないか

**所要時間の推定**は、GRBL の加減速（`--accel` = $120/$121、`--junction` = $11、`--max-rate` = $110/$111）を入れた
台形速度で出します。実機の設定を変えたら、同じ値を渡してください。ずれの補正の速度係数もこの値を使うので、
実機と食い違うと線の両端の補正量が狂います。

**脱調の確認**は `--dry-run X Y` で行います。絵はペンを上げたまま流し、その前と後に (X, Y) へ 4 mm の十字を描くので、
2 つの十字が重なれば歩数を落としていません。

**G-code は ASCII だけにすること。** GRBL は 0x80 以上のバイトを、コメントの中であってもリアルタイムコマンド
（0x84 = セーフティドア、0x90 台 = オーバーライド）として扱います。日本語や「°」が混ざると `[MSG:Check Door]` で
止まったり、送り速度が変わったりします。`svg2ring` のヘッダは ASCII で出し、`grbl_send.py` はコメントを落としてから送り、
ASCII 以外が残っていれば止めます。

## img2svg: 画像から線を取り出す

`img2svg drawing.png` で `drawing.svg` と、確認用の `drawing_trace.png`（元画像に結果を重ねたもの）を出します。

- 二値化は Otsu で自動（`--threshold` で指定、`--invert` で白線）
- `--fill-width`（既定 1.5 mm）より太い部分は「塗り」、細い部分は「線」に分ける。線は細線化して中心線にする
- 塗りの扱いは `--fill hatch|outline|none|skeleton`。`refine-passes` に渡すなら `none` でよい（塗りは 2 段目が作る）
- 線は交点で区切り、短いひげを落とし（`--spur`）、まっすぐ続く線どうしをつなぎ（`--join`）、
  平滑化し（`--smooth`）、点を間引く（`--simplify`）。`--min-length`（既定 0.6 mm）未満の線は落とす
- 大きさは `--width`、`--height`、`--dpi` のどれか。指定がなければ半径 165 mm の円に収まる大きさ
- 出力の SVG は、線、平行パス、輪郭、ハッチングが別レイヤーなので、Inkscape で開いて直せる

`--fill hatch` で直接描く場合:

- ハッチングは 1 本おきに向きを変え、隣の線と端が近ければジグザグに 1 本につなぐ（`--hatch-join`、`0` でつながない）
- 塗りから細くなっていく先端は、`--fill-taper`（既定 15 mm）以内で終わるものを塗りに含める
- `--close`（既定 0.3 mm）で、細かい手描きハッチングの密集部を 1 つの塗りにまとめる。
  `--keep-gap`（既定 10 mm）以上続く細い隙間（二重線の間など）は埋めない

**太さの違う線**: `--pen-width`（ペンの線幅、mm）を与えると、ペン幅より太い線を、中心線に沿って横にずらした
平行パスで描きます（`--fill-width` は 4〜5 mm に上げて、本当の面だけを塗りにします）。

```
img2svg drawing.png --pen-width 0.4 --fill-width 4 --render
```

- 線の幅は距離変換で測り、線に沿って平滑化する（`--width-smooth`）。本数は、覆える幅がいちばん近くなる数
- 1 本目は `strokes` レイヤー、2 本目以降は `passes` レイヤー。`passes` を消せば単線の版に戻る
- `--render` で、全部の線をペン幅で描いた `_render.png`（紙の上の見え方の目安）も出す

## refine-passes: 線を画像に合わせて整える

`img2svg` の「線と塗りを分ける」やり方には、切断面に輪郭線が乗る、ひげが残る、本数の段差が出る、といった癖があります。
`refine-passes` は、`img2svg` の線を初期値にして、ペン幅で描いた像が元の画像に合うように折れ線の頂点を動かし、
要らない線を消し、足りないところに線を足します。A4 の絵 1 枚が、GPU（RTX 4090）で 20 秒ほどです。

```
img2svg drawing.png --fill none --pen-width 0.46 --threshold 210 -o lines.svg
refine-passes drawing.png lines.svg --pen-width 0.46 --fill-pitch 0.65 --threshold 210 -o refined.svg
```

仕組み:

- **像**: 各線分のまわりの小さな窓に、縁をぼかした線 c = sigmoid((w/2 − d)/τ) を描き、和集合 1 − Π(1 − c) で合成する。
  縁がぼけているので、頂点をどちらへ動かせば誤差が減るかが微分で求まる
- **目標**: 元の画像のグレースケール。ペン幅より細い線はどう描いてもペン幅になるので、細い線をペン幅まで太らせた
  「ペンの下限幅」を目標に入れる（これがないと、細い線は消したほうが誤差が小さいと判定される）
- **損失**: 二乗誤差 ＋ 形の保持（2 階差分の変化）＋ 伸び縮みの抑制。頂点だけを Adam で動かす
- **ラウンド**（`--rounds`、既定 3）: 最適化 → 削除 → 追加。削除するのは、消しても誤差がほとんど増えない線
  （`--remove-below`、`--lift-cost`）。追加は、足りないインクのうち太い部分は流線、細い部分は中心線、孤立した塊は点
- **流線の塗り**（`ring_plotter/hatch.py`）: まわりの線の接線から方向場を作り、その流線を等間隔に引く
  （Jobard–Lefer の方法）。人が塗るように、向きのそろった線になる。初期値の SVG に短い線を手で足しておくと、
  その向きに塗られる。`--fill-mode contour` は、縁に平行な線（等高線）で塗る以前の方式
- **塗りの中の間隔**（`--cover`、既定 1）: 二乗誤差だけだと、塗りの線がペン幅ぎりぎりの間隔まで間引かれ、
  描いたときのわずかなずれで白い筋が出る。塗りの内部を、少し細いペンで描いた像が埋めているかも損失に入れる
- **線の向き**: 隣り合う線を逆向きに描くと、向きで決まるずれが逆になって白い筋が出やすい。追加する線は向きをそろえる。
  そろえた向きのまま描くには `svg2ring --keep-direction` が要る（戻りの移動が増えるぶん、時間は 3 割ほど延びる）
- **出力**: SVG（初期値由来は `strokes`、追加分は `added` レイヤー）、`_render.png`（ペン幅で描いた像）、
  `_residual.png`（赤: 足りない、青: 余分）、`_target.png`（目標）

**ペン幅の決め方**: 実測値を使います（手順 3 で出る線幅か、`compare_drawing_scan.py` の墨の幅の中央値）。
隣り合う線は、向きをそろえて描いても線ごとに 0.1 mm 前後ずつ別々にずれるので、塗りの間隔はペン幅より 0.15 mm ほど狭くします
（線幅 0.48 mm のペンなら `--pen-width 0.46 --fill-pitch 0.65`）。向きがそろっていないと、すき間はさらに開きやすくなります。

## 較正と補正の手順

機械は図面どおりには組み上がらず、描く向きによってもペン先がずれます。そこで 2 つの層に分けて測ります。

- **第 1 層**: 向きに関係のないずれ。輪の半径 R、棒に沿った距離 d、横ずれ e を、点を打って合わせる
- **第 2 層**: 描く向きで変わるずれ。測定用の模様を描いて、補正の係数を出す

ペン、紙、ストッパー、レールの固定を変えたら、手順 1 から順に取り直します。スキャンは 600 dpi、白地に黒です
（A4 縦、紙の中心を輪の中心に）。下の例は、ファイルをすべて作業フォルダに置いた場合です。
描くときの速度と加速度は、あとの当てはめにも同じ値を渡します。

**手順 1: 第 1 層（R、d、e）** 点の格子を打ってスキャンし、当てはめます。

```
python scripts/gen_dots.py                                   # → dots.gcode、dots.csv
python scripts/grbl_send.py COM3 dots.gcode                  # 描いて dots.TIF にスキャン
python scripts/fit_dots.py dots.TIF dots.csv --ellipse -o fit
   # → fit_params.json、fit_residuals.csv、fit_residuals.png
   # 見る数字: rms_mm（0.13 程度）、楕円の大きさ（0.15 mm 程度）
```

設置が大きく変わっていなければ、`--init 前回の fit_params.json` で粗い探索を飛ばせます。
紙の隅の書き込みのような、点らしくない墨は先に落とします。

**手順 2: 向きで決まるずれ（a、b、λ、c_speed）** 補正なしのジグザグの模様を 2 枚描いて測り、まとめて当てはめます。

```
python scripts/gen_hatch_patches.py --params fit_params.json --pitch 2 --block 17 --grid-x -65 0 65 --grid-y -35 35 --speed 30 --fmax 2250 --accel 200 --dwell 0.1 --dwell-up 0.1 -o hatch.gcode
   # → hatch.gcode、hatch_layout.json。2 枚描いて hatch_1.TIF、hatch_2.TIF にスキャン
python scripts/measure_hatch_patches.py hatch_1.TIF --layout hatch_layout.json --params fit_params.json --plot hatch_1.png
python scripts/measure_hatch_patches.py hatch_2.TIF --layout hatch_layout.json --params fit_params.json --plot hatch_2.png
   # → hatch_1_patches.csv、hatch_2_patches.csv
python scripts/fit_deflection.py hatch_1_patches.csv hatch_2_patches.csv --params fit_params.json --model rod_axial_sin2_sqrt --speed-factor --speed 30 --fmax 2250 -o deflect_params.json
   # 見る数字: 残差 rms（0.015 程度）、2 枚を別々に当てはめたときの a の差
```

- 隣り合う線の間隔が p − 2D と p + 2D の交互になることを使って、ずれ D（進行方向の左が正）を測る
- 往復のずれが大きいと、狭い側の線がくっついて測れない。「融合」と警告が出たら、間隔を広げて描き直す
- 測定の最後に、紙 1 枚の健康診断（模様の中心の半径方向のずれと、間隔の倍率）が出る。
  同じ設置のまま半径が伸びていくなら、ペンが傾いてきているなど、機械のほうが変わっている

**手順 3: 線の両端（c_end、リードイン、リードアウト）** 補正を掛け、線ごとにペンを上げる模様を 1 枚描いて測ります。
c_end は当てはめる前なので、仮の値 0.4 を渡します。

```
python scripts/gen_hatch_patches.py --params fit_params.json --deflection deflect_params.json --deflection-c-end 0.4 --lead-in 3 --lead-in-min-travel 0 --lead-out 0.15 --separate alt --pitch 2 --block 17 --grid-x -65 0 65 --grid-y -35 35 --speed 30 --fmax 2250 --accel 200 --dwell 0.2 --dwell-up 0.1 -o sep_alt.gcode
   # 描いて sep_alt.TIF にスキャン
python scripts/measure_hatch_patches.py sep_alt.TIF --layout sep_alt_layout.json --params fit_params.json --deflection deflect_params.json --starts --ends
   # 見る数字: 線の中央のずれ D_mid の rms（0.015）と最大（0.05）、
   #           始点の曲がりが 0.15 mm を超える線の割合（5%）、終点の当てはめの切片、線幅の実測
python scripts/fit_end_effect.py sep_alt.TIF --params fit_params.json --deflection deflect_params.json --accel 200 --fit c_end --plot ends.png
   # → deflect_params_end.json（c_end を入れた、最終の係数ファイル）
```

`--lead-out` は、`--ends` が出す「終点の当てはめの切片」で決めます（+ は行き過ぎ、− は不足）。
この紙は 0.15 を入れて描いているので、**次の値 = 描いたときの値 − 切片** です
（切片 +0.05 なら 0.10、−0.08 なら 0.23、±0.03 程度なら 0.15 のまま）。

**手順 4: 絵を描く** 手順 1 と 3 の 2 つの JSON と、描いたときと同じ速度・加速度を渡します。

```
svg2ring refined.svg --params fit_params.json --deflection deflect_params_end.json --lead-in 3 --lead-out 0.15 --speed 30 --fmax 2250 --accel 200 --dwell 0.08 --dwell-up 0.08 --paper 210x297 --unreachable error
```

**手順 5: 結果の確認（任意）** 描いた絵をスキャンして、設計の SVG と比べます。

```
python scripts/compare_drawing_scan.py drawing.TIF refined.svg --params fit_params.json --dot-residuals fit_residuals.csv --gcode refined.gcode --plot compare.png
   # 見る数字: 線に直角な向きのずれ rms（0.13 程度。第 1 層の残りを引くと 0.09）、描いた向きごとの偏り
```

| 手順 | 描くもの | 出力 | 決まるもの |
| --- | --- | --- | --- |
| 1 | 点の格子 | fit_params.json | R、d、e |
| 2 | ジグザグの模様 2 枚（補正なし） | deflect_params.json | a、b、λ、c_speed |
| 3 | 線ごとにペンを上げる模様 1 枚（補正あり） | deflect_params_end.json | c_end、リードアウト、線幅 |
| 4 | 絵 | G-code、プレビュー | |
| 5 | | 絵のスキャンとの比較 | 残差の内訳 |

## ずれのモデルと補正の仕組み

**モデル**: ハッチングの線が往復で 0.4 mm ほどずれる現象は、機械の柔らかさと、向きで反転する摩擦の組み合わせです。
ペン先は棒の方向にだけずれ、その量は 2 つの係数で表せます。

```
ずれ = ( a·sin²(Δ/2)·g(v·u) + b·sign(ċ) )·u
```

v は進む向き、u は棒の向き、ċ は 2 台の間隔の伸び縮み、Δ は 2 台の角度差です。
g は、最初は g(x) = x（`--model rod_axial_sin2`）でしたが、実測は「小さい x で早く立ち上がって頭打ちになる」形だったので、
g(x) = sign(x)·√|x|（`rod_axial_sin2_sqrt`）にしています。向きを変えたあと、ずれは距離 λ（1〜2 mm）をかけて育ちます。

**補正**（`svg2ring --deflection`）: 経路を細かく歩きながら、ずれの状態 δ を δ += (δ目標 − δ)(1 − exp(−ds/λ)) で更新し、
目標点 − δ を逆運動学に渡します。ペンを上げた移動のあいだも状態は引き継ぎます。
プレビューは補正後の指令経路を描くので、SVG から最大 0.5 mm ずれて見えます。

**リードイン**（`--lead-in 3`）: ずれの状態は直前の移動の向きで決まるので、ペンを下ろした直後の線は、
前の移動の向きのずれから始まって曲がります。各線の始点の 3 mm 手前から、ペンを上げたまま線と同じ向きに入ることで、
これをほぼ消します。短い移動（`--lead-in-min-travel`）と、予測した振れが小さい線（`--lead-in-min-swing`）には入れません。

**速度の係数**（`c_speed`、`c_end`）: 加減速の区間では、速度とともにずれが小さくなります。
目標値に f(v) = 1 − c_speed·max(0, 1 − v/v_ref) を掛け、線の両端にはさらに (1 − c_end·max(0, 1 − v/v_c)) を掛けます。
速度は、GRBL の台形の加減速（`ring_plotter/timing.py`）を回して区間ごとに出します。
`fit_end_effect.py` は、描いたときの G-code を同じモデルで歩いて両端の形を再現し、実測に合う c_end を探します。

**リードアウト**（`--lead-out 0.15`）: 台車が止まっても、ペン先は紙に引きずられて少し後ろに残るので、
どの線も一律に少し短くなります。終点を、最後の向きにそのぶん延ばします。

## 測定の例

作者の機械での経過です。数字の大きさと、何が効いたかの目安として載せます。

| 段階 | 結果 |
| --- | --- |
| 最初の測定（113 か所） | a = +0.53、b = +0.08 mm、λ = 1.9 mm。残差 rms 0.03 mm（補正なしなら 0.16）。sin² を掛けない形では、中心に近い側で補正が 3 割ほど足りない |
| レールを固定し直したあと | a = +0.59、b = +0.08、λ = 1.4。補正を掛けた紙の線の中央の残差は rms 0.036、最大 0.10 mm |
| g を √ の形に変更 | 係数の数は同じまま、当てはめの rms が 0.038 → 0.015 mm。補正して描いた紙の中央の残差は rms 0.014、最大 0.047 mm |
| リードイン 3 mm | 始点が 0.15 mm 以上曲がる線が 30% → 10% |
| 速度の係数 | c_speed = 0.17。遅い中心付近の残差が消える。線の両端には足りなかったので c_end（0.4 前後）を分けた |
| 最終の設定 | a = 0.444、b = 0.099、λ = 0.93、c_speed 0.15、c_end 0.4、リードイン 3、リードアウト 0.15。線の中央のずれは最大 0.052 mm、始点が 0.15 mm 以上曲がる線は 5%、終点の不足は +0.002 mm |

わかったこと:

- a の符号は正で、ペン先は進む向きに「遅れる」のではなく「進む」。摩擦で引きずられるだけでは説明できない
- 同じ模様を 2 枚描いたときの差は rms 0.004〜0.03 mm。残差はおおむね測定の再現性の範囲
- 振動を抑えようとレールの溝にテープを貼った日は、ずれが全体に 1.4 倍になった。機械が柔らかくなると、そのぶんずれる
- 線幅は公称より太いことがある（公称 0.5 mm のペンで実測 0.72〜0.74 mm）。端点の位置を測るときは実測の線幅を使う
- 加速度を 100 → 200 度/秒² にすると、測定用の模様 1 枚が 10 分 22 秒 → 9 分 15 秒、絵 1 枚が 19 分 32 秒 → 16 分 59 秒（推定）。300 以上は頭打ち
- ペン先を 30 → 50 mm/s にすると、リードアウトの適正値も変わる（0.15 → 0.10 程度）

## ファイルの一覧

**ring_plotter/**

| ファイル | 役割 |
| --- | --- |
| `kinematics.py` | 順運動学と逆運動学（閉じた式）、届く範囲。R、d、e の既定値 |
| `svg_input.py` | SVG を読み、transform を適用した折れ線（mm）にする |
| `gcode.py` | 折れ線 → G-code。細分化、逆運動学、並べ替え、回り方の選択、区間ごとの F、リードイン・リードアウト |
| `simulate.py` | G-code を読み、順運動学で紙の上の線に戻す。ガード、ねじれ、速度の検査 |
| `timing.py` | GRBL の加減速（台形速度、junction deviation）を入れた所要時間の推定 |
| `deflection.py` | 向きで決まるずれの測定用の模様、スキャンの読み取り、力のモデル |
| `endfit.py` | 線の両端の係数の当てはめ（G-code を速度つきで歩いて、ずれを再現する） |
| `preview.py` | プレビューの描画と `ring-preview` コマンド |
| `svg2ring.py` | `svg2ring` コマンド |
| `raster.py`、`img2svg.py` | 線画の画像 → 折れ線。二値化、線と塗りの分離、細線化、平滑化、ハッチング |
| `refine_passes.py` | `refine-passes` コマンド。縁をぼかした描画器で、折れ線を画像に合わせる |
| `hatch.py` | 流線の塗り。まわりの線の向きから方向場を作り、等間隔の流線を引く |

**scripts/**

「検証」の番号は、組み立てた直後に確かめた順です。スクリプトの `--help` に出る「手順 N」は、この番号を指します
（上の「較正と補正の手順」の番号とは別です）。

| ファイル | 検証 | 役割 |
| --- | --- | --- |
| `grbl_send.py` | 共通 | GRBL への送信。2 台の角度差が範囲を外れる命令は送らない |
| `gen_circles.py` | 1 | 角度差を固定して 2 台をいっしょに回し、同心円を描く |
| `gen_arcs.py` | 2 | A を止めて B だけ動かし、円弧を描く |
| `gen_dots.py`、`fit_dots.py` | 3 | 点の格子を打ち、スキャンから R、d、e を合わせる |
| `gen_shapes.py` | 4 | 正方形と円を描いて確かめる |
| `gen_overdraw.py`、`measure_overdraw.py` | 5 | 同じ線を往復で重ね描きし、帯の幅（往復のずれ、符号なし）を測る |
| `gen_hatch_patches.py`、`measure_hatch_patches.py`、`fit_deflection.py` | 6 | ジグザグの模様を描き、符号つきのずれを測って、補正の係数を出す |
| `fit_end_effect.py` | 6 | 線の両端の係数 c_end を当てはめる |
| `compare_drawing_scan.py` | 絵 | 描いた絵のスキャンを設計の SVG と重ね、線ごとのずれを測る |
