#!/usr/bin/env python3
"""リング式プロッタ用のGRBL送信スクリプト。

X = ユニットAの角度α、Y = ユニットBの角度β（どちらも度）。
移動先の Δ = Y - X が [dmin, dmax] を外れる行は送らない。

  python grbl_send.py COM3                 対話モード
  python grbl_send.py COM3 file.gcode      ファイル送信
  python grbl_send.py COM3 --dmin 88 --dmax 180

原点合わせ: Bをストッパー（カラー）に触れる寸前まで寄せて G92 X0 Y88。
ジョグに G91 を使ったら、G90 に戻すこと（プロンプトに現在のモードを表示する）。

対話モードの特別な入力:  q 終了 / ? 状態表示 / ! 一時停止 / ~ 再開
                          x 破棄（減速停止→リセットでバッファを捨てペンを上げ、座標は引き継ぐ）

送信中や移動中の Ctrl+C は一時停止して対話モードに戻る。
そのあとは ~（GRBL に送信済みの分だけ続行）か x（破棄）を選ぶ。ファイルの残りは送らない。
"""
import argparse
import re
import signal
import sys
import time

import serial

WORD = re.compile(r"([A-Z])\s*(-?\d+\.?\d*|-?\.\d+)")
UNSUPPORTED = {10, 28, 30, 53}  # 位置の追跡が狂うので通さない


MOVING = ("Run", "Jog")


class GuardError(Exception):
    pass


def report_state(report):
    """'<Hold:0|MPos:...>' → 'Hold:0'"""
    return report.strip("<>").split("|")[0]


def report_wpos(report, wco=None):
    """状態レポートから作業座標 (x, y) を求める。返り値は (作業座標 or None, 分かっているWCO)。"""
    fields = dict(f.split(":", 1) for f in report.strip("<>").split("|")[1:] if ":" in f)
    xy = lambda key: tuple(float(v) for v in fields[key].split(",")[:2])
    if "WCO" in fields:
        wco = xy("WCO")
    if "WPos" in fields:
        return xy("WPos"), wco
    if "MPos" in fields and wco is not None:
        mx, my = xy("MPos")
        return (mx - wco[0], my - wco[1]), wco
    return None, wco


class Grbl:
    def __init__(self, port, baud, dmin, dmax):
        self.dmin, self.dmax = dmin, dmax
        self.pos = None  # (x, y)。G92で設定するまで不明
        self.absolute = True  # G90
        self.unacked = 0  # 送ったがまだ応答が来ていない行の数（Ctrl+C で待つのをやめた行）
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)  # ポートを開くとArduinoがリセットされる
        self.ser.write(b"\r\n\r\n")
        time.sleep(1)
        print(self.ser.read_all().decode(errors="replace").strip())

    # --- ガード -----------------------------------------------------------
    def check(self, line):
        """行を検査し、送ってよければ (新しい位置, 新しいG90/G91) を返す。"""
        code = re.sub(r"\(.*?\)", "", line.split(";")[0]).strip().upper()
        if code.startswith("$") and not code.startswith("$J="):
            return self.pos, self.absolute  # 設定コマンドはそのまま通す
        jog = code.startswith("$J=")
        words = WORD.findall(code[3:] if jog else code)
        gs = {float(v) for k, v in words if k == "G"}
        if gs & UNSUPPORTED:
            raise GuardError("G10/G28/G30/G53 は未対応です")

        absolute = self.absolute
        if 90 in gs:
            absolute = True
        if 91 in gs:
            absolute = False
        x = next((float(v) for k, v in words if k == "X"), None)
        y = next((float(v) for k, v in words if k == "Y"), None)
        new_abs = self.absolute if jog else absolute  # ジョグのG90/G91はその行限り

        if x is None and y is None:
            return self.pos, new_abs

        if 92 in gs:  # 座標の設定
            if self.pos is None and (x is None or y is None):
                raise GuardError("最初のG92はXとYの両方を指定してください")
            px, py = self.pos if self.pos else (0.0, 0.0)
            pos = (px if x is None else x, py if y is None else y)
            d = pos[1] - pos[0]
            if not self.dmin <= d <= self.dmax:
                print(f"  注意: 設定したΔ={d:.2f}°はガード範囲の外です")
            return pos, new_abs

        if self.pos is None:
            raise GuardError("位置が未設定です。先に G92 X.. Y.. を送ってください")
        px, py = self.pos
        if absolute:
            tx = px if x is None else x
            ty = py if y is None else y
        else:
            tx = px + (x or 0.0)
            ty = py + (y or 0.0)
        d = ty - tx
        if not self.dmin <= d <= self.dmax:
            raise GuardError(
                f"Δ={d:.2f}° が範囲外です（{self.dmin}〜{self.dmax}°）"
            )
        return (tx, ty), new_abs

    # --- 送信 -------------------------------------------------------------
    def send(self, line):
        """1行送り、ok なら True、error/ALARM なら False。"""
        # コメントは送らない。GRBL は 0x80 以上のバイトをリアルタイムコマンドとして解釈するので
        # （0x84 = セーフティドア、0x90 台 = オーバーライド）、日本語をそのまま送ると誤動作する
        line = re.sub(r"\(.*?\)", "", line.split(";")[0]).strip()
        if not line:
            return True
        if not line.isascii():
            raise GuardError(f"ASCII 以外の文字は送れません: {line!r}")
        pos, absolute = self.check(line)
        self.ser.write((line + "\n").encode())
        self.unacked += 1
        # 受信の途中で KeyboardInterrupt が上がると読みかけの ok が失われ、応答の数がずれる。
        # 受信中は印を付けるだけにして、1行読み終えてから抜ける（遅れは最大でも timeout の 0.1 秒）
        interrupted = []
        prev = signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
        result = None
        try:
            while result is None and not interrupted:
                resp = self.ser.readline().decode(errors="replace").strip()
                if resp and self._note(resp) and not self.unacked:
                    result = resp == "ok"
        finally:
            signal.signal(signal.SIGINT, prev)
        # 中断した場合も、この行はもう GRBL に届いていて再開すれば実行されるので、位置は進める。
        # その応答は後から来るので unacked に残る
        if result is not False:
            self.pos, self.absolute = pos, absolute
        if interrupted:
            raise KeyboardInterrupt
        return result

    def _note(self, resp):
        """受信した1行を処理する。行の完了を示す応答なら True。

        待つのをやめた行の ok は、次の行の応答と取り違えないようここで数えて捨てる。
        """
        done = resp == "ok" or resp.startswith("error") or resp.startswith("ALARM")
        if done and self.unacked:
            self.unacked -= 1
            if resp == "ok":
                return True
        print("  " + resp)
        return done

    def realtime(self, ch):
        self.ser.write(ch.encode())
        time.sleep(0.2)
        for resp in self.ser.read_all().decode(errors="replace").splitlines():
            if resp.strip():
                self._note(resp.strip())

    def status(self, timeout=0.5):
        """? を送り、状態レポートの行を返す。来なければ None。"""
        self.ser.write(b"?")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self.ser.readline().decode(errors="replace").strip()
            if resp.startswith("<"):
                return resp
            if resp:
                self._note(resp)
        return None

    def stop(self, only_if_moving=False):
        """Ctrl+C の処理。フィードホールドを送り、実際の状態を表示する。

        only_if_moving: 動いていなければ何もしない（Idle で ! を送ると Hold に入り、
        次の指令が ~ まで動かなくなるため。プロンプトでの Ctrl+C 用）。
        """
        try:
            if only_if_moving:
                st = self.status()
                if st and report_state(st) not in MOVING:
                    print("\n  動作中ではありません（終了は q）")
                    return
            self.ser.write(b"!")  # 最優先で止める
            print("\n  中断: 一時停止を送りました（~ 送信済みの分を続行 / x 破棄してペンを上げる）")
            time.sleep(0.2)
            st = self.status()
            if st:
                print("  " + st)
        except KeyboardInterrupt:  # 連打されても対話モードに戻す
            self.ser.write(b"!")

    def discard(self):
        """減速停止してからソフトリセットし、バッファを捨てる。

        リセットでペンは上がる（スピンドル停止）が、G92 も消えるので、
        リセット前の作業座標を読み取って設定し直す。Hold 完了前にリセットすると
        ALARM になり位置も失うので、止まるまで待つ。
        """
        self.ser.write(b"!")
        deadline = time.monotonic() + 15
        st = None
        stopped = False
        while time.monotonic() < deadline:
            time.sleep(0.05)
            st = self.status()
            if st and report_state(st) not in MOVING + ("Hold:1",):
                stopped = True
                break
        # WCO は数十回に1回しかレポートに載らない
        wpos, wco = None, None
        for _ in range(40 if stopped else 0):
            wpos, wco = report_wpos(st, wco)
            if wpos:
                break
            time.sleep(0.02)
            st = self.status() or st

        self.ser.write(b"\x18")
        self.unacked = 0
        self.pos, self.absolute = None, True  # リセットで G92 は消え G90 に戻る
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            resp = self.ser.readline().decode(errors="replace").strip()
            if resp.startswith("Grbl"):
                print("  " + resp)
                break
        else:
            print("  リセットの応答がありません")
            return
        time.sleep(0.1)  # 起動メッセージの残り（ALARM など）を受ける
        for resp in self.ser.read_all().decode(errors="replace").splitlines():
            if resp.strip():
                print("  " + resp.strip())

        if wpos is None:
            print("  現在位置を読めませんでした。G92 X.. Y.. で設定し直してください")
            if st:
                print("  " + st)
            return
        if self.send(f"G92 X{wpos[0]:.3f} Y{wpos[1]:.3f}"):
            print(f"  破棄しました。座標を引き継ぎました: {self.describe()}")
        else:
            print(f"  G92 に失敗しました。止まった位置は X={wpos[0]:.3f} Y={wpos[1]:.3f} です")

    def describe(self):
        if self.pos is None:
            return "位置未設定"
        x, y = self.pos
        return f"X={x:.3f} Y={y:.3f} Δ={y - x:.3f} {'G90' if self.absolute else 'G91 相対'}"


def run_file(grbl, path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    n = 0
    try:
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            print(f"[{n}/{len(lines)}] {line.strip()}")
            try:
                if not grbl.send(line):
                    print("エラーで停止しました")
                    return False
            except GuardError as e:
                print(f"ガードで停止: {e}")
                return False
    except KeyboardInterrupt:
        grbl.stop()
        print(f"  ファイル送信を {n}/{len(lines)} 行目で打ち切りました")
        return False
    print("送信完了（動作の終了は ? で確認）: " + grbl.describe())
    return True


def run_interactive(grbl):
    print("対話モード。q で終了、? で状態表示、x で破棄")
    while True:
        try:
            line = input(f"[{grbl.describe()}] > ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:  # プロンプトでも、送信済みの移動が続いていれば止める
            grbl.stop(only_if_moving=True)
            continue
        if line == "q":
            break
        try:
            if line in ("?", "!", "~"):
                grbl.realtime(line)
            elif line == "x":
                grbl.discard()
            elif line.startswith("@"):  # @file.gcode でファイルを流す
                run_file(grbl, line[1:].strip())
            else:
                grbl.send(line)
        except GuardError as e:
            print(f"  送信しません: {e}")
        except KeyboardInterrupt:
            grbl.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("port")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dmin", type=float, default=88.0, help="Δの下限（ストッパー位置）")
    ap.add_argument("--dmax", type=float, default=180.0)
    args = ap.parse_args()

    grbl = Grbl(args.port, args.baud, args.dmin, args.dmax)
    try:
        if args.file:
            run_file(grbl, args.file)
        run_interactive(grbl)  # ファイル送信後も対話に残る（ポートを閉じるとリセットされるため）
    finally:
        grbl.ser.close()


if __name__ == "__main__":
    sys.exit(main())
