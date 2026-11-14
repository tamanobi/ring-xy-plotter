"""scripts/grbl_send.py の送信前処理（シリアルポートは使わない）。"""
import os
import signal
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from grbl_send import Grbl, GuardError, run_file, run_interactive  # noqa: E402


class FakeSerial:
    def __init__(self):
        self.sent = []

    def write(self, data):
        self.sent.append(data)

    def readline(self):
        return b"ok\r\n"


def make():
    g = Grbl.__new__(Grbl)
    g.dmin, g.dmax = 88.0, 180.0
    g.pos, g.absolute = None, True
    g.unacked = 0
    g.ser = FakeSerial()
    return g


class ScriptedSerial:
    """応答を順に返す。"INT" は Ctrl+C を発生させて空行を返す（受信待ちの中断を模す）。

    GRBL の状態は MPos/WCO を固定で返し、Hold 完了までに数回かかるようにする。
    """

    def __init__(self, replies=(), reports=()):
        self.sent = []
        self.replies = list(replies)
        self.reports = list(reports)
        self.pending = []
        self.was_reset = False

    def write(self, data):
        self.sent.append(data)
        if data == b"?" and self.reports:
            # 最後のレポートは返し続ける
            self.pending.append(self.reports.pop(0) if len(self.reports) > 1 else self.reports[0])
        if data == b"\x18":
            self.was_reset = True
            self.pending += ["ok", "Grbl 1.1h ['$' for help]"]
        if data.startswith(b"G92") and self.was_reset:
            self.pending.append("ok")

    def readline(self):
        if self.pending:
            return (self.pending.pop(0) + "\r\n").encode()
        if self.replies:
            r = self.replies.pop(0)
            if r == "INT":
                signal.raise_signal(signal.SIGINT)
                return b""
            return (r + "\r\n").encode()
        return b""

    def read_all(self):
        out = "".join(p + "\r\n" for p in self.pending + self.replies)
        self.pending, self.replies = [], []
        return out.encode()


def test_ctrl_c_during_file_returns_false_holds_and_keeps_lines_unsent(tmp_path, capsys):
    path = tmp_path / "a.gcode"
    path.write_text("G92 X0 Y88\nG1 X1 Y89 F100\nG1 X2 Y90 F100\nG1 X3 Y91 F100\n")
    g = make()
    g.ser = ScriptedSerial(["ok", "ok", "INT"])  # 3行目の応答待ちで Ctrl+C
    assert run_file(g, str(path)) is False
    assert b"!" in g.ser.sent
    assert b"G1 X3 Y91 F100\n" not in g.ser.sent
    assert g.pos == (2.0, 90.0)  # 3行目は送信済みなので、再開すればそこへ行く
    assert g.unacked == 1
    assert "3/4" in capsys.readouterr().out


def test_late_ok_of_interrupted_line_is_not_taken_for_the_next_line():
    g = make()
    g.ser = ScriptedSerial(["ok", "INT"])
    g.send("G92 X0 Y88")
    with pytest.raises(KeyboardInterrupt):
        g.send("G1 X1 Y89 F100")
    # 再開後、中断した行の ok → 次の行のエラー。次の行の結果はエラーでなければならない
    g.ser.replies = ["ok", "error:2"]
    assert g.send("G1 X5 Y95 F100") is False
    assert g.unacked == 0
    assert g.pos == (1.0, 89.0)


def test_sigint_handler_is_restored_after_send():
    g = make()
    prev = signal.getsignal(signal.SIGINT)
    g.send("G92 X0 Y88")
    assert signal.getsignal(signal.SIGINT) is prev


def test_discard_waits_for_hold_then_restores_work_position():
    g = make()
    g.pos, g.unacked = (9.0, 99.0), 1
    g.ser = ScriptedSerial(reports=[
        "<Run|MPos:1.000,2.000,0.000|FS:100,0>",
        "<Hold:1|MPos:3.000,4.000,0.000|FS:50,0>",
        "<Hold:0|MPos:5.000,6.000,0.000|FS:0,0>",  # 止まったが WCO がまだ載らない
        "<Hold:0|MPos:5.000,6.000,0.000|FS:0,0|WCO:-10.000,-100.000,0.000>",
    ])
    g.discard()
    assert g.ser.sent[0] == b"!"
    assert g.ser.sent.index(b"\x18") == 5  # ! と ? 4回のあと
    assert g.ser.sent[-1] == b"G92 X15.000 Y106.000\n"
    assert g.pos == (15.0, 106.0)
    assert g.absolute is True
    assert g.unacked == 0


def test_discard_without_position_requires_new_g92(capsys):
    g = make()
    g.pos = (9.0, 99.0)
    g.ser = ScriptedSerial(reports=["<Hold:0|MPos:5.000,6.000,0.000|FS:0,0>"])
    g.discard()
    assert g.pos is None
    assert not any(s.startswith(b"G92") for s in g.ser.sent)
    assert "G92" in capsys.readouterr().out


def test_ctrl_c_at_prompt_does_not_hold_when_idle():
    g = make()
    g.ser = ScriptedSerial(reports=["<Idle|MPos:0.000,0.000,0.000|FS:0,0>"])
    g.stop(only_if_moving=True)
    assert b"!" not in g.ser.sent


def test_ctrl_c_at_prompt_holds_when_running():
    g = make()
    g.ser = ScriptedSerial(reports=["<Run|MPos:0.000,0.000,0.000|FS:100,0>"])
    g.stop(only_if_moving=True)
    assert b"!" in g.ser.sent


def test_interactive_stays_at_prompt_after_ctrl_c_in_file(tmp_path, monkeypatch):
    path = tmp_path / "a.gcode"
    path.write_text("G92 X0 Y88\nG1 X1 Y89 F100\n")
    g = make()
    g.ser = ScriptedSerial(["ok", "INT"])
    inputs = iter([f"@{path}", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    run_interactive(g)  # 例外で抜けず、q まで到達する
    assert list(inputs) == []
    assert b"!" in g.ser.sent


def test_comment_lines_are_not_sent():
    g = make()
    assert g.send("; svg2ring 2000-01-01 (日本語のコメント) 届く半径 1.5〜172 mm")
    assert g.send("(comment only)")
    assert g.send("   ")
    assert g.ser.sent == []


def test_inline_comment_is_stripped_before_sending():
    g = make()
    assert g.send("G92 X0 Y88 ; 原点")
    assert g.ser.sent == [b"G92 X0 Y88\n"]
    assert g.pos == (0.0, 88.0)


def test_non_ascii_code_is_refused():
    g = make()
    with pytest.raises(GuardError):
        g.send("G1 X10 Y100 F1000°")
    assert g.ser.sent == []


def test_delta_guard_still_applies():
    g = make()
    g.send("G92 X0 Y88")
    with pytest.raises(GuardError):
        g.send("G1 X0 Y200 F1000")
    assert g.ser.sent == [b"G92 X0 Y88\n"]
