from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


LENGTH_BIN_KEYS = ["1", "2", "3-5", "6-10", "11+"]


def length_bin(gt_len: int) -> str:
    gt_len = int(gt_len)
    if gt_len == 1:
        return "1"
    if gt_len == 2:
        return "2"
    if gt_len <= 5:
        return "3-5"
    if gt_len <= 10:
        return "6-10"
    return "11+"


def levenshtein_ops(a: Iterable[int], b: Iterable[int]) -> tuple[int, int, int, int]:
    a = [int(x) for x in a]
    b = [int(x) for x in b]
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "eq"
            else:
                candidates = [
                    (dp[i - 1][j] + 1, "del"),
                    (dp[i][j - 1] + 1, "ins"),
                    (dp[i - 1][j - 1] + 1, "sub"),
                ]
                dp[i][j], op[i][j] = min(candidates, key=lambda item: item[0])

    i, j = n, m
    insertions = deletions = substitutions = 0
    while i > 0 or j > 0:
        step = op[i][j]
        if step == "eq":
            i -= 1
            j -= 1
        elif step == "sub":
            substitutions += 1
            i -= 1
            j -= 1
        elif step == "del":
            deletions += 1
            i -= 1
        elif step == "ins":
            insertions += 1
            j -= 1
        else:
            break

    return dp[n][m], insertions, deletions, substitutions


@dataclass
class CtcTotals:
    n: int = 0
    gt_chars: int = 0
    pred_chars: int = 0
    dist: int = 0
    ins: int = 0
    dels: int = 0
    subs: int = 0
    empty_pred: int = 0

    def add(self, gt_len: int, pred_len: int, dist: int, ins: int, dels: int, subs: int) -> None:
        self.n += 1
        self.gt_chars += int(gt_len)
        self.pred_chars += int(pred_len)
        self.dist += int(dist)
        self.ins += int(ins)
        self.dels += int(dels)
        self.subs += int(subs)
        self.empty_pred += int(int(pred_len) == 0)

    def to_summary(self) -> dict:
        return {
            "samples": self.n,
            "cer_micro": self.dist / max(self.gt_chars, 1),
            "avg_gt_len": self.gt_chars / max(self.n, 1),
            "avg_pred_len": self.pred_chars / max(self.n, 1),
            "pred_gt_len_ratio": self.pred_chars / max(self.gt_chars, 1),
            "empty_pred_rate": self.empty_pred / max(self.n, 1),
            "ins_rate": self.ins / max(self.gt_chars, 1),
            "del_rate": self.dels / max(self.gt_chars, 1),
            "sub_rate": self.subs / max(self.gt_chars, 1),
        }


def summarize_by_length(rows: Iterable[dict]) -> dict[str, dict]:
    by_bin: dict[str, CtcTotals] = defaultdict(CtcTotals)
    for row in rows:
        bucket = length_bin(int(row["gt_len"]))
        by_bin[bucket].add(
            gt_len=int(row["gt_len"]),
            pred_len=int(row["pred_len"]),
            dist=int(row["dist"]),
            ins=int(row["ins"]),
            dels=int(row["dels"]),
            subs=int(row["subs"]),
        )
    return {key: by_bin[key].to_summary() for key in LENGTH_BIN_KEYS if by_bin[key].n > 0}
