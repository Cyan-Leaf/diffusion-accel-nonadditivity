#!/usr/bin/env python3
"""check_report_counts.py — 收尾检查：报告里的计数必须由表格生成，不得手工递增（T14 §3）

## 为什么存在

`REPORT.md` §10.4 的标题「N 次自我更正，产出 X 条通则 + Y 条可复用机制」
**数错过两次**：先是「3 条」应为「4 条」，改成 4 之后又加了三条机制、
标题递增成了 8，而表里只有 7。

**错因是同一个，而且它就是通则 8**：
> 计数是一个**派生量**。加了行之后「递增」了它，而没有「重数」。
> 通则 8 的原话是「修正基线时必须列出所有以它为分母的派生量，逐个确认」——
> **那张表就是基线，标题里的两个数就是派生量。**

→ **通则 8 补一句：表格的计数必须由表格生成，不得手工递增。**
一行 `grep -c` 就能算的东西，不要用脑子加。

## 它检查什么

1. §10.4 标题里的三个数 vs 表格实际行数 / 「通则 N」出现数 / 「机制」出现数
2. 「过程型 ⚪ / 边界型 🔴」两栏列出的条目数之和 = 表行数（T14 §3.1 指出漏了 #8 与 #18）
3. 两栏之间没有重复编号、没有超出范围的编号
4. 所有 `![...](path)` 引用的图都存在
5. 所有 `results/**/*.json` 可解析、所有脚本可 `ast.parse`

**退出码非 0 即有不一致** —— 可以挂进任何提交前的检查。

用法：
    python3 benchmarks/check_report_counts.py
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    R = open(os.path.join(REPO, "REPORT.md")).read()
    problems = []

    # ---- 1. §10.4 的表 ---------------------------------------------------
    i0 = R.index("### 10.4 ")
    i1 = R.index("\n## ", i0)
    sec = R[i0:i1]
    title = sec.split("\n", 1)[0]

    # 表格行：以 "| " 开头且第二格是编号（数字，可带 b/c 后缀与 ** 与 emoji）
    rows = re.findall(r"^\|\s*\*{0,2}(\d+[a-z]?)\*{0,2}\s*(?:🔴|⚪)?\s*\|", sec, re.M)
    n_rows = len(rows)
    # 右列里引用了哪些既有通则、以及有多少条标 **机制**
    n_mech = len(re.findall(r"\*\*机制\*\*", sec))
    guidelines = sorted(set(int(x) for x in re.findall(r"\*\*通则 (\d+)\*\*", sec)))

    m = re.search(r"(\d+)\s*次自我更正，产出\s*(\d+)\s*条通则\s*\+\s*(\d+)\s*条可复用机制", title)
    if not m:
        problems.append(f"§10.4 标题格式不认识，无法校验：{title!r}")
    else:
        t_n, t_g, t_m = (int(x) for x in m.groups())
        print(f"§10.4 标题：{t_n} 次更正 / {t_g} 条通则 / {t_m} 条机制")
        print(f"§10.4 表格：{n_rows} 行 / 引用通则 {guidelines} ({len(guidelines)} 条) / "
              f"标 **机制** {n_mech} 条")
        if t_n != n_rows:
            problems.append(f"更正次数：标题 {t_n} ≠ 表行数 {n_rows}")
        if t_g != len(guidelines):
            problems.append(f"通则条数：标题 {t_g} ≠ 表里引用的不同通则数 {len(guidelines)}"
                            f" {guidelines}")
        if t_m != n_mech:
            problems.append(f"机制条数：标题 {t_m} ≠ 表里标 **机制** 的行数 {n_mech}")

    # ---- 2. 过程型 / 边界型 的分类覆盖 -----------------------------------
    def listed(label):
        """只取每个「、」分段**开头**的编号 —— 分段正文里也有数字（§8.5、0.12、1.107），
        整段 findall 会把它们当编号，这个坑在第一版校验器里真的踩到了。"""
        mm = re.search(r"\*\*" + label + r"[^|]*\|[^|]*\|([^|]*)\|", sec)
        if not mm:
            return None
        out = set()
        for frag in mm.group(1).split("、"):
            g = re.match(r"\s*\*{0,2}(\d+[a-z]?)\*{0,2}", frag)
            if g:
                out.add(g.group(1))
        return out

    proc = listed("过程型")
    bound = listed("⚠️ 边界型")
    if proc is None or bound is None:
        problems.append("找不到「过程型 / 边界型」两栏，无法校验分类覆盖")
    else:
        allrows = set(rows)
        print(f"分类：过程型 {len(proc)} 条 + 边界型 {len(bound)} 条 = "
              f"{len(proc | bound)} 条（表行数 {n_rows}）")
        miss = allrows - (proc | bound)
        extra = (proc | bound) - allrows
        dup = proc & bound
        if miss:
            problems.append(f"未分类的条目：{sorted(miss)}")
        if extra:
            problems.append(f"分类里出现了表中没有的编号：{sorted(extra)}")
        if dup:
            problems.append(f"同时出现在两栏的编号：{sorted(dup)}")

    # ---- 3. 图片引用 -----------------------------------------------------
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", R):
        if not os.path.exists(os.path.join(REPO, rel)):
            problems.append(f"图片引用不存在：{rel}")

    # ---- 4. json / 脚本可解析 -------------------------------------------
    njson = 0
    for f in glob.glob(os.path.join(REPO, "results/**/*.json"), recursive=True):
        njson += 1
        try:
            json.load(open(f))
        except Exception as e:
            problems.append(f"json 无法解析：{os.path.relpath(f, REPO)} ({e})")
    nscript = 0
    for pat in ("benchmarks/*.py", "quant/*/*.py", "fusion/*.py", "sparse/*.py",
                "eval/*.py", "figures/*.py"):
        for f in glob.glob(os.path.join(REPO, pat)):
            nscript += 1
            try:
                ast.parse(open(f).read())
            except Exception as e:
                problems.append(f"脚本语法错误：{os.path.relpath(f, REPO)} ({e})")
    n_img = len(re.findall(r"!\[[^\]]*\]\(", R))
    print(f"产物：{njson} json + {nscript} scripts，图片引用 {n_img} 处")

    if problems:
        print("\n=== 不一致 ===")
        for x in problems:
            print("  ✗ " + x)
        sys.exit(1)
    print("\n=== 全部一致 ===")


if __name__ == "__main__":
    main()
