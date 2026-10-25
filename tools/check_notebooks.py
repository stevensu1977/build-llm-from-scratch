"""不用启动 Jupyter，也能逐格执行各章默认开启的教学代码。

需要先安装 requirements.txt。第 5 章的大模型训练和权重下载默认关闭。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")  # 检查脚本在终端里运行，不弹出绘图窗口。

for chapter in ("chapter02", "chapter03", "chapter04", "chapter05"):
    path = ROOT / chapter / f"{chapter}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook["nbformat"] != 4:
        raise ValueError(f"{path} 不是 notebook 格式 4")
    ids = [cell["id"] for cell in notebook["cells"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} 有重复的 cell id")
    namespace: dict = {"__name__": "__main__"}
    print(f"\n正在执行 {chapter}，代码单元数：{sum(c['cell_type'] == 'code' for c in notebook['cells'])}")
    for i, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"{path}:cell{i}", "exec"), namespace)
        except Exception as exc:
            raise RuntimeError(f"{chapter} 第 {i} 格执行失败") from exc
    print(f"{chapter}：通过")
