# 少步视频扩散模型的推理加速：四条轴的收益为什么不可加

在一个 **4 步蒸馏**的视频扩散模型（Wan2.1-T2V-1.3B，单卡 RTX 4090）上装齐四条推理加速轴
——**步数蒸馏、FP8 W8A8 量化、结构化稀疏 attention、算子融合**——
并测量它们的收益**为什么不能相乘**。

📄 **完整报告：[`REPORT.md`](REPORT.md)**（摘要 + 十节 + 附录，含 5 张图与 22 条自我更正）

课程作业（生成模型基础）。

---

## 三条主结论

**1. 否证了最直觉的那个解释：手段之间并不争抢。**
FP8 量化与结构化稀疏**精确不相交**且**超乘性**复合 ——
交互比 `I = c/(a·b) = 1.0435`，与「完全不相交」解析模型的预测 **1.0431 相差 0.0004**，
在 4 步与 50 步下均成立。

**2. 差额可以逐项拆开，残差为零。**

```
155.86x  朴素乘积（25.11 阶段 x 1.895 算子 x 3.28 算子）
  /1.7475   FP8：裸 GEMM -> denoise
  /2.2005   稀疏：mask kernel -> denoise
  x1.0134   compile（口径项，不属三成因）
  x1.0435   ⭐ 超乘 —— 唯一往回加的因子
 42.86x  denoise 段实测        残差 +0.00000%
  /2.6287   VAE decode + text encode 在所有轴的作用域之外
 16.30x  全链路实测 = 朴素乘积的 10.5%
```

⚠️ **残差为零是恒等式，不是检验** —— 这张表的内容在于每一项的归因，不在于它闭合。

**3. 精度轴的天花板与硬件无关。**
由实测占比（GEMM 占 denoise **20.43%**）给出、**对任意快的 GEMM 都成立**的
Amdahl 上界是 **1.2568x**；FP8 已实测拿到其 **86.3%**。
→ **剩下的不是一个更低的数值格式，而是 79.6% 的非 GEMM 时间。**

---

## 目录

| 路径 | 内容 |
|---|---|
| **`REPORT.md`** | **主交付物**。所有结论、算法原理（§4.4 GPTQ）、自我更正清单（§10.4） |
| `figures/` | F1 瓶颈迁移 / F2 融合折线 / **F3 瀑布（主图）** / F4 指数位曲线 + 生成脚本 |
| `results/frames/` | 眼评用的带标注对照图（九臂 × 三 prompt，每格标 arm 名与该臂的 PSNR / latent L2 / VBench） |
| `results/E0…E9/` | 全部原始 json 产物 |
| `benchmarks/` | 各实验的复算脚本 |
| `eval/` | VBench、逐视频配对分解、跨臂距离、对照图生成 |
| `quant/`、`fusion/`、`sparse/` | FP8 量化 / GPTQ 求解器、三级融合、SVG 稀疏接入 |

## 复现入口

```bash
export SF_ROOT=<上游 Wan 推理工作目录>          # 含 wan/、utils/、prompts/
export DISTILL_CKPT=<distill_native.pt 的路径>
export PYTHONPATH=$SF_ROOT:$PWD

python3 benchmarks/e5_fullchain.py          # 瀑布（纯算术，不跑模型）
python3 benchmarks/e7_nvfp4_projection.py   # 三个 Amdahl 上界（同上）
python3 benchmarks/check_report_counts.py   # 报告内部一致性检查
```

需要 GPU 的实验（`e5_ladder.py`、`e2_quality.py`、`e4_svg.py` 等）见各脚本的文件头。

⚠️ **测量口径两条**：报 attention 数字必须显式指定 SDPA backend
（flash 40.45 ms vs efficient 62.43 ms，差 54%）；报显存必须每配置独立进程
（同进程连跑会让峰值显存偏小 5.6 倍）。

## 模型权重

4 步蒸馏权重取自 **lightx2v** 公开发布的 `Wan2.1-T2V-1.3B-Distill-Models`
（完整 BF16 权重，revision `ef72050`），**不是 LoRA**。
本文不自行蒸馏 —— 用公开权重使基线与第三方完全一致、结果可逐位复现。

稀疏轴使用 **SVG**（Sparse VideoGen，arXiv 2502.01776）的 mask 设计与 head 分类准则，
**kernel 后端换成 FlexAttention**（SVG 自己的第二后端）。不声称复现 SVG 的端到端数字。

## 未收进仓库的

权重与 Hessian（约 45 GB）、生成的 mp4（331 MB，报告只引用 `results/frames/*.png`）。
两者都可由 `benchmarks/` 下的脚本重新生成。
