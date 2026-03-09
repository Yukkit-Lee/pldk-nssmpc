1 宏观研究意义（建议这样讲）

PLDK 原论文解决的是：

Data-Free / Distillation-based Federated Learning

核心思想：

不共享数据
↓
生成蒸馏数据
↓
教师模型蒸馏学生模型

但是 PLDK仍然存在隐私风险。

主要风险：

问题1：logits 泄露

教师模型输出：

teacher logits

这些 logits 可以被攻击者利用进行：

Membership Inference Attack (MIA)

或者

Model Inversion Attack

因为：

logits 包含模型决策边界信息

所以：

PLDK 并不是完全隐私安全的。

问题2：蒸馏数据暴露

PLDK流程：

synthetic images
+
teacher logits

直接传输。

问题：

distilled data is plaintext

攻击者可以：

恢复训练数据特征

反推模型

问题3：多教师冲突

PLDK默认：

single teacher

但现实医疗场景：

Hospital A
Hospital B
Hospital C

教师知识：

可能冲突

PLDK没有处理：

multi-teacher knowledge aggregation
所以你的宏观研究意义可以这样写

（这是论文级表达）

Research Problem

Existing distillation-based collaborative learning methods
such as PLDK enable knowledge sharing without exposing raw data.
However, the intermediate distillation artifacts,
including synthetic data and teacher logits,
remain vulnerable to privacy leakage.

然后提出你的解决方案：

To address these issues,
we propose a secure multi-teacher knowledge distillation framework
based on MPC.
二、你提出的四个“研究意义”是否成立？

你提出了4点：

1️⃣ 蒸馏数据暴露问题
2️⃣ logits泄露问题
3️⃣ 多教师知识冲突
4️⃣ 异构同类数据问题

我逐条给你判断。

1 蒸馏数据暴露问题（成立）

你的系统：

float → RingTensor → secret share

蒸馏数据：

encrypted transmission

因此可以说：

解决了蒸馏数据明文传输问题

论文可以写：

We protect distilled data using MPC-based secret sharing,
preventing direct exposure of synthetic samples during transmission.

这是 完全成立的贡献。

2 logits泄露问题（部分成立，需要谨慎）

你的系统：

logits → secret sharing

保护的是：

logits transmission

但是：

注意一个关键点：

你后面又：

ring_to_float()

恢复明文。

所以严格来说：

你保护的是：

logits in transmission

而不是：

logits in computation

因此论文应该写：

❌ 不要说

eliminate logits leakage

建议说：

mitigate logits leakage risk

或者

protect logits during transmission

这样是合理的。

3 多教师知识冲突问题（成立）

你做了：

Knowledge Aggregation (KA)

实验：

95 + 95 → 67.61
95 + 80 → 64.36

你证明了：

teacher quality impacts final student

这个就是：

multi-teacher distillation analysis

完全可以作为一个贡献。

4 异构数据问题（成立）

你实验：

CIFAR10 + STL10

结果：

49.47

说明：

domain shift hurts distillation

这是：

cross-domain distillation analysis

可以写成：

We further evaluate the framework under heterogeneous datasets
to analyze the impact of domain shift.
三、是否可以结合 MIA 攻击来讲故事？

答案：

可以，但要注意方式。

你可以写：

Previous studies show that model outputs such as logits
may leak membership information.

然后说：

our framework protects logits through MPC.

但不要写：

we defend MIA

因为你 没有做MIA实验。

如果你真的想加强论文，可以加一个：

MIA Attack实验

比如：

Attack accuracy
baseline PLDK: 0.71
Ours: 0.55

那论文直接升档次。

四、你的论文贡献建议写成

论文通常写 3~4条贡献

建议这样写：

Contribution 1
We propose a secure collaborative knowledge distillation framework
that integrates PLDK with MPC.
Contribution 2
The framework protects both distilled data and teacher logits
through secret sharing during transmission.
Contribution 3
We introduce a knowledge aggregation mechanism
to resolve conflicts in multi-teacher scenarios.
Contribution 4
Extensive experiments demonstrate the effectiveness
under both homogeneous and heterogeneous datasets.
五、第二个问题：是否做蒸馏数据可视化？

你的想法：

蒸馏数据
vs
MPC加密数据

我必须很诚实地告诉你：

直接可视化意义不大。

原因是：

MPC secret share 的数据：

looks like random noise

例如：

image → secret share

可视化：

random pixels

这是 理论必然结果。

六、如果要做可视化，正确做法是

做 三张图

图1 **原始蒸馏数据**
synthetic CIFAR10 images
图2 **secret share**
random noise

说明：

cannot reveal visual information
图3 **重构图像**
decode(secret shares)

恢复原图。

论文可以说明：

secret sharing masks visual information.


八、我给你一个真实评价

你的工作其实可以定位为：

Secure Multi-Teacher Distillation Framework

创新点：

PLDK
+
Secure MPC transmission
+
Multi-teacher KA
+
Cross-domain analysis