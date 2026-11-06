
核心目标只有一个：

$$
\boxed{
\textbf{验证 PEFT-BoA 的 Patch/Class Adapter，能否补足 UAD 在 NIR/TIR，尤其 TIR/Aerial 下的身份特征形成不足。}
}
$$

因此修改原则是：**UAD 主干、损失、采样、训练协议、推理协议全部锁死，只改变 ViT 内部的特征适配路径。**

---

## 一、整体代码结构原则

不要重构整个UAD，也不要把PEFT-BoA整套工程搬进来。

只在现有 UAD ViT 中新增两个可独立开关的模块：

* `DPA`：Patch Adapter
* `TCA`：Class Adapter

形成三组正式实验：

$$
\boxed{
P1=UAD+DPA
}
$$

$$
\boxed{
P2=UAD+TCA
}
$$

$$
\boxed{
P3=UAD+DPA+TCA
}
$$

原UAD继续作为：

$$
P0
$$

第一阶段**不加入IFA**，避免把“特征形成”和“跨模态融合”混在一起。

---

# 二、DPA修改方向

DPA是这轮最核心的改动。

它应该插在**每个 ViT Transformer block 的 FFN 并行路径**上，只处理：

$$
\boxed{\text{Patch Tokens}}
$$

不直接处理CLS。

原block：

$$
X\rightarrow MHSA\rightarrow FFN\rightarrow X'
$$

改成：

$$
X^{mhsa}=[CLS,P]
$$

然后Patch部分：

$$
P'
=
P
+
FFN(P)
+
DPA(P)
$$

CLS仍走原UAD路径。

DPA保持PEFT原论文结构：

$$
\boxed{
DownProjection
\rightarrow
ReLU
\rightarrow
UpProjection
}
$$

不增加：

* modality gate；
* platform gate；
  -额外attention；
  -新loss；
  -新的part token。

第一轮保持**RGB/NIR/TIR共享同一套DPA参数逻辑**，不要自行改成三个模态三个adapter。

目的不是“显式分模态”，而是给共享ViT增加一个专门针对patch的可学习补偿通路。

---

# 三、TCA修改方向

TCA只作用于：

$$
\boxed{\text{CLS Token}}
$$

保持PEFT原论文两部分：

1. CLS在attention前后的learnable modulation vector；
2. 一套跨Transformer层共享的CLS adapter。

不要让TCA处理patch，也不要把TCA改成modality-specific。

TCA的作用是作为DPA的对照：

$$
\text{Patch adaptation}
\quad VS\quad
\text{CLS adaptation}
$$

这样结果才能直接判断：

$$
\boxed{
\text{当前瓶颈更偏Patch形成还是CLS任务适配}
}
$$

---

# 四、UAD原有内容全部保留

以下内容一律不动：

* ImageNet ViT-B/16初始化；
* backbone full fine-tuning；
* PKM sampler；
* 120 epochs；
* optimizer；
* learning rate；
* CE；
* WRT；
* ProCA；
* GPD；
* global BNNeck；
* WHU测试协议；
* 单模态query/gallery；
* 推理descriptor。

损失继续：

$$
\boxed{
L=
L_{CE}
+
L_{WRT}
+
0.01L_{ProCA}
+
L_{GPD}
}
$$

不要引入PEFT论文自己的训练范式。

这样实验变量才唯一：

$$
\boxed{\text{ViT内部adapter}}
$$

---

# 五、不要冻结UAD backbone

PEFT-BoA原论文冻结CLIP，是因为它研究的是：

$$
\text{Parameter-Efficient Fine-Tuning}
$$

但我们的目标不是复现PEFT训练范式，而是验证adapter机制能否补偿UAD的弱模态feature formation。

因此：

$$
\boxed{
ViT继续full fine-tune
}
$$

同时：

$$
DPA/TCA\text{一起训练}
$$

否则就会把：

$$
\text{UAD+Adapter}
$$

变成：

$$
\text{Frozen ViT+Adapter}
$$

失去和当前baseline的公平性。

---

# 六、参数初始化原则

Adapter新增参数采用PEFT原论文的轻量初始化原则即可：

* learnable modulation vectors零初始化；
* adapter projection采用小幅随机初始化；
* 最好保证初始状态下新模块对原UAD输出扰动尽可能小。

核心目标是：

$$
\boxed{
Epoch\ 0:
UAD+Adapter\approx UAD
}
$$

让训练真正学习“补偿”，而不是一开始就破坏原表示。

---

# 七、推理阶段保持完全不变

P1/P2/P3测试时：

$$
\boxed{\text{输入仍然只需要单张RGB/NIR/TIR图像}}
$$

最终仍然使用：

$$
\boxed{\text{UAD原global pre-BN CLS descriptor}}
$$

不要：

* 拼接RGB/NIR/TIR；
* 拼接patch；
* 增加adapter输出；
  -增加rerank；
  -改变距离计算。

否则无法判断性能变化来自哪里。

---

# 八、配置层面只需要几个开关

抽象上保持：

```text
USE_DPA
USE_TCA
```

三组配置：

```text
P1: DPA=True,  TCA=False
P2: DPA=False, TCA=True
P3: DPA=True,  TCA=True
```

其它所有配置必须完全继承P0 UAD。

不要因为加adapter顺便调：

* LR；
* loss weight；
* sampler；
* augmentation；
* epoch。

第一轮只测结构。

---

# 九、实验输出必须额外记录两类结果

除了最终：

$$
mAP/R1/R5/R10
$$

还应继续输出：

### 模态结果

$$
RGB/NIR/TIR
$$

九种Q→G组合。

重点看：

$$
TIR\rightarrow RGB
$$

$$
TIR\rightarrow NIR
$$

$$
TIR\rightarrow TIR
$$

以及NIR相关组合。

### 平台结果

重点看：

$$
A\rightarrow G
$$

尤其：

$$
TIR/Aerial
$$

因为这才是当前最弱场景。

---

# 十、P1必须保留layer-wise诊断接口

只需要P0和P1重点做：

$$
L3,L6,L9,L12
$$

的Patch/CLS特征导出。

目的是直接比较：

$$
UAD
\quad VS\quad
UAD+DPA
$$

是否出现：

$$
Patch^{DPA}_{NIR/TIR}
>
Patch^{UAD}_{NIR/TIR}
$$

如果最终指标涨，但中层Patch身份信息没变化，那么DPA并没有真正解决我们定义的问题。

---

# 十一、第一阶段禁止增加的东西

明确不要做：

* DPA+Part；
* DPA+SH/SP；
* modality-specific DPA；
* platform-specific DPA；
* DPA gate；
* DPA+frequency；
* DPA+新的对齐loss；
* DPA+PGPD；
* TCA重新设计；
* IFA提前加入。

先把原生机制跑清楚。

---

## 最终修改方向

整个代码改动可以概括成：

$$
\boxed{
\textbf{在UAD ViT内部新增Patch-only DPA和CLS-only TCA两条原生adapter路径，
保持UAD其它训练与推理逻辑完全不变，
通过P1/P2/P3验证弱模态瓶颈究竟能否通过内部特征适配得到缓解。}
}
$$

这是这轮最应该坚持的边界：**只改特征形成路径，不改问题定义，不改监督，不改推理。**

注意最后给出在Linux服务器的运行命令