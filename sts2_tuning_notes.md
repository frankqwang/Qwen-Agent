# STS2 WebUI / LM Studio 调参备忘

这份文档是给当前这台机器用的：

- GPU: RTX 2070 Super
- VRAM: 8 GB
- 使用场景: `examples/sts2_webui.py` + LM Studio 本地 OpenAI 兼容接口

## 先说结论

如果感觉 `Qwen3.5 4B` 很慢，优先按下面顺序调：

1. 降低 `max-input-tokens`
2. 降低 LM Studio 的 `Context Length`
3. 把 `Max Concurrent Predictions` 改成 `1`
4. 需要更快就换 `0.8B`

## 为什么 4B 也会慢

`sts2_webui` 不是单次聊天，而是 Agent 场景：

- 会读取游戏状态
- 会调用工具
- 会把工具结果再喂回模型
- 一轮自动步骤里可能会多次调用模型

所以慢的来源不只是模型大小，还包括：

- 上下文长度
- KV cache 占用
- 工具调用轮数
- 思考输出太长

对 8 GB 显存来说，最容易卡的是上下文和 KV cache，而不是“4B 参数本体”。

## 脚本侧参数

`examples/sts2_webui.py` 当前最关键的参数是：

- `--model`
- `--max-input-tokens`

### `max-input-tokens` 是什么

它表示单次请求里，最多喂给模型多少输入 token。

它影响：

- 显存占用
- KV cache 大小
- 推理速度
- 模型能记住多少历史和工具结果

它不直接等于输出长度。

### 推荐值

对当前机器：

- `16000`: 太大，不推荐
- `8192`: 还是偏大
- `4096`: 推荐，比较平衡
- `2048`: 更快，适合优先追求速度

### 推荐启动命令

4B 平衡方案：

```powershell
python.exe examples\sts2_webui.py --model qwen/qwen3.5-4b --max-input-tokens 4096
```

4B 速度优先方案：

```powershell
python.exe examples\sts2_webui.py --model qwen/qwen3.5-4b --max-input-tokens 2048
```

0.8B 速度优先方案：

```powershell
python.exe examples\sts2_webui.py --model qwen3.5-0.8b --max-input-tokens 4096
```

如果 `0.8B` 在 LM Studio 里的模型名不同，替换成实际名字即可。

## LM Studio 参数建议

以下是当前硬件下更合适的方向。

### 1. Context Length

最重要。

推荐：

- `4096`
- 如果还慢就试 `2048`

不建议长期用：

- `8192`
- 更高

原因：

- Context 越大，KV cache 越大
- 更吃显存
- 更容易用到共享内存
- 速度会明显掉

### 2. GPU Offload

建议：

- 尽量拉满

原因：

- 这台机器是典型“尽量全放 GPU 上跑”更划算

### 3. CPU Thread Pool Size

建议：

- 保持默认附近即可
- 不需要为了提速盲目拉很高

原因：

- 当前主要瓶颈更像显存和上下文，不是 CPU 线程数

### 4. Evaluation Batch Size

建议：

- 先试 `256`
- 如果稳定且不爆显存，再试 `512`

原因：

- 批量太大时显存压力会上升
- 2070S 这种 8 GB 卡更容易因为这个卡边缘

### 5. Max Concurrent Predictions

建议：

- 改成 `1`

不要设成：

- `4`

原因：

- 当前是单用户本地 Agent，不需要并发预测
- 并发数更高会额外占缓存和显存
- 很可能只会更慢，不会更快

### 6. Unified KV Cache

建议：

- 保持开启

除非遇到兼容问题，否则先不动。

### 7. Offload KV Cache to GPU Memory

建议：

- 默认保持开启

如果显存始终压线、速度很不稳定，可以试着关闭做对比。

注意：

- 关闭后可能更稳
- 但通常会更慢

### 8. Keep Model in Memory

建议：

- 开启

这样反复请求时不会频繁重新加载。

### 9. Try mmap()

建议：

- 开启

### 10. Flash Attention

建议：

- 开启

## 怎么判断是不是显存压力过大

如果任务管理器里出现下面这种情况，说明已经非常接近瓶颈：

- 专用显存接近满，比如 `7.5 / 8.0 GB`
- 共享 GPU 内存开始增长
- 总 GPU 内存高于专用显存很多
- 回复明显变慢

这通常不是“直接 OOM 报错”，而是：

- 专用显存不够了
- 一部分数据被挪到共享内存
- 速度因此明显下降

## 本地服务和代理

如果你开了全局代理，本地 WebUI、Gradio、LM Studio 可能会出现很怪的报错。

典型现象：

- 启动时已经打印了 `Running on local URL`
- 但随后报错：
  - `gradio_api/startup-events failed (code 502)`
- `clean_webui.py` 和 `sts2_webui.py` 都会挂
- 改成规则代理或直连后恢复正常

原因通常是：

- Gradio 启动时会请求本机地址
- 比如 `http://127.0.0.1:7860/...`
- 如果全局代理把这些本地回环请求也代理了
- 就可能导致本地服务自己访问自己时失败

建议：

- 不要让 `localhost` / `127.0.0.1` 走全局代理
- 对本地服务优先使用规则代理或直连

推荐加入 bypass / 直连的地址：

- `localhost`
- `127.0.0.1`
- `::1`
- `*.local`

如果遇到下面这种情况，先检查代理模式，不要先怀疑代码：

- Gradio 启动时报 `startup-events 502`
- LM Studio 本地接口明明开着，但程序连不上
- 纯本地服务互相访问异常

## 推荐配置

### 配置 A: 4B 平衡方案

适合想保留一定质量，同时别太慢。

LM Studio:

- Context Length: `4096`
- GPU Offload: 拉满
- Evaluation Batch Size: `256`
- Max Concurrent Predictions: `1`
- Unified KV Cache: 开
- Offload KV Cache to GPU Memory: 开
- Keep Model in Memory: 开
- Try mmap(): 开
- Flash Attention: 开

脚本:

```powershell
python.exe examples\sts2_webui.py --model qwen/qwen3.5-4b --max-input-tokens 4096
```

### 配置 B: 4B 低压方案

适合显存紧张、共享内存已经开始上涨时。

LM Studio:

- Context Length: `2048`
- GPU Offload: 拉满
- Evaluation Batch Size: `256`
- Max Concurrent Predictions: `1`

脚本:

```powershell
python.exe examples\sts2_webui.py --model qwen/qwen3.5-4b --max-input-tokens 2048
```

### 配置 C: 0.8B 极速方案

适合先追求流畅度。

LM Studio:

- Context Length: `2048` 或 `4096`
- GPU Offload: 拉满
- Evaluation Batch Size: `256`
- Max Concurrent Predictions: `1`

脚本:

```powershell
python.exe examples\sts2_webui.py --model qwen3.5-0.8b --max-input-tokens 4096
```

如果质量不够，再回到 4B。

## 额外建议

### UI 里的每轮动作数

建议先设小一点：

- `1`
- 或 `2`

原因：

- 每轮动作数越大
- Agent 一轮里可能调用更多次模型
- 看起来就会更慢

### 思考输出

如果模型输出很长的思考内容，也会变慢。

可选做法：

- 关闭“展开思考”
- system prompt 里要求少解释、优先行动

## 建议的排查顺序

如果觉得慢，就按这个顺序试：

1. LM Studio `Context Length` 改到 `4096`
2. 脚本 `--max-input-tokens` 改到 `4096`
3. `Max Concurrent Predictions` 改到 `1`
4. `Evaluation Batch Size` 改到 `256`
5. 观察任务管理器里共享 GPU 内存是否下降
6. 如果还慢，改成 `2048`
7. 如果还想更快，换 `0.8B`

## 常见报错速查

### 1. `n_keep >= n_ctx`

例子：

```text
The number of tokens to keep from the initial prompt is greater than the context length
```

含义：

- 实际输入比模型上下文窗口还大

处理：

- 提高 LM Studio 的 `Context Length`
- 或降低脚本侧 `--max-input-tokens`

### 2. `Error code: 502`

如果出现在模型调用阶段：

- 更像 LM Studio / 推理后端失败

优先检查：

- 模型是否需要重新加载
- 显存是否压线
- `Max Concurrent Predictions` 是否太高
- `Evaluation Batch Size` 是否太大

如果出现在 Gradio 启动阶段，且路径像这样：

```text
/gradio_api/startup-events
```

优先检查：

- 是否开启了全局代理
- `localhost` / `127.0.0.1` 是否被代理了

## 一句话版本

对这台 `2070 Super 8GB`：

- 4B 能跑，但上下文别开太大
- `4096` 通常比 `8192/16000` 合适得多
- `Max Concurrent Predictions` 应该设成 `1`
- 想明显提速，直接试 `0.8B`
