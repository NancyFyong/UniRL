# UniRL Rollout-side TP/EP/PP 实施方案

> 来源：3-agent 并行设计 + 1-agent 综合（workflow `99f1ac36`），已解决 G3 gap（driver-side 过滤调用点）。

## 目标

让 1 个 SGLang rollout engine 跨多 GPU（colocate + sleep/wake，对齐 verl/slime），支持 `tp_size>1`。EP/PP 字段预留，PP 权重同步暂 fail-closed。

## 核心架构（已验证）

- verl/slime 都是 colocate + sleep/wake + rollout TP>1（**非 separate slabs**）
- 1 SGLang engine = 1 Ray actor，通过 `base_gpu_id + tp_size=N` 让 SGLang 子进程占 N 个 GPU
- 只有 tp_rank==0 建 HTTP adapter，其他 tp_rank 只参与 FSDP train
- SGLang 原生支持 TP-aware 权重同步：`init_weights_update_group(rank_offset=R)` 内部 `global_rank = R + self.tp_rank`
- UniRL 已有完整 sleep/wake（`engine.py:172-219`），**不是 gap**

## 三个组件 + 综合方案

### 组件 1：Handle / RankInfo / `_build_rank_infos`（Rank Layout）

**文件**：`unirl/distributed/group/handle.py`

**关键改动**：

1. 替换 `_sp_size_from_init_kwargs` → `_parallel_shape_from_init_kwargs`，返回 `(dp, sp, tp, pp, ep)`。校验：`sp>1` 与 `tp/pp>1` 互斥（不同 handle）；`world_size % (tp*pp) == 0`。

2. 替换 `_build_rank_infos`，行优先 (dp, pp, tp) 布局：
```python
def _build_rank_infos(world_size, sp_size=1, tp_size=1, pp_size=1, ep_size=1):
    inner = tp_size * pp_size
    dp_size = world_size // inner
    return [
        RankInfo(
            rank=i, world_size=world_size,
            dp_rank=(i // tp_size) // pp_size, dp_size=dp_size,
            tp_rank=i % tp_size, tp_size=tp_size,
            pp_rank=(i // tp_size) % pp_size, pp_size=pp_size,
            sp_rank=((i // tp_size) // pp_size) % sp_size, sp_size=sp_size,
            ep_rank=0, ep_size=ep_size,  # SGLang 内部管理
        )
        for i in range(world_size)
    ]
```
- `engine_idx = dp_rank * pp_size + pp_rank`（一个 SGLang engine per (dp,pp)）
- NCCL `rank_offset = engine_idx * tp_size + 1`
- `tp_size=pp_size=ep_size=1` 时与现有布局**完全一致**

3. `Handle.__init__` 消费 recipe-level `tp_size`/`pp_size`/`ep_size` hint，构建 `rank_infos` 后**重新注入** per-worker `init_kwargs`：`tp_rank`/`tp_size`/`tp_device_ids`/`pp_size`/`ep_size`。`tp_device_ids = self.device_ids[engine_idx*tp_size : (engine_idx+1)*tp_size]`。

4. 新增 properties：`tp_size`/`pp_size`/`ep_size`/`num_engines`。

**不变**：`remote.py`（字段已存在）、`dispatch.py`（`_collect_dp_merge` 已过滤 `tp_rank==0`）、`placement.py`、`worker.py`。

---

### 组件 2：SGLangRolloutEngine TP rank 分化

**文件**：`unirl/rollout/engine/sglang/engine.py`、`config.py`

**关键改动**：

1. `SGLangEngineConfig` 加字段：`pp_size`/`ep_size`/`dp_size`/`enable_expert_parallel`（`tp_size` 已有）。**不加** `base_gpu_id`（runtime-only）。

2. `server_intent` 加 `runtime_overrides` 参数（最高优先级，仅次于 ports）：
```python
def server_intent(self, *, ports, extra=None, runtime_overrides=None):
    # ... 现有逻辑 ...
    if runtime_overrides:
        intent.update(runtime_overrides)
    intent.setdefault("pp_size", 1)
    intent.setdefault("ep_size", 1)
    # ...
```

3. `SGLangRolloutEngine.__init__` 扩展签名加 `tp_rank=0, tp_size=1, tp_device_ids=None, pp_size=1, ep_size=1`：

```python
self._tp_rank = int(tp_rank)
self._tp_size = int(tp_size)
self._is_tp_zero = (self._tp_rank == 0)

if not self._is_tp_zero:
    # tp_rank>0: no-op shell，不 boot SGLang
    self.adapter = None
    self._backend = None
    self._weight_sync = None
    return

# tp_rank==0 + tp_size>1: override CUDA_VISIBLE_DEVICES
if self._tp_size > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self._tp_device_ids)
    runtime_overrides = {"base_gpu_id": 0, "gpu_id_step": 1}
# ... 正常 boot（现有逻辑）...
```

4. 所有 rollout verb 加 no-op guard（**手动 early-return，非装饰器**，否则 `@distributed` 的 `_distributed_config` 会被隐藏）：
```python
@distributed(dispatch_mode=Dispatch.DP_SCATTER)
def generate(self, req):
    if not self._is_tp_zero:
        return None
    # ... 现有逻辑 ...

@distributed(dispatch_mode=Dispatch.BROADCAST)
def sleep(self, tags=None):
    if not self._is_tp_zero:
        return
    # ... 现有逻辑 ...
```
覆盖：`generate`/`sleep`/`wake_up`/`onload_weights`/`update_weights_from_tensor`/`init_weights_update_group`/`update_weights_from_distributed`/`destroy_weights_update_group`/`set_lora_from_tensors`/`health_check`/`shutdown`/`lora_dirty`。

5. `tp_size=1` → `tp_rank` 恒 0 → `_is_tp_zero=True` → 行为与今天**完全一致**。

---

### 组件 3：NCCLWeightSync + TensorWeightSync rank_offset

**文件**：`unirl/distributed/weight_sync/full/nccl.py`、`tensor.py`

**关键改动**：

1. `NCCLWeightSync.connect` 加 `tp_size` 参数：
```python
def connect(self, *, master_addr, master_port, num_rollout_gpus, tp_size=1):
    tp = int(tp_size)
    num_engines = len(self._rollout_targets)  # 已过滤为 tp_rank==0
    world = num_engines * tp + 1
    refs = [
        handle.call.remote(..., 
            rank_offset=i * tp + 1,  # was i + 1
            world_size=world, ...)
        for i, handle in enumerate(self._rollout_targets)
    ]
```
- `sync()` 循环**不变** —— `dist.broadcast` 已到达所有 NCCL group rank，SGLang `weight_loader` 自行切分
- 加 `if pp_size > 1: raise NotImplementedError`（fail-closed）

2. **Driver-side 过滤**（G3 已解决）—— 两个调用点：
   - `unirl/trainer/async_ar.py:218-223`（AsyncARTrainer）
   - `unirl/trainer/diffusion.py:272-273`（DiffusionTrainer，NCCL 分支）
   
   改为：
```python
tp_size = self.rollout.rank_infos[0].tp_size
if tp_size > 1:
    tp_rank0_workers = [
        w for w, ri in zip(self.rollout.workers, self.rollout.rank_infos)
        if ri.tp_rank == 0
    ]
else:
    tp_rank0_workers = self.rollout.workers
self.weight_sync.set_rollout_targets(tp_rank0_workers, self.rollout.role_name)
self.weight_sync.connect(
    master_addr=addr, master_port=port,
    num_rollout_gpus=len(tp_rank0_workers) * tp_size,
    tp_size=tp_size,
)
```

3. `TensorWeightSync.sync`（colocate）—— 复制 payload `tp_size` 次（零拷贝，字符串引用）：
```python
ri = self.rank_info
tp_size = ri.tp_size if ri is not None else 1
is_tp_rank0 = (ri is None) or (ri.tp_rank == 0)

for bucket, is_last in self._iter_buckets():  # generator 驱动 FSDP all-gather
    if not is_tp_rank0:
        continue  # all-gather 已完成，只跳过 push
    # ... 构建 serialized ...
    self._rollout.update_weights_from_tensor(
        serialized_named_tensors=[payload] * tp_size,  # was [payload]
        load_format="flattened_bucket",
        ...
    )
```
- tp_rank>0 必须仍消费 generator（保持 FSDP all-gather 同步），只跳过 push

4. `tp_size=1` → `rank_offset=i+1`，`[payload]*1=[payload]`，与现有**完全一致**。

**不变**：`base.py`（FSDP all-gather 逻辑不变）、`native.py`（SGLang 内部 fan-out）、`weight_sync.py`（forwarder 透传）。

---

## 实施顺序（关键路径）

```
Phase 1 (Handle layout) ──┬──> Phase 2 (Engine fan-out) ──┐
                          │                                ├──> Smoke test (tp=2)
                          └──> Phase 3 (Weight sync)     ──┘
```

1. **[CP]** `handle.py`: `_parallel_shape_from_init_kwargs` + `_build_rank_infos` + `Handle.__init__` 注入 per-worker init_kwargs
2. **[CP]** `config.py`: 加 `pp_size`/`ep_size`/`dp_size`/`enable_expert_parallel` 字段 + `server_intent` 的 `runtime_overrides`
3. **[CP]** `engine.py`: `__init__` TP 分化 + 所有 verb 的 no-op guard
4. **[CP]** `nccl.py`: `connect` 加 `tp_size` + 新 `rank_offset` 公式
5. **[CP]** `async_ar.py` + `diffusion.py`: driver-side 过滤 + 传 `tp_size`
6. **[CP]** `tensor.py`: 复制 payload + tp_rank guard

Phase 2 和 Phase 3 可并行（不同文件），都依赖 Phase 1。

## 风险

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | `CUDA_VISIBLE_DEVICES` override 与 Ray placement 冲突 | **高** | 只在 tp_rank==0 actor 进程内 override，`backend.boot` 前。`shutdown` 恢复原 env。Tier 1 smoke test 验证 |
| R2 | SGLang `flattened_bucket` loader 在 tp_rank>0 拒绝全量 tensor | 中 | Tier 1 验证；失败则 TensorWeightSync 切到 `update_weights_from_distributed`（NCCL 路径） |
| R3 | FSDP all-gather 失同步（tp_rank>0 跳过 sync body） | 中 | guard 必须在 generator 消费之后；单元测试 mock generator 断言被调用 |
| R4 | `@distributed` 装饰器被 wrapper 隐藏 | 低 | 用手动 early-return，非装饰器 wrapper |

## Smoke Test 计划

### Tier 0 — 单元测试（CPU，无 SGLang）
- `_build_rank_infos(world=4, tp=2)` → rank 0,2 tp_rank=0；rank 1,3 tp_rank=1
- `SGLangRolloutEngine(tp_rank=1, tp_size=2)` 不 import sglang 即可构造
- `SGLangPorts.reserve` 在 tp_rank>0 时不被调用
- `server_intent` tp=1 路径 snapshot 不变
- `NCCLWeightSync.connect(tp_size=2, 2 engines)`: `rank_offset=[1,3]`, `world=5`

### Tier 1 — 集成（1 node, 2 GPUs, tp_size=2）**[GATE]**
- rollout rank 0: `CUDA_VISIBLE_DEVICES=="0,1"`, SGLang boots `tp_size=2, base_gpu_id=0`
- rollout rank 1: 无 SGLang, 无 ports, `_backend is None`
- `generate(req)` 只命中 rank 0 backend
- `sleep`/`wake_up` 在 rank 1 是 no-op
- 权重同步: NCCL group 3 ranks（1 train + 2 engine tp），`rank_offset=1`，broadcast 到达两个 tp rank
- **通过标准**: 无 OOM，无 NCCL timeout，输出与 tp=1 baseline 差异 <1%

### Tier 2 — 集成（1 node, 4 GPUs, tp=2, dp=2）
- 2 engines: group 0 on [0,1], group 1 on [2,3]
- `generate(batch=4)` 分 2 shards，`_collect_dp_merge` 合并
- 权重同步: 2 engines, `rank_offset={1,3}`, `world=5`

### Tier 3 — 回归（tp_size=1）
- 所有现有测试不变

## 回滚

Tier 1 失败则 revert Phase 2+3；Phase 1 无害（tp=1 时只填字段不改行为）。在 `_parallel_shape_from_init_kwargs` 加 `tp_size>1` 拒绝作为 feature flag，直到 Tier 1 通过。

## PP/EP 范围

- **EP**: SGLang 内部管理，UniRL 只透传 `ep_size` 到 `server_args`，无需权重同步改动
- **PP**: 字段预留（`RankInfo.pp_rank` 会填充），但 `NCCLWeightSync.connect` 和 `TensorWeightSync.sync` 在 `pp_size>1` 时 `raise NotImplementedError`。未来需要 per-stage rank_offset map
