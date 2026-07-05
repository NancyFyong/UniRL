import inspect
import logging
import time
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class ARTrainer(BaseTrainer):
    """Autoregressive (VLM / LLM) RL trainer: rollout + train colocated.

    Sibling of :class:`~unirl.trainer.diffusion.DiffusionTrainer` for the
    AR path. Structurally identical except ``_build_req`` carries **no SDE step
    scheduling** — that is diffusion-only (``DiffusionSamplingParams`` owns
    ``scheduler`` / ``sde_indices`` / ``resolve_sde_indices``), and
    ``ARSamplingParams`` has none of it. Keeping the AR trainer separate means
    the AR path never touches diffusion code (no ``hasattr`` guard, no
    ``dataclasses.replace`` of SDE fields).

    Trainside colocate (the qwen_vl recipe): the training pipeline IS the
    sampler, so ``sync_cfg`` is absent and ``weight_sync`` stays ``None``.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = 60,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        rollout_tp_size: Optional[int] = None,
        rollout_num_engines: Optional[int] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # "group" (textbook GRPO, default) or "global" (v1 baseline parity).
        self.adv_normalization_scope = adv_normalization_scope
        # True (default) = standard GRPO: divide the group-relative advantage by the
        # group std. False = mean-center only (reward - group_mean), NO std division —
        # removes the difficulty bias that over-amplifies low-std (hard) prompts.
        self.normalize_adv_by_std = normalize_adv_by_std
        # verl trainer.balance_batch parity: driver-side reorder of the rollout
        # batch so each DP shard receives a similar total-token workload. FSDP
        # collectives sync all ranks every micro, so a step runs at the SLOWEST
        # rank's pace — without balancing, the rank that drew the longest
        # sequences straggles (~+/-11%% rank-total variance at heavy lengths).
        self.balance_shards = bool(balance_shards)  # overrides the BaseTrainer default (False)
        # AIME-style periodic eval — avg@k accuracy on the eval prompt set
        # (run.eval_data_path), logged under eval/*. eval_interval=0 disables it.
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Set below from the `sync` block; None trainside (shares the module).
        self.weight_sync = None
        # When True (anchor TP + LoRA), the SGLang engine stays resident and
        # sleep/wake is skipped — SGLang TP>1's release/resume_memory_occupation
        # blocks the actor event loop, deadlocking subsequent calls.
        self._rollout_persistent = False

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)

            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

            if rollout_tp_size is None:
                # Default colocate-sibling layout: rollout is one per-device
                # sibling in the placement scope (DP-sharded, TP=1).
                rollout_parsed = parse_hydra_cfg(rollout_cfg)
                if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                    self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)  # for direct sampling
                else:
                    self.rollout = remote(**rollout_parsed)  # for vllm / sglang

                if sync_cfg is not None:
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)
            else:
                # Anchor TP layout: weight_sync is a train-slab sibling; create
                # it inside the placement block (remote_hydra needs the scope).
                # Supports NCCLWeightSync (full-weight, cross-slab NCCL) and
                # RemoteLoraWeightSync (LoRA, cross-slab Ray RPC push).
                if sync_cfg is not None:
                    target = str(sync_cfg.get("_target_", ""))
                    if not target.endswith("NCCLWeightSync") and not target.endswith("RemoteLoraWeightSync"):
                        raise ValueError(
                            f"ARTrainer (rollout_tp_size set) requires NCCLWeightSync or "
                            f"RemoteLoraWeightSync (rollout is cross-slab); got sync._target_={target!r}."
                        )
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)

        if rollout_tp_size is not None:
            # Anchor TP layout (cf. unified_model.py:_wire_engine + slime):
            # pin N rollout engines to N workers, each engine spans tp_size GPUs
            # via clear_cuda_visible + base_gpu_id=rank*tp_size. Handle.world_size
            # = N; DP_SCATTER auto-dispatches the batch across engines.
            tp = int(rollout_tp_size)

            # Auto-compute num_engines if not specified: num_devices // tp_size.
            # EP is a sub-division of TP (tp_size % ep_size == 0), does NOT
            # increase GPU count; PP increases it (tp_size * pp_size per engine).
            # For now only TP (and EP as TP sub-division) is supported; PP is
            # a future extension (would multiply tp by pp_size in the formula).
            if rollout_num_engines is None:
                n_engines = self.num_devices // tp
            else:
                n_engines = int(rollout_num_engines)

            # Boundary check: n_engines * tp_size must not exceed num_devices
            # (rollout slab is a subset of or equal to the train slab).
            total_rollout_gpus = n_engines * tp
            if total_rollout_gpus > self.num_devices:
                raise ValueError(
                    f"ARTrainer (anchor TP): n_engines * tp_size = {n_engines} * {tp} "
                    f"= {total_rollout_gpus} > num_devices={self.num_devices}. "
                    f"Adjust rollout_tp_size / rollout_num_engines so they do not "
                    f"exceed num_devices (EP is a TP sub-division, not a multiplier)."
                )

            # device_ids: which Worker process hosts each engine. Engine 0 is
            # anchored on device 1 (NOT 0) to avoid sharing a Worker with
            # trainer rank 0 (device 0) — RemoteLoraWeightSync.push calls the
            # engine via Ray RPC from rank 0, which self-deadlocks if they
            # share an actor (cf. unified_model.py:290 "base+1, NOT base").
            # SGLang still uses GPUs 0..tp_size-1 via base_gpu_id=0 (engine.py
            # forces base_gpu_id=0 when anchor+tp would exceed num_node_gpus).
            device_ids = [max(1, i * tp) for i in range(n_engines)]

            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            role_cls = rollout_parsed.pop("role_cls")
            self.rollout = self.pool.create_remote(
                role_cls,
                device_ids=device_ids,
                init_kwargs=rollout_parsed,
            )
            # When using RemoteLoraWeightSync, keep the engine resident (no
            # sleep/wake) — SGLang TP>1's release/resume_memory_occupation
            # blocks the actor event loop. FSDP cpu_offload compensates for
            # the GPU memory the engine holds during training.
            sync_target = str(sync_cfg.get("_target_", "")) if sync_cfg else ""
            if sync_target.endswith("RemoteLoraWeightSync"):
                self._rollout_persistent = True

            if self.weight_sync is not None:
                sync_target = str(sync_cfg.get("_target_", ""))
                if sync_target.endswith("RemoteLoraWeightSync"):
                    # RemoteLoraWeightSync.set_rollout_targets takes List[(role, [workers])]
                    self.weight_sync.set_rollout_targets([(self.rollout.role_name, self.rollout.workers)])
                else:
                    # NCCLWeightSync.set_rollout_targets takes (actor_handles, role_name)
                    self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
                    if sync_target.endswith("NCCLWeightSync"):
                        addr, port = self.weight_sync.pick_master()[0]
                        self.weight_sync.connect(
                            master_addr=addr,
                            master_port=port,
                            num_rollout_gpus=len(self.rollout.workers),
                            tp_size=tp,
                        )

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data source batch into a typed :class:`RolloutReq`.

        Expands ``inputs`` by ``total_samples_per_prompt(sampling_params)`` so
        each prompt produces an N-sample GRPO group (sibling samples consecutive).
        AR sampling params ride to the engine untouched — there is no SDE step
        schedule to resolve (that is the diffusion trainer's job).
        """
        inputs = inputs.expand(total_samples_per_prompt(self.sampling_params))
        req = RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=self.sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )
        return req

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the single track (0.0 if none), for the log line.
        ``rollout_id`` only keys the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        if not self._rollout_persistent:
            self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        resp = self.rollout.generate(req)
        if not self._rollout_persistent:
            self.rollout.sleep()

        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)

        mean_reward = 0.0
        for track in resp.tracks.values():
            if track.rewards is None:
                continue
            # Hydrate in place so the wandb reward/advantage stats reuse this
            # fetch instead of re-pulling the TensorRef from the worker.
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
            break  # single-track for now; revisit if multi-track lands

        for name, track in list(resp.tracks.items()):
            if track.rewards is not None:
                resp.tracks[name] = track.compute_advantages(
                    normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope
                )

        self._dump_rollout_samples(req, resp, rollout_id)
        self._drop_decoded(req, resp, rollout_id=rollout_id)
        (track,) = resp.tracks.values()
        # verl balance_batch parity: reorder so each DP shard gets a near-equal
        # token load before DP_SCATTER (no-op when already balanced).
        if self.balance_shards:
            track = track.balance_shards(int(self.num_devices))
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            resp,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
        )
        return result, mean_reward

    def evaluate(self, rollout_id: int) -> float:
        """Periodic eval — ``avg@k`` accuracy on the eval prompt set (no training).

        Mirrors :meth:`train_step`'s rollout+reward path but skips
        advantage/backward: pull ``eval_num_prompts`` eval prompts
        (``run.eval_data_path``), expand each to ``eval_samples_per_prompt``
        siblings, generate at ``eval_temperature``, score, and log the mean
        reward (= avg@k accuracy since reward is 0/1) under ``eval/*``. Returns it.
        """
        import dataclasses

        eval_inputs = self.data_source.get_eval_samples(self.eval_num_prompts)
        inputs = eval_inputs.expand(self.eval_samples_per_prompt)
        eval_ar = dataclasses.replace(
            self.sampling_params.get("ar"),
            samples_per_prompt=self.eval_samples_per_prompt,
            temperature=self.eval_temperature,
        )
        eval_sp = {**self.sampling_params, "ar": eval_ar}
        req = RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=eval_sp,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )
        if not self._rollout_persistent:
            self.rollout.wake_up()
        if self.weight_sync is not None:
            self.weight_sync.sync()
        resp = self.rollout.generate(req)
        if not self._rollout_persistent:
            self.rollout.sleep()

        acc = 0.0
        for track in resp.tracks.values():
            if track.segment is not None:
                track = self.reward.score_and_attach(req=req, track=track)
            if track.rewards is not None:
                track.rewards = hydrate(track.rewards)
                acc = float(track.rewards.to(torch.float32).mean().item())
                break  # single-track for now; revisit if multi-track lands
        logger.info(
            "EVAL rollout %d  eval_acc(avg@%d over %d prompts)=%.4f",
            rollout_id + 1,
            self.eval_samples_per_prompt,
            self.eval_num_prompts,
            acc,
        )
        self.wandb_logger.log_eval(rollout_id + 1, {"acc": acc})
        return acc

    def _dump_rollout_samples(self, req, resp, rollout_id: int) -> None:
        """Debug dump of the first N (prompt, output, reward) triples per rollout.

        Off unless ``ROLLOUT_DUMP_DIR`` is set (driver-side env). Writes one
        ``rollout_<id>.jsonl`` per rollout (``ROLLOUT_DUMP_N`` samples, default
        4) so rollout-engine quality can be eyeballed without keeping the full
        decoded batch alive. Must run BEFORE ``_drop_decoded``. Never raises.
        (Ported from the b182a511 LIN-371 lineage — lost in the rebase.)
        """
        import json
        import os

        out_dir = os.environ.get("ROLLOUT_DUMP_DIR", "")
        if not out_dir:
            return
        try:
            n = int(os.environ.get("ROLLOUT_DUMP_N", "4"))
            prompts = getattr(req.primitives.get("text"), "texts", None) or []
            (track,) = resp.tracks.values()
            outputs = getattr(track.decoded, "texts", None) or []
            rewards = track.rewards.to(torch.float32).tolist() if track.rewards is not None else []
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"rollout_{int(rollout_id):04d}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(min(n, len(outputs))):
                    f.write(
                        json.dumps(
                            {
                                "rollout": int(rollout_id),
                                "sample": i,
                                "prompt": prompts[i] if i < len(prompts) else None,
                                "output": outputs[i],
                                "output_chars": len(outputs[i] or ""),
                                "reward": rewards[i] if i < len(rewards) else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception as exc:  # debug path — never let it kill training
            logger.warning("rollout sample dump failed: %s", exc)

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` is the output folder (defaults to
        ``./checkpoints``); ``save_mode="auto"`` writes LoRA-only checkpoints
        when LoRA is active and full checkpoints otherwise.
        ``load_dir``: restore from a checkpoint directory and RESUME from its
        saved step — ``num_rollouts`` is the TOTAL budget.

        Deferred: ``num_updates_per_batch`` multi-epoch replay, eval cadence.
        """
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope},
        )
        try:
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)  # baseline AIME accuracy, logged at eval step 0
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate.
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )
                result, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id=rollout_id)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
