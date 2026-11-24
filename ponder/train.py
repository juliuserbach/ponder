"""Train an ordering policy for a frozen generator with GRPO.

    torchrun --nproc-per-node=8 -m ponder.train configs/mar.yaml --output runs/mar
    python -m ponder.train configs/sudoku.yaml --output runs/sudoku sudoku.generator=/path/to.pt

Trailing ``key=value`` arguments override config entries (dotted keys).
Rerunning the same command resumes from ``<output>/checkpoint.pt``.

A task (``ponder.generators.<name>.Task``) supplies the generator adapter
and the training distribution:

    Task(config: dict, device, seed)
    task.adapter, task.positions, task.token_width, task.noise_capacity
    task.batch(update, group_ids) -> (schedule, cond, canvas, writable)
        One schedule for the whole update. ``task.noise_capacity`` is read
        after this call, so a task may change it per update.
    task.reward(rollout, cond, group_ids) -> rewards [B]
    task.state_dict() / task.load_state_dict(state)    # e.g. FIDgrad moments
"""

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from ponder import grpo
from ponder.noise import NoiseBank
from ponder.policy import Policy
from ponder.rng import group_ids_for_update
from ponder.sampler import concatenate, sample


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = config
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(value)
    return config


def setup_distributed():
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, torch.device("cuda")
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    return dist.get_rank(), dist.get_world_size(), device


def global_mean(value: torch.Tensor) -> float:
    total = torch.stack([value.sum(), torch.tensor(float(value.numel()), device=value.device)])
    if dist.is_initialized():
        dist.all_reduce(total)
    return (total[0] / total[1]).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    rank, world_size, device = setup_distributed()
    seed = config["seed"]
    group_size = config["group_size"]
    microbatch = config["microbatch"]  # trajectories per rollout and replay chunk, per GPU
    if microbatch % group_size:
        raise ValueError("microbatch must hold whole groups")

    task = importlib.import_module(f"ponder.generators.{config['task']}").Task(
        config[config["task"]], device, seed)
    torch.manual_seed(seed)
    policy = Policy(task.positions, **config["policy"]).to(device).train()
    if world_size > 1:
        policy = DistributedDataParallel(policy, device_ids=[device.index], broadcast_buffers=False)
    module = policy.module if world_size > 1 else policy
    adam = config["adam"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"]["initial"],
                                 betas=tuple(adam["betas"]), eps=adam["eps"], weight_decay=0.0)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    start = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        module.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        task.load_state_dict(state["task"])
        start = state["update"]
    elif rank == 0:
        (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    for update in range(start, config["updates"]):
        group_ids = group_ids_for_update(update, config["groups_per_update"], group_size,
                                         rank, world_size, device)
        schedule, cond, canvas, writable = task.batch(update, group_ids)
        rollouts = []
        for first in range(0, len(group_ids), microbatch):
            rows = slice(first, first + microbatch)
            noise = NoiseBank(seed, group_ids[rows], group_size, task.positions,
                              task.token_width, task.noise_capacity)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rollouts.append(sample(task.adapter, module, schedule, noise, cond[rows],
                                       canvas[rows], writable[rows], record=True))
        rollout = concatenate(rollouts)
        rewards = task.reward(rollout, cond, group_ids)
        advantages = grpo.group_advantages(rewards, group_size, config["advantage_threshold"])
        rate = config["learning_rate"]
        entropy = config["entropy"]
        loss, norm = grpo.update(
            policy, optimizer, rollout, writable, schedule, advantages,
            learning_rate=grpo.linear_schedule(update, rate["initial"], rate["final"],
                                               rate["hold"], rate["decay"]),
            entropy_bonus=grpo.entropy_coefficient(update, entropy["initial"], entropy["final"],
                                                   entropy["decay"]),
            microbatch=microbatch, clip=config["gradient_clip"])

        metrics = {"update": update, "reward": global_mean(rewards), "loss": loss.item(),
                   "grad_norm": norm.item(), "removals": schedule.removals,
                   "lr": optimizer.param_groups[0]["lr"]}
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            with open(args.output / "metrics.jsonl", "a") as log:
                log.write(json.dumps(metrics) + "\n")
            done = update + 1
            if done % config["save_every"] == 0 or done == config["updates"]:
                policy_config = {"positions": task.positions, **config["policy"]}
                torch.save({"policy": module.state_dict(), "policy_config": policy_config},
                           args.output / f"policy_{done:06d}.pt")
                torch.save({"policy": module.state_dict(), "optimizer": optimizer.state_dict(),
                            "task": task.state_dict(), "update": done},
                           args.output / "checkpoint.tmp")
                os.replace(args.output / "checkpoint.tmp", checkpoint_path)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
