# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from typing import Any, Mapping, Optional

import dinov2.distributed as distributed


@dataclass(slots=True)
class WandbSettings:
    project: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[list[str]] = None
    mode: str = "online"

    @property
    def enabled(self) -> bool:
        return self.project is not None and distributed.is_main_process()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WandbSettings":
        return cls(
            project=getattr(args, "wandb_project", None),
            entity=getattr(args, "wandb_entity", None),
            run_name=getattr(args, "wandb_run_name", None),
            group=getattr(args, "wandb_group", None),
            tags=getattr(args, "wandb_tags", None),
            mode=getattr(args, "wandb_mode", "online"),
        )

    @classmethod
    def add_args(
        cls,
        parser: argparse.ArgumentParser,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        wandb_mode: str = "online",
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            "--wandb-project",
            type=str,
            default=None,
            help="Weights & Biases project name. If unset, wandb logging is disabled.",
        )
        parser.add_argument(
            "--wandb-entity",
            type=str,
            default=None,
            help="Weights & Biases entity or team name.",
        )
        parser.add_argument(
            "--wandb-run-name",
            type=str,
            default=None,
            help="Weights & Biases run name.",
        )
        parser.add_argument(
            "--wandb-group",
            type=str,
            default=None,
            help="Weights & Biases run group.",
        )
        parser.add_argument(
            "--wandb-tags",
            nargs="*",
            default=None,
            help="Optional Weights & Biases tags.",
        )
        parser.add_argument(
            "--wandb-mode",
            type=str,
            default="online",
            choices=["online", "offline", "disabled", "dryrun"],
            help="Weights & Biases mode.",
        )
        parser.set_defaults(
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
            wandb_mode=wandb_mode,
        )
        return parser


def get_wandb_module():
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def is_wandb_enabled(project: Optional[str]) -> bool:
    return project is not None and distributed.is_main_process()


def build_wandb_config(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def add_wandb_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return WandbSettings.add_args(parser)


def init_wandb(
    *,
    settings: WandbSettings,
    config: Optional[Mapping[str, Any]] = None,
    reinit: bool = True,
):
    if not settings.enabled:
        return None

    wandb = get_wandb_module()
    if wandb is None:
        raise ImportError("wandb is not installed, but a wandb project was provided")

    return wandb.init(
        project=settings.project,
        entity=settings.entity,
        name=settings.run_name,
        group=settings.group,
        tags=settings.tags,
        mode=settings.mode,
        config=dict(config or {}),
        reinit=reinit,
    )


def log_wandb(run, data: Mapping[str, Any], *, step: Optional[int] = None) -> None:
    if run is not None:
        run.log(dict(data), step=step)


def update_wandb_summary(run, data: Mapping[str, Any]) -> None:
    if run is not None:
        run.summary.update(dict(data))


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()