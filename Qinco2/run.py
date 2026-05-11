# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

_qinco2_root = os.path.abspath(os.path.dirname(__file__))
_workspace_root = os.path.abspath(os.path.join(_qinco2_root, '..'))
_lib_qinco_root = os.path.join(_workspace_root, 'lib', 'Qinco')

# Bundled extended fork lives under ``Qinco2/Qinco`` but imports use ``qinco.*`` (lowercase).
_bundled_qinco_cap = os.path.join(_qinco2_root, 'Qinco')
_bundled_qinco_pkg = os.path.join(_qinco2_root, 'qinco')
if os.path.isdir(_bundled_qinco_cap) and not os.path.exists(_bundled_qinco_pkg):
    try:
        os.symlink('Qinco', _bundled_qinco_pkg)
    except OSError:
        pass

# Prefer Qinco2 first so bundled ``qinco`` / ``Qinco`` win over ``lib/Qinco/qinco``.
for _p in (_lib_qinco_root, _workspace_root):
    if _p not in sys.path:
        sys.path.append(_p)
if _qinco2_root not in sys.path:
    sys.path.insert(0, _qinco2_root)

import hydra
from omegaconf import DictConfig

# Imports experiments (necessary to register experiments)
from qinco.qinco_tasks import QincoConvertTask, QincoEvalTask, QincoTrainTask
from qinco.search.search_tasks import (
    BuildIndexTask,
    EncodeDBTask,
    IVFTrainTask,
    SearchTask,
    TrainPairwiseDecoderTask,
    Shen_SearchTask,
    ShenBuildIndicesTask,
    ShenComputeDistancesTask,
    TrainQincoAQTask,
    QincoAQComputeDistancesTask,
    QincoAQFaissComputeDistancesTask,
)

EXPERIMENTS = {
    "train": QincoTrainTask,
    "eval_valset": QincoTrainTask,
    "eval": QincoEvalTask,
    "eval_time": QincoEvalTask,
    "convert": QincoConvertTask,
    "ivf_centroids": IVFTrainTask,
    "encode": EncodeDBTask,
    "build_index": BuildIndexTask,
    "train_pairwise_decoder": TrainPairwiseDecoderTask,
    "shen_build_indices": ShenBuildIndicesTask,
    "shen_compute_distances": ShenComputeDistancesTask,
    "search": SearchTask,
    "shen_search": Shen_SearchTask,
    'train_qinco_aq': TrainQincoAQTask,
    'qinco_aq_compute_distances': QincoAQComputeDistancesTask,
    'qinco_aq_faiss_compute_distances': QincoAQFaissComputeDistancesTask,
}


@hydra.main(version_base=None, config_path="config", config_name="qinco_cfg")
def main(cfg: DictConfig):

    print(cfg)


    if cfg.task is None:
        raise ValueError(
            "Please specify a task (train, eval, etc.) using the 'train=<...>' argument"
        )
    if cfg.task not in EXPERIMENTS:
        raise ValueError(f"Unknown task '{cfg.task}'. Known tasks: {sorted(EXPERIMENTS.keys())}")
    expe = EXPERIMENTS[cfg.task](cfg)

    expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
    expe.run()
    expe.accelerator.print("Task done")
    expe.accelerator.end_training()  # Destroy process group


if __name__ == "__main__":
    main()  # pylint: disable=all
