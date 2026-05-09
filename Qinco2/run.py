# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append('/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2')

import hydra
from omegaconf import DictConfig

# Imports experiments (necessary to register experiments)
from Qinco.qinco_tasks import QincoConvertTask, QincoEvalTask, QincoTrainTask
from Qinco.search.search_tasks import (
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
}


@hydra.main(version_base=None, config_path="config", config_name="qinco_cfg")
def main(cfg: DictConfig):

    print(cfg)


    if cfg.task is None:
        raise ValueError(
            "Please specify a task (train, eval, etc.) using the 'train=<...>' argument"
        )
    expe = EXPERIMENTS[cfg.task](cfg)

    expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
    expe.run()
    expe.accelerator.print("Task done")
    expe.accelerator.end_training()  # Destroy process group


if __name__ == "__main__":
    main()  # pylint: disable=all
