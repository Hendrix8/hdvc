"""QINCo2: neural decode + GEMM for IP (torch + lib.Qinco required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from distance_eval.base import ADCBackend


class QINCo2Backend(ADCBackend):
    method_name = "QINCo2"

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        db_fvecs_path: str | Path,
        codes_npz: str | Path | None = None,
        *,
        use_gpu: bool = False,
    ):
        from omegaconf import OmegaConf

        from lib.Qinco.qinco.qinco_tasks import QincoEvalTask

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.db_fvecs_path = Path(db_fvecs_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not self.db_fvecs_path.is_file():
            raise FileNotFoundError(self.db_fvecs_path)

        cz = Path(codes_npz) if codes_npz is not None else self._default_codes_npz(self.model_path)
        data = np.load(str(cz))
        self._codes_np = np.asarray(data["codes"])
        if self._codes_np.ndim != 2:
            self._codes_np = self._codes_np.reshape(len(self._codes_np), -1)
        nb_codes = int(self._codes_np.shape[0])

        base_cfg = OmegaConf.load(str(self.config_path))
        overrides = OmegaConf.create(
            {
                "task": "eval",
                "model": str(self.model_path),
                "db": str(self.db_fvecs_path),
                "seed": 0,
                "cpu": not use_gpu,
                "inference": True,
                "ds": {"db": nb_codes},
                "batch": int(getattr(base_cfg, "batch", 1024)),
            }
        )
        eval_cfg = OmegaConf.merge(OmegaConf.create(base_cfg), overrides)

        self._eval_task = QincoEvalTask(eval_cfg)
        self.model = self._eval_task.qinco_model
        self._accelerator = self._eval_task.accelerator
        self.device = self._accelerator.device
        self._torch = __import__("torch")
        self._recon_cache: np.ndarray | None = None

    @staticmethod
    def _default_codes_npz(model_path: Path) -> Path:
        parent = model_path.parent
        cand = parent.parent / "qinco2_codes" / f"{model_path.stem}_codes.npz"
        if cand.is_file():
            return cand
        raise FileNotFoundError(
            f"codes_npz not provided and no file at {cand} (expected QINCo2.py layout)"
        )

    def close(self) -> None:
        try:
            if hasattr(self, "_eval_task") and self._eval_task is not None:
                self._eval_task.accelerator.end_training()
                self._eval_task = None
        except Exception:
            pass

    def _ensure_recon(self, nb: int) -> np.ndarray:
        if self._recon_cache is not None and self._recon_cache.shape[0] >= nb:
            return self._recon_cache[:nb]
        codes_t = self._torch.from_numpy(self._codes_np[:nb].copy()).to(
            self.device, dtype=self._torch.long
        )
        with self._torch.inference_mode():
            recon = self.model.decode(codes_t)
        if self.device.type == "cuda":
            self._torch.cuda.synchronize()
        recon_np = recon.float().cpu().numpy()
        self._recon_cache = recon_np
        return recon_np[:nb]

    def encode(self, db: np.ndarray) -> Any:
        nb = int(db.shape[0])
        return self._codes_np[:nb]

    def db_norms_sq(self, codes: Any) -> np.ndarray:
        nb = int(np.asarray(codes).shape[0])
        recon = self._ensure_recon(nb)
        return np.sum(recon.astype(np.float64) ** 2, axis=1).astype(np.float32)

    def prepare_query(self, q: np.ndarray) -> Any:
        return np.ascontiguousarray(q, dtype=np.float32)

    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        q = np.ascontiguousarray(qstate, dtype=np.float32)
        nb = int(np.asarray(codes).shape[0])
        recon = self._ensure_recon(nb)
        return (q @ recon.T).astype(np.float32)
