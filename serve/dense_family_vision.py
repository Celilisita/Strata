"""Lifecycle bridge for the accepted small-dense SDV1 encoder in the shared server."""
from pathlib import Path


class DenseFamilyVision:
    format = "dense_sdv1"

    def __init__(self, engine):
        self.engine = engine
        self.dir = Path(engine.cfg["temp_dir"]) / "dense-vision"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prepare_visual = None

    def alive(self):
        return self.prepare_visual is not None and self.prepare_visual.encoder.alive()

    def restart(self):
        from serve.dense_vision import ResidentVision, VisualPrepare
        self.unload()
        cfg = self.engine.cfg
        vision = cfg["vision"]
        encoder = ResidentVision(vision["exe"], cfg["model"], vision["mmproj"],
            self.dir, self.dir / "logs", self.engine.environment, self.engine.max_context,
            gpu=vision.get("gpu", True), threads=vision.get("threads"),
            max_tokens=vision.get("max_tokens"), startup_timeout=cfg.get("startup_timeout", 1800),
            request_timeout=cfg.get("request_timeout", 7200))
        try:
            self.prepare_visual = VisualPrepare(encoder, cfg.get("max_output", 65536),
                                                vision.get("keep_artifacts", False))
        except BaseException:
            encoder.close()
            raise

    def prepare(self, messages, tools, kwargs, maximum, cancel=None):
        if not self.alive():
            self.restart()
        return self.prepare_visual(self.engine.frontend, messages, tools, kwargs,
                                   maximum, self.engine.max_context, cancel=cancel)

    def unload(self):
        if self.prepare_visual is not None:
            self.prepare_visual.close()
            self.prepare_visual = None

    def shutdown(self):
        self.unload()

    def close(self):
        self.unload()
