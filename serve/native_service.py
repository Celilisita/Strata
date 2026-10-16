"""Native context accounting and MiniCPM DVP1 preparation over the shared API."""
import time
import uuid
from serve.frontend import images_of
from serve.server import Service


class PreparedTokens(list):
    pass


class NativeFamilyService(Service):
    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None):
        started = time.perf_counter()
        if req is not None and req.get("strata_prefix") is not None:
            raise ValueError("native family prefix resume is not supported")
        images = images_of(messages)
        if not images:
            ids, thinking, max_new = super().prepare(messages, tools, kwargs, max_new, force, req)
            prepared = PreparedTokens(ids)
            prepared.frontend_facts = {"frontend_ms": 1000 * (time.perf_counter() - started)}
            return prepared, thinking, max_new
        rendered = self.render_prompt(messages, tools, kwargs)
        if force and kwargs.get("enable_thinking", True) is False:
            rendered += force
        render_done = time.perf_counter()
        payload = None
        try:
            if images:
                if self.vision is None:
                    raise ValueError("this native family has no image encoder")
                payload = self.vision.dir / ("prompt-" + uuid.uuid4().hex + ".dvp")
                if self.vision.format == "minicpm_canvas":
                    from serve.minicpm_visual_prompt import build_visual_prompt
                    ids = build_visual_prompt(self.tok, rendered, images, self.vision, payload)
                elif self.vision.format == "qwen":
                    from serve.qwen_visual_prompt import build_visual_prompt
                    cfg = self.engine.cfg
                    architecture = cfg["asset_binding"]["metadata"]["general.architecture"]
                    width = cfg["asset_binding"]["metadata"][architecture + ".embedding_length"]
                    ids = build_visual_prompt(self.tok, rendered, images, self.vision, payload,
                                              width=width, context=self.engine.max_context)
                else:
                    raise ValueError("unsupported native visual format")
            else:
                ids = self.tok.encode(rendered, parse_special=True)
            room = self.engine.max_context - len(ids)
            if not ids or room < 1:
                raise ValueError("native prompt leaves no room to answer")
            if max_new is None or max_new <= 0:
                max_new = room
            if max_new > room:
                raise ValueError("native prompt plus answer exceeds context")
            prepared = PreparedTokens(ids)
            if payload is not None:
                prepared.visual_path = payload
            prepared.frontend_facts = {"render_ms": 1000 * (render_done - started),
                "tokenize_ms": 1000 * (time.perf_counter() - render_done),
                "frontend_ms": 1000 * (time.perf_counter() - started)}
            self.embeddings.path = None
            return prepared, kwargs.get("enable_thinking", True) is not False, max_new
        except BaseException:
            if payload is not None:
                payload.unlink(missing_ok=True)
            raise
