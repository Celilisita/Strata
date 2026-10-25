"""Small dense prompt preparation over the shared HTTP/SSE/request lifecycle."""
import time

from serve.frontend import images_of
from serve.server import Service


class DensePreparedTokens(list):
    """Complete native prompt plus its optional DVI1 payload ownership."""


class DenseFamilyService(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_ids = set(self.engine.frontend.stop_ids)

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None):
        started = time.perf_counter()
        if req is not None:
            self.engine.validate_request(req)
        frontend = self.engine.frontend
        maximum = max_new if max_new is not None and max_new > 0 else None
        limit = self.engine.cfg.get("max_output", 65536)
        if maximum is not None and maximum > limit:
            raise ValueError("small_dense requested output exceeds configured max_output")
        if images_of(messages):
            if self.vision is None:
                raise ValueError("this small_dense instance has no image encoder")
            if force:
                raise ValueError("forced tool-call prefixes with small_dense images are not supported")
            # Share the same FIFO used by decoder requests, preserving the
            # accepted boundary that prevents encoder/decoder GPU overlap.
            with self.fifo:
                prepared = self.vision.prepare(messages, tools, kwargs, maximum)
        else:
            prepared = frontend.prepare(messages, tools=tools, template_kwargs=kwargs,
                                        max_new=1, context=self.engine.max_context)
            if force and kwargs.get("enable_thinking", True) is False:
                prepared["input_ids"] += self.tok.encode(force, parse_special=True)
                prepared["rendered_prompt"] += force
            room = self.engine.max_context - len(prepared["input_ids"])
            if room < 1:
                raise ValueError("complete small_dense prompt leaves no output room")
            if maximum is None:
                maximum = min(limit, room)
            elif maximum > room:
                if not self.fit_max_tokens:
                    raise ValueError("complete small_dense prompt plus requested output exceeds context")
                maximum = room
            prepared.update(prompt_tokens=len(prepared["input_ids"]), max_new_tokens=maximum)
        self.embeddings.path = None
        self.embeddings.dense_cleanup = prepared.get("cleanup")
        ids = DensePreparedTokens(prepared["input_ids"])
        ids.dense_prepared = prepared
        ids.frontend_facts = {"frontend_ms": 1000 * (time.perf_counter() - started)}
        prompt = prepared["rendered_prompt"]
        thinking = prompt.rfind("<think>") > prompt.rfind("</think>")
        return ids, thinking, prepared["max_new_tokens"]

    def drop_embeddings(self):
        cleanup = getattr(self.embeddings, "dense_cleanup", None)
        self.embeddings.dense_cleanup = None
        try:
            if cleanup is not None:
                cleanup()
        finally:
            super().drop_embeddings()
