"""V100Qwen metadata-bound prompts over the shared HTTP/SSE service."""
import time
from serve.frontend import images_of
from serve.server import Service


class V100QwenPreparedTokens(list):
    pass


class V100QwenService(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_ids = set(self.engine.frontend.stop_ids)

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None):
        started = time.perf_counter()
        self.engine.validate_request(req or {})
        if images_of(messages):
            raise ValueError("this completed V100 Qwen provider accepts text requests")
        maximum = max_new if max_new is not None and max_new > 0 else None
        limit = self.engine.cfg.get("max_output", 65536)
        if maximum is not None and maximum > limit:
            raise ValueError("V100 Qwen requested output exceeds configured max_output")
        prepared = self.engine.frontend.prepare(messages, tools=tools, template_kwargs=kwargs,
            max_new=1, context=self.engine.max_context)
        if force and kwargs.get("enable_thinking", True) is False:
            prepared["input_ids"] += self.tok.encode(force, parse_special=True)
            prepared["rendered_prompt"] += force
        room = self.engine.max_context - len(prepared["input_ids"])
        if room < 1:
            raise ValueError("complete V100 Qwen prompt leaves no output room")
        if maximum is None:
            maximum = min(limit, room)
        elif maximum > room:
            if not self.fit_max_tokens:
                raise ValueError("complete V100 Qwen prompt plus requested output exceeds context")
            maximum = room
        self.embeddings.path = None
        ids = V100QwenPreparedTokens(prepared["input_ids"])
        ids.frontend_facts = {"frontend_ms": 1000*(time.perf_counter()-started)}
        prompt = prepared["rendered_prompt"]
        return ids, prompt.rfind("<think>") > prompt.rfind("</think>"), maximum
