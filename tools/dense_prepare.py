"""Prepare native dense-runtime requests from a GGUF's tokenizer and chat template.

This reads the GGUF header only. It never loads weights or runs inference. The
optional suite describes independent requests; its input lengths are actual
complete rendered prompt lengths, and its output limits never suppress EOS.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from gguf_reader import GGUFFile
from dense_tokenizer import PRE_PATTERNS, Tokenizer


# EOG spellings are taken from the vocabulary itself, never assigned fixed IDs.
# These include the EOS/turn markers used by MiniCPM5 and Qwen3.5. FIM markers
# that end a generated file are retained when present, as in llama-vocab.cpp.
EOG_TEXT = frozenset((
    "</s>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>",
    "<|end_of_text|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>",
    "<|calls|>", "<end_of_turn>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
    "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
    "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>",
    "<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
    "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
    "<|file_sep|>", "<|plamo:file_separator|>",
))


def _json_filter(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise jinja2.TemplateError(message)


class DenseFrontend:
    def __init__(self, path, template_name: str | None = None):
        self.path = pathlib.Path(path)
        gguf = GGUFFile(self.path)
        md = self.md = gguf.metadata
        pre = md.get("tokenizer.ggml.pre")
        if pre not in PRE_PATTERNS:
            raise ValueError(f"unsupported tokenizer.ggml.pre {pre!r}")
        if md.get("tokenizer.ggml.model") != "gpt2":
            raise ValueError("dense frontend requires the GGUF's byte-level BPE tokenizer")
        self.special_ids = {k: int(v) for k, v in md.items()
                            if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        self.tokenizer = Tokenizer(list(md["tokenizer.ggml.tokens"]),
                                   list(md["tokenizer.ggml.merges"]),
                                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                                   pre, self.special_ids)
        for key, token_id in self.special_ids.items():
            if not 0 <= token_id < len(self.tokenizer.tokens):
                raise ValueError(f"{key}={token_id} is outside the vocabulary")
        key = "tokenizer.chat_template" + ("." + template_name if template_name else "")
        source = md.get(key)
        if not isinstance(source, str) or not source.strip():
            available = sorted(k for k in md if k.startswith("tokenizer.chat_template"))
            raise ValueError(f"GGUF has no nonempty {key}; available={available}")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _json_filter
        env.globals.update(raise_exception=_raise,
                           strftime_now=lambda fmt: datetime.datetime.now().strftime(fmt))
        self.template = env.from_string(source)
        self.template_key = key
        self.template_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.template_chars = len(source)
        self.template_tokens = {k[len("tokenizer.ggml."):-3]: self.tokenizer.tokens[v]
                                for k, v in self.special_ids.items()}
        stop_ids = {v for k, v in self.special_ids.items()
                    if k.rsplit(".", 1)[-1] in ("eos_token_id", "eot_token_id", "eom_token_id")}
        stop_ids.update(i for token, i in self.tokenizer.ids.items() if token in EOG_TEXT)
        self.stop_ids = sorted(stop_ids)
        if not self.stop_ids:
            raise ValueError("GGUF provides no recognized end-of-generation token")
        self.identity = {
            "architecture": md.get("general.architecture"), "name": md.get("general.name"),
            "gguf_version": gguf.version, "tokenizer_model": "gpt2", "pre": pre,
            "vocab_size": len(self.tokenizer.tokens), "template_key": key,
            "template_sha256": self.template_hash, "template_chars": self.template_chars,
            "special_ids": self.special_ids,
            "add_bos_token": bool(md.get("tokenizer.ggml.add_bos_token", False)),
            "add_eos_token": bool(md.get("tokenizer.ggml.add_eos_token", False)),
            "stop_tokens": {str(i): self.tokenizer.tokens[i] for i in self.stop_ids},
        }

    def prepare(self, messages: list[dict], *, template_kwargs: dict | None = None,
                tools: list[dict] | None = None, add_generation_prompt: bool = True,
                max_new: int = 512, context: int = 16384, name: str | None = None) -> dict:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if any(not isinstance(m, dict) or not isinstance(m.get("role"), str)
               or "content" not in m for m in messages):
            raise ValueError("each message requires role and content")
        if max_new < 1 or context < 1:
            raise ValueError("max_new and context must be positive")
        kwargs = dict(template_kwargs or {})
        reserved = {"messages", "tools", "add_generation_prompt"}.intersection(kwargs)
        if reserved:
            raise ValueError(f"template_kwargs cannot replace {sorted(reserved)}")
        rendered = self.template.render(**dict(self.template_tokens, **kwargs), messages=messages,
                                        tools=tools, add_generation_prompt=add_generation_prompt)
        ids = self.tokenizer.encode(rendered, parse_special=True)
        # Chat templates often contain BOS themselves. Apply a declared GGUF
        # insertion only when it is absent, preserving exactly one boundary.
        for kind, front in (("bos", True), ("eos", False)):
            if not self.identity[f"add_{kind}_token"]:
                continue
            token_id = self.special_ids.get(f"tokenizer.ggml.{kind}_token_id")
            if token_id is None:
                raise ValueError(f"GGUF requests {kind.upper()} insertion without its token ID")
            if not ids or ids[0 if front else -1] != token_id:
                ids = [token_id] + ids if front else ids + [token_id]
        if not ids:
            raise ValueError("the rendered prompt is empty")
        if len(ids) + max_new > context:
            raise ValueError(f"prompt {len(ids)} + output limit {max_new} exceeds context {context}")
        image_count = sum(1 for m in messages if isinstance(m["content"], list)
                          for part in m["content"] if isinstance(part, dict)
                          and part.get("type") in ("image", "image_url", "input_image"))
        result = {
            "model": str(self.path.resolve()), "input_ids": ids, "messages": messages,
            "stop_ids": self.stop_ids, "prompt_tokens": len(ids), "max_new_tokens": max_new,
            "context": context, "rendered_prompt": rendered, "template_kwargs": kwargs,
            "add_generation_prompt": add_generation_prompt, "metadata": dict(self.identity,
                image_count=image_count, requires_visual_expansion=bool(image_count)),
        }
        if name is not None:
            result["name"] = name
        return result

    def prepare_sized(self, body: str, target: int = 8192, *, max_new: int = 128,
                      context: int = 16384, template_kwargs: dict | None = None) -> dict:
        """Trim only a user message's document, preserving its complete chat framing."""
        prefix = "Read the complete document below and summarize its engineering tradeoffs.\n\n"
        suffix = "\n\nProvide a concise summary of this document."

        def request(text):
            return self.prepare([{"role": "user", "content": prefix + text + suffix}],
                                max_new=max_new, context=context, template_kwargs=template_kwargs,
                                name=f"cold_pp_{target}")

        empty = request("")["prompt_tokens"]
        if target <= empty or target + max_new > context:
            raise ValueError("target prompt length cannot fit its framing/output in this context")
        body_ids = self.tokenizer.encode(body)
        if len(body_ids) < target:
            raise ValueError(f"source document has only {len(body_ids)} tokens; need at least {target}")
        guess, seen = min(target - empty, len(body_ids)), set()
        for _ in range(12):
            if guess in seen:
                break
            seen.add(guess)
            text = self.tokenizer.decode(body_ids[:guess], errors="ignore")
            result = request(text)
            delta = target - result["prompt_tokens"]
            if delta == 0:
                break
            guess = max(1, min(len(body_ids), guess + delta))
        else:
            result = None
        if not result or result["prompt_tokens"] != target:
            for length in range(max(1, guess - 16), min(len(body_ids), guess + 16) + 1):
                result = request(self.tokenizer.decode(body_ids[:length], errors="ignore"))
                if result["prompt_tokens"] == target:
                    break
            else:
                raise ValueError("could not fit exact complete prompt length using body token boundaries")
        result["workload"] = {"target_prompt_tokens": target, "body_source_tokens": len(body_ids),
                              "prefix_reuse": False, "eos_enabled": True}
        return result


def load_frontend(gguf, template_name: str | None = None) -> DenseFrontend:
    return DenseFrontend(gguf, template_name)


def _suite(frontend: DenseFrontend, context: int, kwargs: dict, pp_text: str | None) -> dict:
    short = frontend.prepare([{"role": "user", "content": "请用一句话解释水为什么会结冰。"}],
                             max_new=128, context=context, template_kwargs=kwargs, name="short_quality")
    long_task = (
        "Write a complete practical tutorial for a junior software engineer on designing a reliable "
        "local document search application. Use twelve numbered sections. Each section should contain "
        "180 to 220 English words and a concrete example, covering input validation, text extraction, "
        "chunking, indexes, ranking, updates, deletion, caching, concurrency, failure recovery, testing, "
        "and measurement. Explain your choices in connected prose. Finish with a short checklist."
    )
    long = frontend.prepare([{"role": "user", "content": long_task}], max_new=8192,
                            context=context, template_kwargs=kwargs, name="natural_long_output")
    if pp_text is None:
        paragraph = (
            "A local document search service keeps an index of records with stable identifiers. "
            "An update first validates the source, extracts text, and records the document version. "
            "Readers use an immutable index snapshot while a writer builds the next generation. "
            "The service reports extraction failures separately from empty search results. "
            "A bounded queue controls memory use, and cancellation releases the request state. "
            "Measurements include complete input processing, first response latency, throughput, "
            "and recovery after a failed update. Repeated content is intentionally part of this "
            "document workload, and every request must process its full supplied input.\n"
        )
        pp_text = "".join(f"Engineering note {i}: {paragraph}" for i in range(1, 401))
    pp = frontend.prepare_sized(pp_text, 8192, context=context, template_kwargs=kwargs)
    return {"model": str(frontend.path.resolve()), "metadata": frontend.identity,
            "cases": [short, long, pp]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--output", "--out", dest="output", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--messages", type=pathlib.Path, help="JSON messages list or request object")
    source.add_argument("--cases", type=pathlib.Path, help="JSON cases list or object with cases")
    source.add_argument("--suite", action="store_true", help="short, natural long output and full 8192 input")
    parser.add_argument("--pp-text", type=pathlib.Path, help="optional suite document in UTF-8")
    parser.add_argument("--template-name")
    parser.add_argument("--template-kwargs", default="{}", help="JSON; model defaults remain when omitted")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--no-generation-prompt", action="store_true")
    args = parser.parse_args()
    frontend = load_frontend(args.gguf, args.template_name)
    kwargs = json.loads(args.template_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--template-kwargs must be a JSON object")
    if args.suite:
        body = args.pp_text.read_text(encoding="utf-8") if args.pp_text else None
        result = _suite(frontend, args.context, kwargs, body)
    else:
        data = json.loads((args.cases or args.messages).read_text(encoding="utf-8")) \
            if args.cases or args.messages else [{"role": "user", "content": "请用一句话解释水为什么会结冰。"}]
        cases = data.get("cases") if isinstance(data, dict) else data
        if not args.cases:
            cases = [data if isinstance(data, dict) else {"messages": data}]
        prepared = []
        for case in cases:
            prepared.append(frontend.prepare(
                case["messages"], tools=case.get("tools"), name=case.get("name"),
                template_kwargs=dict(kwargs, **case.get("template_kwargs", {})),
                max_new=int(case.get("max_new_tokens", args.max_new)),
                context=int(case.get("context", args.context)),
                add_generation_prompt=case.get("add_generation_prompt", not args.no_generation_prompt)))
        result = {"model": str(frontend.path.resolve()), "metadata": frontend.identity,
                  "cases": prepared} if args.cases else prepared[0]
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    items = result.get("cases", [result])
    print(json.dumps({"output": str(output), "pre": frontend.tokenizer.pre,
                      "template_sha256": frontend.template_hash, "stop_ids": frontend.stop_ids,
                      "cases": [{"name": item.get("name"), "prompt_tokens": item["prompt_tokens"],
                                 "max_new_tokens": item["max_new_tokens"]} for item in items]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
