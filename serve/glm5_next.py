"""GLM's resident HTTP adapter, GGUF stop tokens and causal image embedding boundaries."""
from __future__ import annotations

import math

from serve.server import StrataEngine


def stop_ids(tokenizer):
    keys = ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id")
    ids = {int(tokenizer.special_ids[k]) for k in keys if k in tokenizer.special_ids}
    if not ids or any(i < 0 or i >= len(tokenizer.tokens) for i in ids):
        raise ValueError("GLM GGUF stop-token contract is absent or invalid")
    return ids


def expand_images(tokenizer, ids, encoded):
    begin = tokenizer.ids["<|begin_of_image|>"]
    image = tokenizer.ids["<|image|>"]
    end = tokenizer.ids["<|end_of_image|>"]
    result, count = [], 0
    for at, token in enumerate(ids):
        if token == image and at > 0 and at + 1 < len(ids) and ids[at - 1] == begin and ids[at + 1] == end:
            if count >= len(encoded) or encoded[count][1] < 1:
                raise ValueError("GLM image placeholders and encoded images differ")
            result.extend([-1] * encoded[count][1])
            count += 1
        else:
            result.append(token)
    if count != len(encoded):
        raise ValueError("GLM image placeholders and encoded images differ")
    return result, count


def configured(cfg):
    """Select only the GLM native line protocol; other families keep their existing adapter."""
    args = cfg.get("args") or []
    return cfg.get("family") == "glm5_next" or any(
        arg == "--family=glm5_next" or (arg == "--family" and args[i + 1:i + 2] == ["glm5_next"])
        for i, arg in enumerate(args))


class Glm5NextEngine(StrataEngine):
    """GLM's full-vocabulary sampler over the normal resident GEN/STOP/DONE protocol."""
    FLOAT_KEYS = {"temperature": "temperature", "top_p": "top_p", "min_p": "min_p",
                  "repetition_penalty": "penalty_repeat", "frequency_penalty": "penalty_freq",
                  "presence_penalty": "penalty_present"}
    INT_KEYS = {"top_k": 262144, "penalty_last_n": 262144, "seed": (1 << 64) - 1}
    BOOL_KEYS = ("ignore_eog", "strata_mtp", "mtp")

    @staticmethod
    def sampling_keys(sampling):
        keys = []
        for name, native in Glm5NextEngine.FLOAT_KEYS.items():
            value = sampling.get(name)
            if value is None:
                continue
            try:
                valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(value))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name}: expected a finite number")
            if ((name == "temperature" and value < 0) or
                    (name == "top_p" and not 0 < value <= 1) or
                    (name == "min_p" and not 0 <= value <= 1) or
                    (name == "repetition_penalty" and value <= 0)):
                raise ValueError(f"{name}: outside the GLM sampling range")
            keys.append(f"{native}={float(value)!r}")
        for name, maximum in Glm5NextEngine.INT_KEYS.items():
            value = sampling.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{name}: expected an integer 0..{maximum}")
            keys.append(f"{name}={value}")
        for name in Glm5NextEngine.BOOL_KEYS:
            if sampling.get(name) is not None and not isinstance(sampling[name], bool):
                raise ValueError(f"{name}: expected true or false")
        if sampling.get("ignore_eog") is True:
            raise ValueError("ignore_eog=true is unavailable through HTTP: the frontend honors GGUF stop tokens")
        if sampling.get("ignore_eog") is False:
            keys.append("ignore_eog=0")
        mtp, alias = sampling.get("strata_mtp"), sampling.get("mtp")
        if mtp is not None and alias is not None and mtp != alias:
            raise ValueError("strata_mtp and mtp disagree")
        mtp = mtp if mtp is not None else alias
        if mtp is not None:
            keys.append(f"mtp={int(mtp)}")
        for name in ("strata_tune", "strata_prefix", "experimental_speed_projection"):
            if sampling.get(name) is not None:
                raise ValueError(f"{name}: unavailable in the GLM engine")
        return "".join(" " + key for key in keys)

    @staticmethod
    def validate_request(req):
        """Run before HTTP response headers or integer coercion, without changing other families."""
        budgets = []
        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = req.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value <= 262144:
                raise ValueError(f"{name}: expected an integer -1..262144 (0/-1: remaining context)")
            budgets.append(value)
        if len(set(budgets)) > 1:
            raise ValueError("the supplied generation token budgets disagree")
        if req.get("stream") is not None and not isinstance(req["stream"], bool):
            raise ValueError("stream: expected true or false")
        if req.get("n") is not None and (type(req["n"]) is not int or req["n"] != 1):
            raise ValueError("GLM serves one completion per request (n=1)")
        for name in ("best_of", "logit_bias", "logprobs", "top_logprobs"):
            if req.get(name) not in (None, False, 0, {}):
                raise ValueError(f"{name}: unavailable in the GLM engine")
        Glm5NextEngine.sampling_keys(req)
        # Canonicalize the alias before Service merges config defaults, so a request overrides either spelling.
        if req.get("mtp") is not None:
            req["strata_mtp"] = req.pop("mtp")

    @staticmethod
    def sampling_defaults(cfg):
        values = cfg.get("sampling")
        values = {} if values is None else values
        if not isinstance(values, dict):
            raise SystemExit("[strata] config sampling: expected an object")
        supported = set(Glm5NextEngine.FLOAT_KEYS) | set(Glm5NextEngine.INT_KEYS) | set(Glm5NextEngine.BOOL_KEYS)
        out = {}
        for name, value in values.items():
            if name in supported and value is not None:
                out[name] = value
            elif name not in supported:
                print(f"[strata] config sampling.{name}={value!r}: unknown key, ignored", flush=True)
        try:
            Glm5NextEngine.validate_request(out)
        except ValueError as error:
            raise SystemExit(f"[strata] config sampling: {error}") from error
        return out

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if isinstance(max_new, bool) or not isinstance(max_new, int) or not 1 <= max_new <= 262144:
            raise ValueError("GLM max_new must be an integer 1..262144")
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)
