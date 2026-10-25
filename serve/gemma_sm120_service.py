"""Gemma SM120 registration in the shared HTTP/SSE/Responses service."""
from serve.frontend import images_of
from serve.server import Service, sampling_defaults_from_config
from serve.gemma_sm120_frontend import EOS, GemmaFrontend, GemmaTemplate
from serve.gemma_sm120_config import validate_config
from serve.gemma_sm120_provider import engine_type
from serve.gemma_sm120_mtp import request_depth
from serve.gemma_original_contract import VARIANTS


def configured(cfg):
    return cfg.get('family') == 'gemma4_dense' and cfg.get('variant') in VARIANTS.values()


def frontend_for_config(cfg):
    validate_config(cfg)
    return GemmaFrontend(cfg)


class GemmaSm120Service(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_ids = set(EOS)
        # #123's close and wrap-up in this template's channel form (the parser reads the channels)
        self.reasoning_close = '\n<channel|>\n\n'
        self.reasoning_wrap_up = '\n\nI have thought about this long enough; time to give my answer.\n<channel|>\n\n'

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None):
        if images_of(messages) or tools:
            raise ValueError('Gemma image/tool providers are not registered in this text candidate')
        cfg, request = self.engine.cfg, req or {}
        request_depth(cfg, request)
        if any(request.get(k) is not None for k in
               ('strata_prefix', 'strata_native_matrix', 'strata_native_pp_reuse')):
            raise ValueError('Unsupported Gemma native request policy')
        if max_new is None or (max_new == 0 and all(request.get(k) is None for k in
                ('max_completion_tokens', 'max_tokens', 'max_output_tokens'))):
            max_new = cfg.get('default_max_new', 2048)
        if type(max_new) is not int or max_new <= 0:
            raise ValueError('Gemma requires a positive actual output token budget')
        kwargs = dict(kwargs)
        kwargs.setdefault('enable_thinking', True)
        return super().prepare(messages, tools, kwargs, max_new, force, req)


GemmaSm120Engine = engine_type()


def make_service(cfg, environment, *, start=False):
    frontend = frontend_for_config(cfg)
    engine = GemmaSm120Engine(cfg, frontend, environment, lazy=not start)
    defaults = sampling_defaults_from_config(cfg)
    defaults.setdefault('temperature', 0.6)
    return GemmaSm120Service(engine, frontend.tokenizer, GemmaTemplate(cfg['tokenizer']),
        model_name=cfg.get('model_name', cfg['variant']+'-candidate'), vision=None,
        sampling_defaults=defaults, fit_max_tokens=False)
