"""Bound sampled-MTP protocol using the shared Gemma drain and native runner."""
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
from serve.gemma_sm120_frontend import EOS
from serve.gemma_sm120_config import validate_config
from serve.gemma_original_contract import VARIANTS, weight_backend
from serve.gemma_sm120_mtp import request_depth, validate_request_extent
from serve.gemma_engine import GemmaRunner as DrainingRunner
from serve.native_engine import NativeFamilyRunner
from serve.native_slots import NativeSlots


def runner_type():
    class GemmaRunner(DrainingRunner):
        def __init__(self, executable, target, draft, gguf_python, context, temp_dir,
                     block_size=4, slots=1, dspark=False, *, family, variant, mtp,
                     kv_policy, environment, **kwargs):
            if (family != 'gemma4_dense' or variant not in VARIANTS.values() or
                    type(mtp) is not int or mtp not in (0, 1, 2) or slots != 1 or dspark):
                raise ValueError('Unsupported Gemma SM120 native runner configuration')
            self.cfg = dict(response_timeout_s=kwargs['response_timeout_s'])
            self.family, self.context, self.slots = family, context, 1
            self.temp_dir, self.draft, self.block_size = temp_dir, None, block_size
            self.dspark = self.dspark_requested = False
            self.dspark_disabled_reason = self.assistant_disabled_reason = None
            self.mtp_depth = self.mtp_requested = self.mtp_launch = mtp
            self._aborted, self.remaining = False, 0
            self.request_info = {}
            self.slot_status = NativeSlots(1)
            self.available = queue.Queue(maxsize=1); self.available.put(0)
            self.commands, self.lines = queue.Queue(), queue.Queue(maxsize=128)
            args = [str(executable), '--family', family, '--variant', variant,
                    '--model', str(target), '--context', str(context), '--slots', '1',
                    '--mtp', str(mtp), '--kv-policy', kv_policy]
            self.proc = subprocess.Popen(args, cwd=temp_dir, env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                text=True, encoding='utf-8', bufsize=1)
            threading.Thread(target=self._read_stdout, daemon=True).start()
            try:
                if self._line().strip() != 'READY slots=1':
                    raise RuntimeError('Gemma SM120 source entry did not become ready')
            except BaseException:
                self.abort(); self.close(); raise
            self.io_thread = threading.Thread(target=self._serve_commands, daemon=True)
            self.io_thread.start()

        def _command(self, line, expected):
            parts = line.split()
            if parts[0] == 'PREPARE':
                self.remaining = int(parts[2])
            elif parts[0] == 'PREPAREV':
                raise ValueError('Gemma visual provider is not registered')
            elif parts[0] == 'STEP':
                if len(parts) != 3 or self.remaining < 1:
                    raise ValueError('Gemma pending/remaining STEP mismatch')
                line += ' ' + str(self.remaining)
            value = super()._command(line, expected)
            if value is not None and expected == 'PREFILL':
                if (int(value.get('speculative', '0')) != int(self.mtp_depth > 0) or
                        int(value.get('mtp', '0')) != self.mtp_depth):
                    raise RuntimeError('Gemma PREFILL did not activate the bound MTP controller')
                self.request_info.update(value)
                self.request_info['controller_bound'] = value.get('speculative', '0')
                self.remaining -= 1
            elif value is not None and expected == 'STEPPED':
                ids = [int(token) for token in value['ids'].split(',')]
                count = len(ids)
                proposed, accepted = int(value['proposed']), int(value['accepted'])
                if (not 1 <= count <= min(self.mtp_depth + 1, self.remaining) or
                        not 0 <= accepted <= proposed <= self.mtp_depth or count > proposed + 1 or
                        any(token < 0 or token >= 262144 for token in ids) or
                        int(value['next']) != ids[-1] or any(token in EOS for token in ids[:-1])):
                    raise RuntimeError('Gemma MTP response changed token/budget/EOG work')
                self.request_info.update(value)
                self.remaining -= count
            elif value is not None and expected == 'RELEASED':
                self.request_info.update(value)
                self.remaining = 0
            return value

        def _generate(self, *args, **kwargs):
            self.request_info = {}
            ids, stats = NativeFamilyRunner._generate(self, *args, **kwargs)
            # Native INFO is intentionally absent in this frozen protocol.
            # RELEASE updates this same object before generate's finally returns.
            effective = self.mtp_depth if stats['checked_proposals'] else 0
            self.request_info.update(configured_mtp=str(self.mtp_depth), actual_mtp=str(effective))
            stats['native_speculative'] = bool(effective)
            stats['native_info'] = self.request_info
            return ids, stats

    return GemmaRunner


def engine_type():
    from serve.native_family import NativeFamilyEngine
    from serve.gemma_family import GemmaFamilyEngine

    class GemmaEngine(GemmaFamilyEngine):
        def __init__(self, cfg, frontend, environment, *, lazy=False):
            cfg.setdefault('response_timeout_s', 900)
            validate_config(cfg)
            backend = weight_backend(cfg)
            self.cfg, self.frontend, self.family, self.max_context = cfg, frontend, 'gemma4_dense', cfg['context']
            self.batch = self.concurrent_slots = 1
            self.stops = set(EOS)
            self.environment = dict(environment)
            self.environment.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                TEMP=cfg['temp_dir'], TMP=cfg['temp_dir'], TMPDIR=cfg['temp_dir'],
                CUDA_CACHE_PATH=str(Path(cfg['temp_dir'])/'cuda-cache-gemma-sm120'),
                STRATA_DENSE_RESERVE_MIB=str(cfg.get('reserve_mib', 4096)),
                STRATA_GEMMA4_MAX_BATCH=str(cfg.get('max_batch', 2048)),
                STRATA_GEMMA4_GRAPH='1' if cfg.get('cuda_graph', True) else '0',
                STRATA_GEMMA4_MTP_DRAFT=cfg.get('draft', '') if cfg['mtp'] else '',
                STRATA_GEMMA_WEIGHT_FORMAT=backend,
                STRATA_GEMMA_ORIGINAL_INPUT_REUSE='1' if cfg.get('original_input_reuse', False) else '0',
                STRATA_GEMMA_PP_TC='1' if cfg.get('pp_tc', False) else '0',
                STRATA_GEMMA_PP_GLOBAL_KEYS1024='1' if cfg.get('pp_global_keys1024', False) else '0',
                STRATA_GEMMA_PP_TC_LOCAL_GQA2='1' if cfg.get('pp_tc_local_gqa2', False) else '0',
                STRATA_GEMMA_PP_ROPE_TABLE='1' if cfg.get('pp_rope_table', False) else '0',
                STRATA_GEMMA_T1_ROPE_TABLE='1' if cfg.get('t1_rope_table', False) else '0',
                STRATA_GEMMA_T1_NORM_CACHE_FUSION='1' if cfg.get('t1_norm_cache_fusion', False) else '0',
                STRATA_GEMMA_TG_GLOBAL_MERGE='1' if cfg.get('tg_global_merge', False) else '0',
                STRATA_GEMMA_TG_GLOBAL_GQA='1' if cfg.get('tg_global_gqa', False) else '0',
                STRATA_GEMMA_PP_GLOBAL_FLASH='1' if cfg.get('pp_global_flash', False) else '0',
                STRATA_GEMMA_PP_GLOBAL_FLASH_F32PV='1' if cfg.get('pp_global_flash_f32pv', False) else '0',
                STRATA_GEMMA_ORIGINAL_Q6_HEAD='1' if cfg.get('original_q6_head', False) else '0',
                STRATA_GEMMA_PP_WHOLE_HISTORY='1' if cfg.get('pp_whole_history', False) else '0',
                STRATA_GEMMA_TG_LOCAL270='1' if cfg.get('tg_local270', False) else '0',
                STRATA_GEMMA4_TINY_COMPENSATED='1' if cfg.get('tiny_compensated', False) else '0',
                STRATA_DENSE_TOP20_SELECTION='0', STRATA_DENSE_TOP20_PREPARE_FUSION='0')
            if cfg.get('dll_dirs'):
                self.environment['PATH'] = os.pathsep.join(cfg['dll_dirs'] + [self.environment.get('PATH', '')])
            self.info = dict(backend='native-gemma4_dense', weight_backend=backend, variant=cfg['variant'], gpu_uuid=cfg['gpu_uuid'], request_slots=1,
                batch_slots=1, dspark=False, mtp=cfg['mtp'], mtp_requested=cfg['mtp'],
                mtp_launch=cfg['mtp'], mtp_effective=0, assistant_disabled_reason=None,
                kv_policy='full', context=self.max_context,
                max_batch=cfg.get('max_batch', 2048), qualification='UNQUALIFIED_CANDIDATE',
                original_input_reuse=cfg.get('original_input_reuse', False),
                tiny_compensated=cfg.get('tiny_compensated', False), pp_tc=cfg.get('pp_tc', False),
                pp_global_keys1024=cfg.get('pp_global_keys1024', False),
                pp_tc_local_gqa2=cfg.get('pp_tc_local_gqa2', False),
                pp_rope_table=cfg.get('pp_rope_table', False),tg_global_merge=cfg.get('tg_global_merge', False),
                tg_global_gqa=cfg.get('tg_global_gqa', False),
                pp_global_flash=cfg.get('pp_global_flash', False),
                pp_global_flash_f32pv=cfg.get('pp_global_flash_f32pv', False),
                original_q6_head=cfg.get('original_q6_head', False),
                pp_whole_history=cfg.get('pp_whole_history', False),
                t1_rope_table=cfg.get('t1_rope_table', False),
                t1_norm_cache_fusion=cfg.get('t1_norm_cache_fusion', False),
                tg_local270=cfg.get('tg_local270', False),
                draft_asset_binding=cfg.get('draft_asset_binding') if cfg['mtp'] else None,
                MTP_load_policy='fixed_depth_new_residency', sampled_MTP_qualified=False,
                draft_context=131072 if cfg['mtp'] else None,
                outside_draft_context='sampled_target_T1' if cfg['mtp'] else None,
                capabilities={'text_sampling':'source_candidate', 'MTP':bool(cfg['mtp']), 'vision':False,
                              'actual_HTTP_SSE_verified':False, 'dynamic_RAM_history':False})
            self._last = {}; self.request_state = threading.local()
            self.progress = self.prompt_perf = self.request_phase = None
            self.frontend_facts = {}; self.runner = None; self.runner_type = None
            self._gate, self._draining = threading.RLock(), None
            self.unloaded = lazy
            if not lazy:
                self.restart()

        def restart(self):
            with self._gate:
                validate_config(self.cfg)
                gpu = self.cfg['gpu_uuid']
                observed = subprocess.run(['nvidia-smi', '--id='+gpu, '--query-gpu=uuid,compute_cap',
                    '--format=csv,noheader'], capture_output=True, text=True, check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0).stdout.strip().split(',')
                if len(observed) != 2 or observed[0].strip() != gpu or float(observed[1]) != 12.0:
                    raise ValueError('Gemma requires the selected SM120 GPU')
                runner = runner_type()
                timeout = self.cfg['response_timeout_s']
                self.runner_type = lambda *a, **kw: runner(*a, response_timeout_s=timeout, **kw)
                NativeFamilyEngine.restart(self)
                self.unloaded = False

        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            if embeddings or getattr(ids, 'visual_path', None):
                raise ValueError('Gemma visual provider is not registered')
            request_depth(self.cfg, sampling)
            validate_request_extent(self.cfg, len(ids), max_new)
            if self.runner is None:
                raise RuntimeError('Gemma native engine has not been started')
            return super().generate(ids, max_new, sampling, cancel, embeddings=None)

    return GemmaEngine
