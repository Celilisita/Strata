"""Measured prefill progress; unknown progress stays unknown until a real sample."""
import math
import threading
import time


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


class PrefillMeter:
    def __init__(self, total):
        self.total = total
        self.started = time.time()
        self.clock = time.perf_counter()
        self.lock = threading.Lock()
        self.read = 0
        self.base = None
        self.ms = 0.0
        self.current = self.mean = None
        self.updated = None

    def update(self, read, total, milliseconds):
        if (type(read) is not int or total != self.total or not 0 <= read <= total or
                not positive(milliseconds)):
            raise ValueError('invalid native prefill progress')
        with self.lock:
            if read < self.read or milliseconds < self.ms:
                raise ValueError('native prefill progress moved backwards')
            if self.base is None:
                # A resumed request reports absolute positions, so its first
                # sample has no measured interval behind the reported read.
                self.base = read
                self.read, self.ms, self.updated = read, milliseconds, time.time()
                return
            if read > self.read and milliseconds > self.ms:
                self.current = 1000 * (read - self.read) / (milliseconds - self.ms)
            self.read, self.ms = read, milliseconds
            self.mean = 1000 * (read - self.base) / milliseconds if read > self.base else None
            self.updated = time.time()

    def complete(self, milliseconds=None, read=None):
        with self.lock:
            done = self.read == self.total
        if milliseconds is None and done:
            return
        milliseconds = milliseconds or 1000 * (time.perf_counter() - self.clock)
        self.update(self.total if read is None else read, self.total, milliseconds)

    def snapshot(self):
        with self.lock:
            return dict(read=self.read, total=self.total, current=self.current, mean=self.mean,
                        started=self.started, updated=self.updated)


def view(service, status=None):
    if status is None:
        with service.status_lock:
            status = dict(service.status)
            if service.live_reqs:
                status.update(list(service.live_reqs.values())[-1][0])
    result = dict(prefill_tok_s=None, prefill_tok_s_mean=None,
                  prefill_sample_age_s=None, prefill_rate_source=None)
    if not status.get('busy'):
        return result
    engine = service.engine
    meter = getattr(engine, 'prefill_meter', None)
    sample = meter.snapshot() if meter is not None else None
    if sample and sample['started'] >= status.get('started', 0) - .01:
        result.update(prefill_tok_s=sample['current'] if not status.get('first_token') else None,
                      prefill_tok_s_mean=sample['mean'],
                      prefill_rate_source='completed_native_chunks',
                      prompt_read=sample['read'], prompt_total=sample['total'])
        if sample['updated']:
            result['prefill_sample_age_s'] = round(max(0, time.time() - sample['updated']), 2)
        return result
    observed = getattr(getattr(engine, 'runner', None), 'prefill_observation', None)
    if (observed and observed['at'] >= status.get('started', 0) and
            positive(observed['ms'])):
        result.update(prefill_tok_s_mean=1000 * observed['tokens'] / observed['ms'],
                      prefill_rate_source='native_prefill_ack', prompt_read=observed['tokens'],
                      prompt_total=observed['tokens'])
        return result
    mean = getattr(engine, 'prefill_tok_s_mean', None)
    instant = getattr(engine, 'prefill_tok_s', None)
    if positive(instant) and not status.get('first_token'):
        # An engine that prints no PP line (dsv41) is measured from its own reading progress instead.
        result.update(prefill_tok_s=instant, prefill_rate_source='native_reported_now')
    if positive(mean):
        result.update(prefill_tok_s_mean=mean, prefill_rate_source='native_reported_mean')
    elif status.get('first_token') and getattr(engine, 'cfg', {}).get('family') in (
            'small_dense', 'qwen35_moe', 'qwen35_dense', 'qwen35_dense_v100',
            'gemma4_dense', 'gemma4_sm120'):
        # Native adapters start from a reset request. This is observed request
        # wall time through the first token, never a claim about in-flight work.
        seconds = status['first_token'] - status.get('started', status['first_token'])
        if seconds > 0:
            result.update(prefill_tok_s_mean=status['prompt_tokens'] / seconds,
                          prefill_rate_source='observed_first_token_wallclock',
                          prompt_read=status['prompt_tokens'], prompt_total=status['prompt_tokens'])
    return result


def apply(service, live, status):
    live.update(view(service, status))


def rates(service):
    prefill = view(service)
    return dict(tok_s=service._tok_s(), tok_s_mean=service._tok_s_mean(),
                prefill_tok_s=prefill['prefill_tok_s'],
                prefill_tok_s_mean=prefill['prefill_tok_s_mean'])
