"""Exercise concurrent slot identity and error/cancellation cleanup without a GPU."""
import queue
import unittest
from serve.native_engine import NativeFamilyRunner
from serve.native_family import NativeFamilyEngine
from serve.native_slots import NativeSlots


class SlotContract(unittest.TestCase):
    def test_two_requests_and_common_endpoint_fields(self):
        slots = NativeSlots(2)
        engine = object.__new__(NativeFamilyEngine)
        engine.batch = 2
        engine.runner = type('Runner', (), {'slot_status': slots})()
        self.assertEqual([x['state'] for x in engine.slots_view()], ['idle', 'idle'])
        slots.begin(0, 47, 64)
        slots.begin(1, 92, 128)
        slots.token(1)
        rows = engine.slots_view()
        self.assertEqual([x['slot'] for x in rows], [0, 1])
        self.assertEqual([x['prompt_tokens'] for x in rows], [47, 92])
        self.assertEqual([x['state'] for x in rows], ['reading', 'generating'])
        slots.release(0)
        self.assertEqual(engine.slots_view()[1]['generated'], 1)

    def test_release_error_preserves_original_error_and_frees_telemetry(self):
        runner = object.__new__(NativeFamilyRunner)
        runner.available = queue.Queue()
        runner.available.put(0)
        runner.slot_status = NativeSlots(1)
        def fail(*args):
            raise ValueError('request failure')
        runner._generate = fail
        runner._command = fail
        with self.assertRaisesRegex(ValueError, 'request failure'):
            runner.generate([1, 2], 4, {3})
        self.assertEqual(runner.available.get_nowait(), 0)
        self.assertEqual(runner.slot_status.view()[0]['state'], 'idle')


if __name__ == '__main__':
    unittest.main()
