"""Integer-only diagnostic: actual masks and every admitted N against CPU counts."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch_musa

REPORT = {'status': 'running', 'completed_pattern_shapes': 0, 'actual_cases': [],
          'patterns': ['zeros', 'ones', 'sparse_period19', 'column_and_tail_period7']}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    proof_path = Path(os.environ['TE_COUNTS_PROOF'])
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'passed_same_call_samples'
    assert proof['all_captured_meet_column_error_preconditions']
    assert proof['verified_count_error_samples'] == 10
    raw = Path(os.environ['TE_COUNTS_ACTUAL_RAW'])
    REPORT.update(proof_sha256=sha(proof_path), source_sha256=sha(Path(__file__)),
                  torch=torch.__version__, torch_musa=torch_musa.__version__,
                  runtime_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  scope='Only raw native Bool and native I64 input column sums, no TE/Float/PG/'
                        'model/optimizer/workaround install. Integer shapes only; no whole-model '
                        'or performance qualification.')
    torch.musa.set_device(0)
    for rank in range(8):
        observed = json.loads((raw / f'te_native_counts_rank{rank}.json').read_text())
        for item in observed['observations']['samples']:
            REPORT['stage'] = f'actual_rank{rank}_{item["category"]}'
            path = raw / item['mask_file']
            assert sha(path) == item['mask_sha256']
            cpu = np.load(path, allow_pickle=False)
            expected = cpu.sum(0, dtype=np.int64)
            mask = torch.from_numpy(cpu).to('musa')
            native = mask.sum(0, dtype=torch.int64).cpu().numpy()
            repaired = mask.to(torch.int64).sum(0).cpu().numpy()
            case = {'rank': rank, 'tokens': len(cpu), 'original_equal': bool(np.array_equal(native, expected)),
                    'I64_input_equal': bool(np.array_equal(repaired, expected)),
                    'original_matches_captured': native.tolist() == item['native_counts']}
            REPORT['actual_cases'].append(case)
            assert case['I64_input_equal'] and case['original_matches_captured']
            del mask
    assert len(REPORT['actual_cases']) == 10
    tokens = list(range(24577, 32768))
    tokens += [1, 17, 257, 8192, 16384, 24575, 24576, 32768, 32769, 40960, 41047]
    n_values = np.asarray(tokens, dtype=np.int64)
    REPORT['N_values'] = tokens
    count = len(tokens)
    originals = np.full((4, count, 32), -1, dtype=np.int64)
    i64_inputs = np.full_like(originals, -1)
    gold = np.full_like(originals, -1)
    completed = np.zeros((4, count), dtype=np.bool_)
    rows = np.arange(max(tokens), dtype=np.int64)[:, None]
    columns = np.arange(32, dtype=np.int64)[None, :]
    patterns = [np.zeros((max(tokens), 32), dtype=np.bool_),
                np.ones((max(tokens), 32), dtype=np.bool_),
                (rows + columns * 3) % 19 == 0,
                ((rows * 3 + columns) % 7 <= 1) & (columns % 5 != 0)]
    begin = time.perf_counter()
    for pattern_id, cpu in enumerate(patterns):
        REPORT['stage'] = f'pattern_{REPORT["patterns"][pattern_id]}'
        prefix = cpu.cumsum(0, dtype=np.int64)
        gpu = torch.from_numpy(cpu).to('musa')
        for index, n in enumerate(tokens):
            REPORT['current_N'] = n
            mask = gpu[:n]
            expected = prefix[n - 1]
            native = mask.sum(0, dtype=torch.int64).cpu().numpy()
            long_sum = mask.to(torch.int64).sum(0).cpu().numpy()
            originals[pattern_id, index] = native
            i64_inputs[pattern_id, index] = long_sum
            gold[pattern_id, index] = expected
            completed[pattern_id, index] = True
            REPORT['completed_pattern_shapes'] += 1
            REPORT['last_comparison'] = {'pattern_id': pattern_id, 'N': n,
                'original_equal': bool(np.array_equal(native, expected)),
                'I64_input_equal': bool(np.array_equal(long_sum, expected)),
                'native_counts': native.tolist(), 'I64_counts': long_sum.tolist(),
                'expected_counts': expected.tolist()}
            assert np.array_equal(long_sum, expected), (pattern_id, n, long_sum.tolist(), expected.tolist())
            if (index + 1) % 1024 == 0:
                print(f'pattern={pattern_id} checked={index + 1}/{count} elapsed={time.perf_counter()-begin:.1f}s', flush=True)
        del gpu, prefix
    arrays_path = Path(os.environ['TE_COUNTS_SUM_ARRAYS'])
    np.savez_compressed(arrays_path, N_values=n_values, original=originals,
                        I64_input=i64_inputs, gold=gold, completed=completed)
    REPORT['array_file'] = str(arrays_path)
    REPORT['array_sha256'] = sha(arrays_path)
    REPORT['original_wrong_N'] = {
        name: n_values[np.any(originals[index] != gold[index], axis=1)].tolist()
        for index, name in enumerate(REPORT['patterns'])}
    assert sha(Path(__file__)) == REPORT['source_sha256']
    assert sha(proof_path) == REPORT['proof_sha256']
    REPORT['status'] = 'passed_integer_shape_sweep'
    REPORT['admitted_integer_N_interval_inclusive'] = [24577, 32767]
    REPORT['total_seconds_diagnostic_not_benchmark'] = time.perf_counter() - begin


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        REPORT.update(status='failed', error=repr(error))
        raise
    finally:
        Path(os.environ['TE_COUNTS_SUM_REPORT']).write_text(json.dumps(REPORT, indent=2) + '\n')
        print(json.dumps({k: v for k, v in REPORT.items() if k not in ('N_values', 'original_wrong_N')}), flush=True)
