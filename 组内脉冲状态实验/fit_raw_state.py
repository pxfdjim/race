"""Fit the two matched state networks and export their validation scores."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_raw_state import FOLD_NAMES


ARMS = ('summary_mlp', 'raw_pulse_mlp')
SEED = 42


def digest(path):
    checksum = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def fit(prepared, output, epochs, threads):
    if output.exists():
        raise FileExistsError(output)
    meta_path = prepared / 'prepared.json'
    meta = json.loads(meta_path.read_text())
    for name, expected in meta['prepared_files_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('pulse_state_mlp.cpp')
    with TemporaryDirectory(prefix='.raw-state-fit-', dir=output.parent) as scratch:
        scratch = Path(scratch)
        exe = scratch / 'state_mlp'
        subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-fopenmp', str(source), '-o', str(exe)], check=True)
        stage = scratch / 'result'
        stage.mkdir()

        def arm_run(arm):
            env = os.environ | {'OMP_NUM_THREADS': str(threads), 'OMP_DYNAMIC': 'FALSE'}
            log = stage / f'{arm}.log'
            model = stage / f'{arm}.model'
            raw = prepared / 'pulses.bin'
            cmd = [str(exe), 'train', str(raw), str(prepared / 'development.bin'), str(prepared / 'selection.bin'),
                   str(model), str(int(arm == 'raw_pulse_mlp')), str(epochs), str(SEED)]
            with log.open('w', encoding='utf-8') as trace:
                with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True, stderr=subprocess.STDOUT) as proc:
                    for line in proc.stdout:
                        trace.write(line)
                        print(arm, line.rstrip(), flush=True)
                    if proc.wait():
                        raise subprocess.CalledProcessError(proc.returncode, cmd)
            for fold in FOLD_NAMES[1:]:
                path = stage / f'{arm}_{fold}.bin'
                subprocess.run([str(exe), 'predict', str(raw), str(prepared / f'{fold}.bin'), str(model), str(path)],
                               env=env, check=True)
                if path.stat().st_size != 12 * meta['fold_sequences'][fold]:
                    raise ValueError(f'Unexpected number of predictions for {arm}/{fold}')

        with ThreadPoolExecutor(max_workers=2) as executor:
            tuple(executor.map(arm_run, ARMS))
        record = {'epochs': epochs, 'threads_per_arm': threads, 'seed': SEED,
                  'backend_sha256': digest(source), 'prepared_manifest_sha256': digest(meta_path),
                  'files_sha256': {p.name: digest(p) for p in stage.iterdir()}}
        (stage / 'fit.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not (args.epochs > 0 and args.threads > 0):
        parser.error('epochs and threads must be positive')
    fit(args.prepared_dir.resolve(), args.output_dir.resolve(), args.epochs, args.threads)


if __name__ == '__main__':
    main()
