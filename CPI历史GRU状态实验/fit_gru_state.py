"""Fit the history GRU and its one-step control, then score each validation fold."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_gru_state import FOLDS


ARMS = ('current_cpi', 'history_gru')
SOURCE = Path(__file__).with_name('gru_state.cpp')
SEED = 42


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fit(prepared, output, epochs, threads):
    if output.exists():
        raise FileExistsError(output)
    manifest = prepared / 'prepared.json'
    info = json.loads(manifest.read_text(encoding='utf-8'))
    for name, expected in info['files_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {'OMP_NUM_THREADS': str(threads), 'OMP_DYNAMIC': 'FALSE'}
    with TemporaryDirectory(prefix='.gru-fit-', dir=output.parent) as folder:
        exe = Path(folder) / 'gru_state'
        subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-fopenmp', str(SOURCE), '-o', str(exe)], check=True)
        check = subprocess.run([str(exe), 'selftest'], check=True, text=True, capture_output=True).stdout.strip()
        print(check, flush=True)
        stage = Path(folder) / 'fit'
        stage.mkdir()

        def train_arm(arm):
            model = stage / f'{arm}.model'
            cmd = [str(exe), 'train', str(prepared / 'development.bin'), str(prepared / 'selection.bin'), str(model),
                   str(int(arm == 'history_gru')), str(epochs), str(SEED)]
            with (stage / f'{arm}.log').open('w', encoding='utf-8') as log:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as process:
                    for line in process.stdout:
                        log.write(line)
                        print(arm, line.rstrip(), flush=True)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, cmd)
            for fold in FOLDS[1:]:
                scores = stage / f'{arm}_{fold}.bin'
                subprocess.run([str(exe), 'predict', str(model), str(prepared / f'{fold}.bin'), str(scores)],
                               check=True, env=env)
                if scores.stat().st_size != 12 * info['folds'][fold]['sequences']:
                    raise ValueError(f'Predict lost sequence rows: {arm}/{fold}')

        with ThreadPoolExecutor(max_workers=len(ARMS)) as executor:
            tuple(executor.map(train_arm, ARMS))
        result = {'prepared_sha256': digest(manifest), 'native_sha256': digest(SOURCE), 'gradient_check': check,
                  'epochs': epochs, 'threads': threads, 'seed': SEED,
                  'files_sha256': {p.name: digest(p) for p in stage.iterdir()}}
        (stage / 'fit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    fit(args.prepared_dir.resolve(), args.output_dir.resolve(), args.epochs, args.threads)


if __name__ == '__main__':
    main()
