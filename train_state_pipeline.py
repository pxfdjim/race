#!/usr/bin/env python3
"""Reproduce state features, comparison models and the final state model bundle."""
import argparse
from pathlib import Path
from prepare_state_features import run as prepare_windows
from prepare_sequence_states import run as prepare_sequences
from augment_state_features import run as augment
from train_target_state import run as train_windows
from train_calibrated_state import run as train_sequences
from select_state_accuracy import run as select_accuracy
from assemble_state_model import run as assemble
from src.state_features import StateFeatureConfig

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--work-dir',type=Path,default=ROOT/'output/state_training_new')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/target_state_new')
    a=p.parse_args();work=a.work_dir.resolve()
    if work.exists() or a.model_dir.exists():raise FileExistsError('Choose fresh state training and model directories')
    for split in ['train','test']:
        prepare_windows(a.pipeline_root,split,work/'state_features'/split,StateFeatureConfig())
        prepare_sequences(a.pipeline_root,work/'state_features',split,work/'sequence_state_features'/split)
        augment(a.pipeline_root,work/'sequence_state_features',work/'calibrated_sequence_state_features',split)
    train_windows(a.pipeline_root,work/'state_features',work/'models/window_baseline')
    train_sequences(a.pipeline_root,work/'calibrated_sequence_state_features',work/'models/sequence_calibrated')
    select_accuracy(work/'calibrated_sequence_state_features',work/'models/sequence_calibrated',work/'models/sequence_accuracy')
    assemble(work/'models/window_baseline',work/'models/sequence_accuracy',a.model_dir,work/'models/sequence_calibrated')


if __name__=='__main__':main()
