# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from pathlib import Path
from tqdm import tqdm

from download_utils import BASE_URL, download_file, extract_multipart

ROOT = Path(__file__).resolve().parent

capture_id_list = [
        'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands',
        'm--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands',
        'm--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands',
        'm--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands',
        'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands',
        'm--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands'
]

name_list = {
        'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands': ['aa'],
        'm--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands': ['aa'],
        'm--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands': ['aa'],
        'm--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands': ['aa'],
        'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands': ['aa'],
        'm--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands': ['aa']
}

def download(capture_id):
    output_dir = ROOT / capture_id / 'keypoints_orig'
    for name in name_list[capture_id]:
        download_file(f"{BASE_URL}/{capture_id}/keypoints_orig/keypoints_orig.tar.gz{name}", output_dir)
    extract_multipart(output_dir, 'keypoints_orig.tar.gz')

def main():
    for capture_id in tqdm(capture_id_list):
        (ROOT / capture_id).mkdir(parents=True, exist_ok=True)
        download(capture_id)


if __name__ == '__main__':
    main()
