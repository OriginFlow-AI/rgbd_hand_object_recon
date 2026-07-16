# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import hashlib
from pathlib import Path
from tqdm import tqdm

current_path = Path(__file__).resolve().parent
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

def verify_capture(capture_id):
    print('Checking ' + capture_id)
    capture_dir = current_path / capture_id
    checksum_filename = 'CHECKSUM'
    checksum_path = capture_dir / checksum_filename
    if not checksum_path.is_file():
        print(str(checksum_path) + ' is missing. Please download it again.')
        raise SystemExit(1)
    with checksum_path.open() as f:
        checksums = f.readlines()

    good = True
    results = []
    for line in tqdm(checksums):
        filename, md5sum = line.split()

        relative_path = Path(filename)
        path = (capture_dir / relative_path).resolve()
        if relative_path.is_absolute() or not path.is_relative_to(capture_dir.resolve()):
            good = False
            results.append(filename + ': unsafe path in CHECKSUM; skipped.')
            continue
        if not path.is_file():
            good = False
            results.append(filename + ': missing. Please download it again.')
            continue

        digest = hashlib.md5(usedforsecurity=False)
        with path.open('rb') as downloaded_file:
            for chunk in iter(lambda: downloaded_file.read(1024 * 1024), b''):
                digest.update(chunk)
        md5sum_yours = digest.hexdigest()
        
        if md5sum == md5sum_yours:
            results.append(filename + ': md5sum is correct.')
        else:
            good = False
            results.append(filename + ': md5sum is wrong. Please download it again.')

    if good:
        print('All of downloaded files are verified.')
    else:
        print('Some of downloaded files are not verified.')

    result_path = 'download_verify_results.txt'
    with (capture_dir / result_path).open('w') as f:
        for result in results:
            f.write(result + '\n')
    print('The verification results are saved in ' + str(capture_dir / result_path))
    return good


def main():
    statuses = [verify_capture(capture_id) for capture_id in capture_id_list]
    return 0 if all(statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
