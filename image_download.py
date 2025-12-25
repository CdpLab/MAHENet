import argparse
import json
import gzip
import os
import cv2
import numpy as np
import shutil
import urllib.parse
import posixpath
import PIL.Image as Image
import requests

def main(args):
    data = json.loads(
        gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb').read()
    )
    print(f'Dataset path is {args.data_path}.')
    print(f'Dataset describes {len(data)} images.')
    print(f'Original images are in {args.originals_path}.')

    i = 1
    for datum in data:
        print(f'picture{i}')
        path = os.path.join(args.data_path, datum['image_path'])
        if os.path.exists(path):
            print('exit！')
            i=i+1
            continue
        url = datum['openimages_metadata']['OriginalURL']

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            relative_path = datum['image_path'].split('/')
            filename = relative_path[2]
            dir_path = os.path.join(args.data_path, relative_path[0], relative_path[1])
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, filename)
            print(f'save{filepath}')
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f'sucess: {filename}')
        except Exception as e:
            print(f'fail: {url}，err: {e}')
        i = i + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, default='', help=''
    )
    parser.add_argument(
        '--originals_path',
        type=str,
        default='',

        help='path to directory of original images',
    )
    args = parser.parse_args()
    main(args)
