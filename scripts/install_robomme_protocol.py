"""Install an isolated Python 3.11 protocol overlay from Kevin's local venv.

Preserves the source environments, including dependency licenses/dist-info.
Run after sourcing experiments/robomme/env.sh on a Python 3.11 host.
"""
import os
from pathlib import Path
import shutil
import sys


def main():
    if sys.version_info[:2] != (3,11):
        raise ValueError('This local binary overlay requires Python 3.11')
    source=Path('/mnt/vepfs02/output/kevin_wang/memorywam/robomme_native/.venv/lib/python3.11/site-packages')
    output=Path(os.environ['ROBOMME_PROTOCOL_PYTHON']).resolve()
    if source in output.parents or output==source:
        raise ValueError('Protocol output must be outside the source environment')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('websockets','websockets-15.0.1.dist-info','msgpack','msgpack-1.2.1.dist-info'):
        if not (output/name).exists():
            shutil.copytree(source/name,output/name,ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0,str(output))
    import websockets.asyncio.server
    import msgpack
    assert websockets.__version__=='15.0.1' and msgpack.__version__=='1.2.1'
    print('ROBOMME_PROTOCOL_OK',output)


if __name__=='__main__':main()
