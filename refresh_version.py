#!/usr/bin/env python3

""" refresh_version.py

writes src/__version__ to correspond to the current git version
"""

import re

import os.path as osp

import subprocess as sp


try:
    VERSION = sp.check_output(
        ['git', 'describe', '--dirty']).decode('utf8').strip()
    # Modify according to PEP440
    VERSION = (
        re.compile(r'([^\-]+?)-([^\-]+?)-(.+?)')
        .sub(r'\g<1>.post\g<2>+\g<3>', VERSION)
        .replace('-', '.')
    )
except sp.CalledProcessError:
    VERSION = '0.0.0+not.described'

with open(
    osp.join(
        osp.dirname(__file__),
        'protonet',
        '__version__.py',
    ),
    'w',
) as f:
    f.writelines(''.join(map(lambda x: x + '\n', [
        '""" Generated automatically---don\'t edit!',
        '"""',
        '',
        f'__version__ = \'{VERSION:s}\''
    ])))
