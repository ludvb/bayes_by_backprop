#!/usr/bin/env python3

""" refresh_version.py

writes src/__version__ to correspond to the current git version
"""

import re

import os.path as osp

import subprocess as sp


def _output_from(cmd):
    return sp.check_output(cmd.split(' '))


def _txt_from(cmd):
    return _output_from(cmd).decode().strip()


try:
    VERSION = _txt_from('git describe --dirty')
    # Modify according to PEP440
    VERSION = (
        re.compile(r'([^\-]+?)-([^\-]+?)-(.+?)')
        .sub(r'\g<1>.post\g<2>+\g<3>', VERSION)
        .replace('-', '.')
    )
except sp.CalledProcessError:
    try:
        VERSION = _txt_from('git describe --dirty --always')
        VERSION = '0.0.0+untagged.{}'.format(VERSION.replace('-', '.'))
    except sp.CalledProcessError:
        VERSION = '0.0.0+git.error'
except FileNotFoundError:
    VERSION = '0.0.0+no.git'

try:
    diff = _output_from('git diff')
except (FileNotFoundError, sp.CalledProcessError):
    diff = b''

with open(
    osp.join(
        osp.dirname(__file__),
        'protonet',
        '__version__.py',
    ),
    'wb',
) as f:
    f.write(b''.join(map(lambda x: x + b'\n', [
        b'""" Generated automatically---don\'t edit!',
        b'"""',
        b'',
        f'__version__ = \'{VERSION:s}\''.encode(),
        b'__diff__ = b"""' + diff.replace(b'"', b'\\"') + b'"""'
    ])))
