#!/usr/bin/env python3

import os.path as osp

from subprocess import run

import setuptools as st


run(['env', 'python3', osp.join(osp.dirname(__file__), 'refresh_version.py')])

st.setup()
