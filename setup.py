#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np


setup(
        name = 'multiscale_centrality',
        version = '1.0',
        author='Alexis Arnaudon, Robert Peach',
        license = 'GNU General Public License v3.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        packages=['.'],
      )
