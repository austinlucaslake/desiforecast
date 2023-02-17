# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desiforecast.test.desiforecast_test_suite
=========================================

Used to initialize the unit test framework via ``python setup.py test``.
"""
#
from __future__ import absolute_import, division, print_function, unicode_literals
#
import unittest
#
#- This is factored out separately from runtests() so that it can be used by
#- python setup.py test
def desiforecast_test_suite():
    """Returns unittest.TestSuite of desiforecast tests"""
    from os.path import dirname
    desiforecast_dir = dirname(dirname(__file__))
    # print(desiforecast_dir)
    return unittest.defaultTestLoader.discover(desiforecast_dir,
        top_level_dir=dirname(desiforecast_dir))

def runtests():
    """Run all tests in desiforecast.test.test_*.
    """
    #- Load all TestCase classes from desiforecast/test/test_*.py
    tests = desiforecast_test_suite()
    #- Run them
    unittest.TextTestRunner(verbosity=2).run(tests)

if __name__ == "__main__":
    runtests()
