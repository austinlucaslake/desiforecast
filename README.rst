============
desiforecast
============

Links to Automation (Work in Progress)
======================================

Full Documentation
------------------

Please visit `desiforecast on Read the Docs`_

.. image:: https://readthedocs.org/projects/desiforecast/badge/?version=latest
    :target: http://desiforecast.readthedocs.org/en/latest/
    :alt: Documentation Status

.. _`desiforecast on Read the Docs`: http://desiforecast.readthedocs.org/en/latest/

Travis Build Status
-------------------

.. image:: https://img.shields.io/travis/desihub/desiforecast.svg
    :target: https://travis-ci.org/desihub/desiforecast
    :alt: Travis Build Status


Test Coverage Status
--------------------

.. image:: https://coveralls.io/repos/desihub/desiforecast/badge.svg?service=github
    :target: https://coveralls.io/github/desihub/desiforecast
    :alt: Test Coverage Status

Introduction
============
This package contains notebooks and script that apply machine learning to the DESI telemetry data to prediction in facilitate temperature predictions.


Installation (Work in Progress)
===============================

1. This package can be installed (and uninstalled) using the following command(s)::

    python setup.py develop --prefix=$INSTALL_DIR.
    python setup.py develop --prefix=$INSTALL_DIR --uninstall.

2. Fixed versions of the code can be installed as follows::
    
    python setup.py install --prefix=$INSTALL_DIR.

3. Installing using pip::

    pip install git+https://github.com/desihub/desiforecast.git@1.1.0

Running the Scripts
===================

Data Visulization Notebooks (Work in Progress)
----------------------------------------------
Data visualization can be found in the folder ``doc/nb``. Below is a list the the currently available notebooks::
    
    time_plots.ipynb
    correlation_plots.ipynb

Executable Scripts (Work in Progress)
-------------------------------------
Python scripts can be found in the folder ``py/desiforecast``. Below is a list of the currently available scripts::

    query_database.py
    neural_network.py

License
=======

desiforecast is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.
