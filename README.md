# Welcome

Ophys-mfish-dev repo is a place for development of ophys/mfish data processing and analysis tools, quickly for adapting to the Code Ocean platform. Its main tool is COMB.

## COMB:

![COMB logo](/img/comb.png)

COMB: Compile Ophys Mfish Behavior

+ Replacement of the AllenSDK for visual behavior like experiments (e.g. changed detection task)
+ A set of classes to load and pre process data on ophys/behavior (later mfish) data on Code Ocean
+ Lazy loading of data products (main class loads quick, expensive processes are executed only when needed)
+ Less abstraction/dependencies/nesting than allenSDK
+ "experiment" is changed to "plane", so ophys_experiment_id is now ophys_plane_id

## Installation

Add these lines to a code ocean capusle docker file

"""
 RUN git clone https://github.com/AllenNeuralDynamics/ophys-mfish-dev \
    && cd ophys-mfish-dev \
    && pip install -e .
"""

## Contributing
+ Make a PR, tag a reviewer




