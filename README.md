# Adaptive Prior Selection for Repertoire-based Adaptation in Robotics

This repository has the python implmentation of the "object pushing" experiment and "hexapod damage recovery" experiment for the paper *Adaptive Prior Selection for Repertoire-based Adaptation in Robotics*.

* Following python libraries bust be installed to run the experiments:

    * pybullet
    * gpy
    * numpy
    * pathlib

* Also, python3 is required to run the experiment.
* All experiments must be run from the base directory 

## Object pushing experiment with kuka:

* Generating the policy repertoires using MAP Elites:

    * Run: python kuka_pushing_exps/map_elites_kuka_pushing.py --toy 5

    * It will start saving the intermediate repertoires after every 100 generations in the same directory. It should take a few hours to reach the maximum number of evaluations. Using the '--toy' the repertoires can be generated for different toys.
--toy can take any integer value between 0 to 13.

    * Some pre-generated repertoires are provided in the data directory.

* Running the experiments

```python kuka_pushing_exps/kukaPushing_astar_ctlr2cartesian_v2.py --toy 0 --ucb_const 0.5 --kernel_var 0.003 --kernel_l 0.03 --visualization_speed 5.0 --search_size 800  --objectEulerAngles -1 --gui```
