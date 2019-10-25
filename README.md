# Adaptive Prior Selection for Repertoire-based Adaptation in Robotics

This repository has the python implmentation of the "object pushing" experiment and "hexapod damage recovery" experiment for the paper *[Adaptive Prior Selection for Repertoire-based Adaptation in Robotics](https://arxiv.org/abs/1907.07029)*.

Watch the video [here](https://www.youtube.com/watch?v=sbhW2rdIxA0&feature=youtu.be)

**Abstract :** *Repertoire-based learning is a data-efficient adaptation approach based on a two-step process in which (1) a large and diverse set of policies is learned in simulation, and (2) a planning or learning algorithm chooses the most appropriate policies according to the current situation (e.g., a damaged robot, a new object, etc.). In this paper, we relax the assumption of previous works that a single repertoire is enough for adaptation. Instead, we generate repertoires for many different situations (e.g., with a missing leg, on different floors, etc.) and let our algorithm selects the most useful prior. Our main contribution is an algorithm, APROL (Adaptive Prior selection for Repertoire-based Online Learning) to plan the next action by incorporating these priors when the robot has no information about the current situation. We evaluate APROL on two simulated tasks: (1) pushing unknown objects of various shapes and sizes with a robotic arm and (2) a goal reaching task with a damaged hexapod robot. We compare with "Reset-free Trial and Error" (RTE) and various single repertoire-based baselines. The results show that APROL solves both the tasks in less interaction time than the baselines. Additionally, we demonstrate APROL on a real, damaged hexapod that quickly learns to pick compensatory policies to reach a goal by avoiding obstacles in the path.*

* Following python libraries bust be installed to run the experiments:

    * pybullet
    * gpy
    * numpy
    * pathlib

* Also, python3 is required to run the experiment.
* All experiments must be run from the base directory 

### Object pushing experiment with kuka:

* Generating the policy repertoires using MAP Elites:

    * Run: ```python kuka_pushing_exps/map_elites_kuka_pushing.py --toy 5```

    * It will start saving the intermediate repertoires after every 100 generations in the same directory. It should take a few hours to reach the maximum number of evaluations. Using the '--toy' the repertoires can be generated for different toys.
--toy can take any integer value between 0 to 13.

    * Some pre-generated repertoires are provided in the data directory.

* Running the experiments

```python kuka_pushing_exps/kukaPushing_astar_ctlr2cartesian_v2.py --toy 0 --ucb_const 0.5 --kernel_var 0.003 --kernel_l 0.03 --visualization_speed 5.0 --search_size 800  --objectEulerAngles -1 --gui```

### Hexapod damage recovery and goal reaching:

* Generating the policy repertoires using MAP Elites:

   * Run: ```python hexapod_experiments/map_elites_hexapod_cartesian.py --lateral_friction 1.0 --blocked_legs 1 3```

   * Where --lateral_friction is the floor friction and --blocked_legs specifies which legs are to be blocked. --blocked_legs can take a list of space separated integers between 0-5. It will start saving the intermediate repertoires after every 100 generations in the same directory. It should take a few hours to reach the maximum number of evaluations.

   * Some pregenerated repertoires are provided in the data directory.

* Running the experiments

```python hexapod_experiments/hexapod_astar_ctlr2cartesian_v2_Arena.py  --kernel_var 0.03 --kernel_l 0.03 --search_size 100 --gui --blocked_legs 0 --visualization_speed 2.0 --lateral_friction 0.8```
