# Rubik's Cube env for PufferLib

A 3x3x3 Rubik's Cube environment for PufferLib 5.0. A 93K-parameter policy
trained with PPO learns to solve random scrambles in about two minutes on one
GPU, with no search: one network call per move.

## Install

Copy the two files into a PufferLib checkout (branch `5.0`):

```
cp ocean/rubiks/rubiks.h  <PufferLib>/ocean/rubiks/rubiks.h
cp config/rubiks.ini      <PufferLib>/config/rubiks.ini
```

Nothing else in PufferLib needs to change.

## Run

From the PufferLib folder:

```
./build.sh rubiks          # builds ./puffer
./puffer train             # 2B steps
./puffer eval latest       # watch it solve
```

For a different seed: `./puffer train --base.seed=1 --env.seed=1`
(`base.seed` is the network and move sampling, `env.seed` is which cubes are dealt).

## Reward

```
home = how many of the 20 pieces are finished (right place, right way round)

each move:        (home after - home before) / 20
the solving move: plus 5 * (200 - moves used) / 200
```

Every episode is a uniformly random scramble. It ends when the cube is solved
or after 200 moves. A half turn counts as one move (18 actions).

## What to expect

On an RTX 4070, the default run takes about 2 minutes at 15M steps/s. The solve
rate stays at zero for the first 250M steps or so, then climbs to about 100%
within the next 100M. A finished run solves about 99.98% of random scrambles in
about 54 moves on average.

The numbers on the final dashboard are a rough guide: that evaluation stops
early and can miss failed episodes. For a clean count, build the CPU evaluator
and run fresh cubes to completion:

```
./build.sh rubiks --cpu
./rubiks checkpoints/rubiks/<run>/<checkpoint>.bin --headless --base.eval_episodes=10000 --env.seed=12345
```

`perf` is the share solved and `steps / n` is the average moves.

## Settings that matter

The discount and the move limit decide whether it learns at all. With
`--train.gamma=0.9` it solves nothing, and with `--env.max_steps=100` it solves
about 17%. The defaults are a discount of 0.995 and a 200-move limit.
