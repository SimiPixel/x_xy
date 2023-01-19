<p align="center">
<img src="figures/icon.svg" height="200" />
</p>

# **Ti**ny **Ki**nematic **Tr**ee Simulator (**TiKiTr**)

The **Ti**ny **Ki**nematic **Tr**ee Simulator (TiKiTr) allows to perform 
- forward dynamics
- inverse dynamics
- forward kinematics
- inverse kinematics

on a general Kinematic Tree structure. 

It uses nothing but JAX (and flax.struct).

It's meant to be minimalistic and simple. It uses spatial vectors and implements algorithms as proposed by Roy Featherstone. Nameing is heavily inspired by `brax`.

It currently does *not* support
- collisions (i.e. every body is (sort of) transparent)

and probably won't in the near future.
