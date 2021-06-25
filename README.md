# Drake. Beta testing branch

PLEASE DO NOT DISTRIBUTE.

This branch contains the novel contact solver developed by Alejandro Castro as
the result of his research efforts. The purpose of this branch is to allow beta
testing users to try out this new solver in their applications while Alejandro
works on an open Journal publication and on the submission of this code as part
of Drake master.

Please do not share this branch with others. Contact Alejandro Castro to gain
access to this repository.

## How to use the new solver in your application.

Please refer to section `Experimental contact solver` of the `Clutter` demo in
[drake/examples/multibody/mp_convex_solver/README.md](https://github.com/amcastro-tri/drake-experimental/blob/primal_sparse_v4_simplified2/examples/multibody/mp_convex_solver/README.md).
You will find that the set of changes you need to run with the new solver is
very minimal. 

The clutter demo also shows how to set contact parameters and even custom
contact solver parameters, but unless you are Alejandro, you won't need to do
this yourself.

## Drake version

This entire experimental branch consists on a single commit on top of Drake
master (more specifically on top of `f72aecf2b` from June 24 2021.)
Therefore this commit should apply without conflicts unless you are too behind
Drake's `f72aecf2b`. If you have a newer version of Drake that conflicts with
this branch, please contact Alejandro Castro to update the experimental branch
to the latest Drake.
