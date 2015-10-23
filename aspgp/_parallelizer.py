"""
Parallelizes the multi-start optimization procedure of GPy models.

Author:
    Ilias Bilionis

Date:
    4/15/2015

"""


import numpy as np
import GPy


__all__ = ['Parallelizer', 'ParallelizedGPRegression']


class Parallelizer(object):
    """
    Parallelize the ``optimize_restarts()`` function.
    """

    def optimize_restarts(self, num_restarts=10, parallel_verbose=True, comm=None, **kwargs):
        """
        Optimize restarts using MPI.

        :param comm:    The MPI communicator

        When we return, we guarantee that every core has the right model.
        """
        if comm is not None:
            from mpi4py import MPI as mpi
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1
        my_num_restarts = num_restarts / size
        num_restarts = my_num_restarts * size
        verbose = rank == 0 and parallel_verbose
        if verbose:
            print '+ optimizing hyper-parameters using multi-start'
            print '+ num available cores:', size
            print '+ num restarts:', num_restarts
            print '+ num restarts per core:', my_num_restarts
        # Let everybody work with its own data
        super(Parallelizer, self).optimize_restarts(num_restarts=my_num_restarts,
                                                    verbose=parallel_verbose,
                                                    **kwargs)
        if comm is not None:
            log_like = np.hstack(comm.allgather(np.array([self.log_likelihood()])))
            max_rank = np.argmax(log_like)
            if verbose:
                print '+ maximum likelihood found:', np.argmax(log_like)
                print '+ maximum likelihood found by core:', max_rank
            if rank == max_rank:
                x_opt = self.optimizer_array.copy()
            else:
                x_opt = None
            best_x_opt = comm.bcast(x_opt, root=max_rank)
            self.optimizer_array = best_x_opt
            if rank == 0:
                print str(self)


class ParallelizedGPRegression(Parallelizer, GPy.models.GPRegression):
    """
    A parallelized version of GPRegression.
    """
    pass
