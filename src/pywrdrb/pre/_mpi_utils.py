"""
MPI helpers shared by pywrdrb ensemble preprocessors.

Importing this module sets HDF5_USE_FILE_LOCKING=FALSE, which prevents deadlocks
when many MPI ranks open the same file read-only concurrently on Lustre/GPFS.
h5py checks this env var at each File() open, so setting it before the first
open (which happens at module import time) is sufficient.
"""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Tag reserved for ensemble-prediction gather operations. Choosing an
# explicit non-zero tag avoids collisions with other comm operations on the
# same communicator.
TAG_GATHER_PREDICTIONS = 7001


def bcast_with_error(comm, rank, payload_fn):
    """Rank-0 runs payload_fn() and broadcasts the result to every rank.

    Uses a sentinel envelope so that a rank-0 exception causes every rank to
    raise RuntimeError rather than hanging indefinitely in bcast.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
    rank : int  Current rank.
    payload_fn : callable
        Zero-argument callable executed only on rank 0.

    Returns
    -------
    The return value of payload_fn(), available on every rank.

    Raises
    ------
    RuntimeError
        On every rank if payload_fn() raises on rank 0.
    """
    if rank == 0:
        try:
            envelope = {"err": None, "value": payload_fn()}
        except Exception as exc:
            envelope = {"err": f"{type(exc).__name__}: {exc}", "value": None}
    else:
        envelope = None

    envelope = comm.bcast(envelope, root=0)

    if envelope["err"] is not None:
        raise RuntimeError(f"rank-0 bcast producer failed: {envelope['err']}")
    return envelope["value"]


def point_to_point_gather(comm, rank, size, local_dict, tag=TAG_GATHER_PREDICTIONS):
    """Scalable replacement for comm.gather(local_dict, root=0).

    mpi4py's pickle-based collective gather constructs a single recvbuf whose
    size equals the sum of all ranks' pickled payloads. At high rank counts with
    large DataFrames this aggregate buffer exceeds INT_MAX, causing MPI_ERR_ARG.
    This function uses pairwise send/recv so each message is one rank's payload,
    keeping peak buffer size bounded to the largest single-rank contribution.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
    rank : int   Current rank.
    size : int   Total number of ranks.
    local_dict : dict
        This rank's contribution to the gathered result.
    tag : int
        MPI message tag.

    Returns
    -------
    dict
        Merged dict on rank 0; None on all other ranks.
    """
    if rank == 0:
        merged = dict(local_dict)
        for source in range(1, size):
            chunk = comm.recv(source=source, tag=tag)
            merged.update(chunk)
        comm.Barrier()
        return merged
    else:
        comm.send(local_dict, dest=0, tag=tag)
        comm.Barrier()
        return None
