import logging
import os
import sys


class _BriefFilter(logging.Filter):
    """Keep the terminal to the lines meant for a person.

    A training run logs a great deal that is worth keeping and not worth
    watching: the sampler's per-class window table, which weights loaded, the
    torch and CUDA versions. All of it still goes to the run's log file; this
    filter decides what also reaches the terminal, and the answer is: anything
    a call site marked `extra={"brief": True}`, plus every warning and error.
    """

    def filter(self, record):
        return getattr(record, "brief", False) or record.levelno >= logging.WARNING


# The marker itself, so call sites read as `logger.info(msg, extra=BRIEF)`.
BRIEF = {"brief": True}


def setup_logger(name, save_dir, distributed_rank=0, filename="log.json", brief=False):
    """A logger writing to stdout and, if `save_dir` is given, to a file.

    `brief=True` puts the terminal on a diet: only records marked with `BRIEF`
    (and warnings) are printed, without the timestamp/level prefix, while the
    file keeps everything. `TRACE_VERBOSE=1` turns the diet off, which is the
    switch to reach for when a run behaves strangely.
    """
    from vtrace.verbosity import verbose

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    if brief and not verbose():
        ch.addFilter(_BriefFilter())
        ch.setFormatter(logging.Formatter("%(message)s"))
    else:
        ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
