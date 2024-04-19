import argparse
import io
import logging
import pickle
from datetime import datetime
from functools import partial
from tempfile import TemporaryFile

import numpy as np
from astropy.time import Time
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.model_observatory.kinem_model import KinemModel
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.sim_archive import drive_sim

from rubin_sim.data import get_baseline

DEFAULT_ARCHIVE_URI = "file:///sdf/data/rubin/user/neilsen/data/test_sim_archive/"

return_zero = np.zeros((1,)).item


class ScatteredKinemModel(KinemModel):
    def __init__(self, *args, visit_scatter_func=None, slew_scatter_func=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.visit_scatter_func = return_zero if visit_scatter_func is None else visit_scatter_func
        self.slew_scatter_func = return_zero if slew_scatter_func is None else slew_scatter_func

    def visit_time(self, *args, **kwargs):
        model_time = super().visit_time(*args, **kwargs)
        scattered_time = model_time + self.visit_scatter_func()
        return scattered_time

    def slew_times(self, *args, **kwargs):
        model_time = super().slew_times(*args, **kwargs)
        scattered_time = model_time + self.slew_scatter_func()
        return scattered_time


def _run_sim(
    mjd_start,
    archive_uri,
    scheduler_io,
    label,
    tags=tuple(),
    sim_length=2,
    visit_scatter_func=return_zero,
    slew_scatter_func=return_zero,
):
    logging.info(f"Running {label}.")

    kinem_model = ScatteredKinemModel(
        mjd0=mjd_start, visit_scatter_func=visit_scatter_func, slew_scatter_func=slew_scatter_func
    )
    observatory = ModelObservatory(
        mjd_start=mjd_start,
        cloud_data="ideal",
        seeing_data="ideal",
        downtimes="ideal",
        kinem_model=kinem_model,
    )

    scheduler_io.seek(0)
    scheduler = pickle.load(scheduler_io)

    drive_sim(
        observatory=observatory,
        scheduler=scheduler,
        archive_uri=archive_uri,
        label=label,
        tags=tags,
        script=__file__,
        mjd_start=mjd_start,
        survey_length=sim_length,
        record_rewards=True,
    )


def _create_scheduler_io(day_obs_mjd, scheduler=None, opsim_db=None):
    if scheduler is None:
        scheduler = example_scheduler()
    elif isinstance(scheduler, CoreScheduler):
        pass
    elif isinstance(scheduler, io.IOBase):
        scheduler.seek(0)
        scheduler = pickle.load(scheduler)
    else:
        with open(scheduler, "r") as sched_io:
            scheduler = pickle.load(sched_io)

    scheduler.keep_rewards = True

    if opsim_db is not None:
        last_preexisting_obs_mjd = day_obs_mjd + 0.5
        obs = SchemaConverter().opsim2obs(opsim_db)
        obs = obs[obs["mjd"] < last_preexisting_obs_mjd]
        if len(obs) > 0:
            scheduler.add_observations_array(obs)

    scheduler_io = TemporaryFile()
    pickle.dump(scheduler, scheduler_io)
    scheduler_io.seek(0)
    return scheduler_io


def run_prenights(day_obs_mjd, archive_uri, scheduler_file=None, opsim_db=None):
    exec_time = Time.now().iso
    mjd_start = day_obs_mjd + 0.5
    scheduler_io = _create_scheduler_io(day_obs_mjd, scheduler_file, opsim_db)

    # Assign args common to all sims for this execution.
    run_sim = partial(_run_sim, archive_uri=archive_uri, scheduler_io=scheduler_io, sim_length=2)

    # Begin with an ideal pure model sim
    run_sim(
        mjd_start,
        label=f"Nominal start and overhead, ideal conditions, run at {exec_time}",
        tags=["ideal", "nominal"],
    )

    # Run a few different scatters of visit time
    visit_scatter_loc = 0.0
    visit_scatter_scale = 10.0
    for visit_scatter_seed in range(101, 106):
        visit_scatter_func = partial(
            np.random.default_rng(visit_scatter_seed).normal, visit_scatter_loc, visit_scatter_scale
        )
        run_sim(
            mjd_start,
            label=f"Scattered visit time ({visit_scatter_loc, visit_scatter_scale, visit_scatter_seed}),"
            + f" Nominal start and slew, ideal conditions, run at {exec_time}",
            tags=["ideal", "scattered_visit_time"],
            visit_scatter_func=visit_scatter_func,
        )


def _parse_dayobs_to_mjd(dayobs):
    try:
        day_obs_mjd = Time(dayobs).mjd
    except ValueError:
        try:
            day_obs_mjd = Time(datetime.strptime(dayobs, "%Y%m%d"))
        except ValueError:
            day_obs_mjd = Time(dayobs, format="mjd")

    return day_obs_mjd


def prenight_sim_cli(*args):
    if Time.now() >= Time("2025-05-01"):
        default_time = Time(int(Time.now().mjd - 0.5), format="mjd").iso
    else:
        default_time = Time("2025-05-14")

    parser = argparse.ArgumentParser(description="Run prenight simulations")
    parser.add_argument(
        "--dayobs",
        type=str,
        default=default_time.iso,
        help="The day_obs of the night to simulate.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=DEFAULT_ARCHIVE_URI,
        help="Archive in which to store simulation results.",
    )
    parser.add_argument("--scheduler", type=str, default=None, help="pickle file of the scheduler to run.")
    parser.add_argument(
        "--opsim", type=str, default=get_baseline(), help="Opsim database from which to load previous visits."
    )
    args = parser.parse_args() if len(args) == 0 else parser.parse_args(args)

    day_obs_mjd = _parse_dayobs_to_mjd(args.dayobs)
    archive_uri = args.archive
    scheduler_file = args.scheduler
    opsim_db = args.opsim

    run_prenights(day_obs_mjd, archive_uri, scheduler_file, opsim_db)


if __name__ == "__main__":
    prenight_sim_cli()
