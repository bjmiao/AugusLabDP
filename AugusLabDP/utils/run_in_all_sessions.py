"""Run a user function across every session listed in session_info.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

try:
    from .readout_utils import get_all_probe_mapping, load_dataset
except ImportError:
    from readout_utils import get_all_probe_mapping, load_dataset

SessionFn = Callable[..., Any]


def run_in_all_sessions(
    fn: SessionFn,
    data_folder: Union[str, Path],
    *,
    datasets: Optional[List[str]] = None,
    session_info_path: Optional[Union[str, Path]] = None,
    dataset_filter: Optional[Iterable[str]] = None,
    session_predicate: Optional[Callable[[pd.Series], bool]] = None,
    probe: Union[str, List[str]] = "all",
    need_modules: Optional[List[str]] = None,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load each session from ``session_info.csv`` and call ``fn`` with the loaded results.

    This mirrors the loop used in scripts such as ``AP/psth_by_region.py``: for each row,
    probe mapping is taken from ``{dataset}_session_mapping.csv``, then
    :func:`readout_utils.load_dataset` is called.

    Parameters
    ----------
    fn
        Callable invoked as ``fn(results, session_name, session_type, dataset, row)`` where:

        - ``results`` — dict returned by :func:`readout_utils.load_dataset`
        - ``session_name`` — ``row['session']``
        - ``session_type`` — ``row['type']``
        - ``dataset`` — ``row['dataset']`` (subfolder under ``data_folder``)
        - ``row`` — the full ``pandas.Series`` for that session

    data_folder
        Root cache folder (contains ``session_info.csv`` and dataset subfolders).
    datasets
        Passed to :func:`readout_utils.get_all_probe_mapping`. Default: ``['ketamine','iso','syncope']``.
    session_info_path
        Override path to the session table. Default: ``data_folder / 'session_info.csv'``.
    dataset_filter
        If set, only rows whose ``dataset`` value is in this collection are run.
    session_predicate
        If set, only rows for which ``session_predicate(row)`` is True are processed.
        Other rows are skipped without loading data (nothing is appended to the returned list).
    probe, need_modules
        Forwarded to :func:`readout_utils.load_dataset`.
    continue_on_error
        If True, log failures and continue; if False, re-raise after the first error.
    verbose
        If True, print session name when starting each session and print errors.

    Returns
    -------
    list of dict
        One entry per attempted session with keys:
        ``session``, ``dataset``, ``session_type``, ``success``, ``result``, ``error`` (``error`` is
        None on success).
    """
    data_folder = Path(data_folder)
    if session_info_path is None:
        session_info_path = data_folder / "session_info.csv"
    else:
        session_info_path = Path(session_info_path)

    df_session_info = pd.read_csv(session_info_path)
    if dataset_filter is not None:
        allowed = set(dataset_filter)
        df_session_info = df_session_info[df_session_info["dataset"].isin(allowed)]

    if datasets is None:
        datasets = ["ketamine", "iso", "syncope"]
    df_probe_mapping = get_all_probe_mapping(str(data_folder), datasets=datasets)

    outcomes: List[Dict[str, Any]] = []

    for i in range(len(df_session_info)):
        item = df_session_info.iloc[i]
        session_name = item["session"]
        session_type = item["type"]
        dataset = item["dataset"]

        if session_predicate is not None and not session_predicate(item):
            if verbose:
                print(f"{session_name} skipped (session_predicate)")
            continue

        if verbose:
            print(session_name)

        try:
            df = df_probe_mapping[dataset]
            df = df[df.session == session_name]
            probe_mapping = {
                probe_name: (probenum, probe_depth)
                for probe_name, probenum, probe_depth in zip(
                    df["probe"], df["probenum"], df["probe_depth"]
                )
            }
            results = load_dataset(
                str(data_folder / dataset),
                session_name,
                session_type,
                probe=probe,
                probe_mapping=probe_mapping,
                need_modules=need_modules,
            )
            out = fn(results, session_name, session_type, dataset, item)
            outcomes.append(
                {
                    "session": session_name,
                    "dataset": dataset,
                    "session_type": session_type,
                    "success": True,
                    "result": out,
                    "error": None,
                }
            )
        except Exception as e:
            if verbose:
                print(e)
                print(f"Session {session_name} failed")
            outcomes.append(
                {
                    "session": session_name,
                    "dataset": dataset,
                    "session_type": session_type,
                    "success": False,
                    "result": None,
                    "error": e,
                }
            )
            if not continue_on_error:
                raise

    return outcomes
