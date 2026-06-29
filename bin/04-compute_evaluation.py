#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import logging
import argparse
import traceback
import ir_measures
import configparser
import pandas as pd
from datetime import datetime
from ir_measures import MRR, nDCG, AP
from typing import List, Dict, Optional

logger = None


def get_logger(config):
    global logger

    if logger is not None:
        return logger

    logs_dir = os.path.join(config["general"]["project_root"], config["logging"]["dir"])
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(
        logs_dir,
        f"{config['logging']['preffix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print("Log file:", log_filename)
    return logger


def get_runs_files(runs_dir, run_preffix):
    run_file_lists = []
    summary_file = None
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.startswith(run_preffix) and file.endswith(".csv"):
                if "summary" in file:
                    if not file.endswith("new.csv"):
                        summary_file = os.path.join(root, file)
                else:
                    run_file_lists.append(os.path.join(root, file))
    return run_file_lists, summary_file


def load_summary_file(summary_file):
    return pd.read_csv(summary_file)


def split_line(line: str, sep=",") -> List[str]:
    global logger

    stripped_line = line.strip()
    try:
        score, doc_id, query_id = stripped_line.split(sep)
    except ValueError:
        logger.debug(f"[DEBUG]: Error parsing line: {stripped_line}")
        logger.debug(f"[DEBUG]: Trying to interpret the line")
        score, rest_of_line = stripped_line.split(sep, maxsplit=1)
        doc_id, query_id = rest_of_line.rsplit(sep, maxsplit=1)
        logger.debug(f"[DEBUG]: Success")
    return score, doc_id, query_id


def trec_run_translator(filename, max_score=100):
    """
    Builds a generator to read run files and return them in TREC format,
    understandable by the ir_measures library.

    Args:
        filename: path to the run file
        max_score: maximum score value, the most relevant result will get this score, the rest of the results will have a decreasing score
    Returns:
        A generator that yields ir_measures.ScoredDoc objects for each line in the run file
    """
    header_line = True
    with open(filename, "r") as f:
        for line in f:
            if header_line:
                # Primer linea de header, la ignoramos
                header_line = False
                continue
            score, doc_id, query_id = split_line(line)
            score = max_score - int(score) + 1
            #  ["query_id", "ignored", "doc_id", "rank", "score", "runid"]
            #  [ l[2]     ,  0       ,  l[1]   ,  0    ,  l[0]  , "runid"]
            yield ir_measures.ScoredDoc(
                query_id=query_id, doc_id=doc_id, score=int(score)
            )


def get_porcentaje_from_filename(run_file):
    """
    Parse run filepath to extract pp value.

    Args:
        filepath (str): Path to the run file

    Returns:
        int: pp value
    """
    # Extract the filename from the path
    filename = run_file.split("/")[-1]

    # Pattern to match the components
    pattern = r".*\.(\d+)\.csv"

    match = re.match(pattern, filename)
    if match:
        pp = int(match.group(1))
        return pp

    return None


def load_qrels(qrels_file, qrels_format="trec"):
    global logger
    # print(f"qrels format: {qrels_format}")
    logger.debug(f"qrels format: {qrels_format}")
    if qrels_format == "dbpedia":
        # return ir_measures.read_trec_qrels(qrels_file)
        df = pd.read_csv(
            qrels_file,
            sep="\t",
        )
        qrels = df.rename(
            columns={
                "query-id": "query_id",
                "corpus-id": "doc_id",
                "score": "relevance",
            }
        )
        return qrels
    elif qrels_format == "trec":
        return ir_measures.read_trec_qrels(qrels_file)
    else:
        raise ValueError(f"Unknown qrels format: {qrels_format}")


def compute_new_metrics(df_summary, run_file_lists, qrels_file, config_experiment):
    global logger

    df_summary_new = df_summary.copy()
    results = {}
    qrels_format = config_experiment.get("qrels_format", "trec")

    for run_file in run_file_lists:
        pp = get_porcentaje_from_filename(run_file)
        logger.info(
            f"run name: {config_experiment["name"]} :: features percentage {pp}"
        )
        run = trec_run_translator(run_file)
        qrels = load_qrels(qrels_file, qrels_format=qrels_format)
        try:
            results = ir_measures.calc_aggregate(
                [
                    MRR @ [10, 100],
                    nDCG @ [10, 100],
                    AP @ [10, 100],
                ],
                qrels,
                run,
            )
        except ValueError:
            logger.error(f"Error computing metrics for {run_file}")
            logger.error(traceback.format_exc())
            logger.error(
                f"Verify that the experiment {config_experiment['name']} has the qrels_format parameter configured correctly"
            )
            logger.error(f"qrels_format value: {qrels_format}")
            continue
        for measure_name, measure_value in results.items():
            df_summary_new.loc[
                df_summary_new["per_features"] == pp, f"{measure_name} (ir)"
            ] = measure_value
    return df_summary_new


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for experiments in the config file"
    )
    parser.add_argument("--config", type=str, required=True, help="INI Config file")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output", default=False
    )

    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    config = configparser.ConfigParser()
    config.read(args.config)
    config.set("general", "project_root", project_root)

    get_logger(config)

    experiments = [
        x.strip() for x in config["general"]["enabled_experiments"].split(",")
    ]
    logger.info(f"Active experiments: {experiments}")

    for experiment in experiments:
        logger.info(f"Starting to process Experiment: {experiment}")
        experiment_key = f"experiments.{experiment}"
        try:
            qrels_file = os.path.join(
                config["general"]["project_root"],
                config[experiment_key]["qrels_file"],
            )
            assert os.path.exists(qrels_file), f"Qrels file {qrels_file} does not exist"
        except KeyError:
            logger.error(
                f"Experiment {experiment} does not exist or is not configured correctly"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error to load the qrels file: {str(e)}")
            sys.exit(1)

        try:

            ############################################
            # Load experiment run files and summary file
            ############################################
            run_file_lists, summary_file = get_runs_files(
                config[experiment_key]["runs_dir"], config[experiment_key]["preffix"]
            )
            assert (
                summary_file is not None
            ), f"Summary file not found, please check the prefix for experiment {experiment_key}"
            assert (
                len(run_file_lists) > 0
            ), f"No run files found, please check the prefix for experiment {experiment_key}"
            logger.info(f"Runs files: {len(run_file_lists)}")
            logger.info(f"Summary file: {summary_file}")

            #######################
            # Load original summary
            #######################
            logger.debug(f"Loading summary file: {summary_file}")
            df_summary = load_summary_file(summary_file)
            assert (
                len(df_summary) > 0
            ), f"Summary file is empty, please check the path for experiment {experiment_key}"

            #####################
            # Compute new metrics
            #####################
            logger.info("Computing new metrics")
            df_summary_new = compute_new_metrics(
                df_summary, run_file_lists, qrels_file, config[experiment_key]
            )
            logger.debug("Columns in the original DF: %s", df_summary.columns)
            logger.debug("Columns in the new DF: %s", df_summary_new.columns)
            assert len(df_summary_new.columns) > len(
                df_summary.columns
            ), f"No new results were added, please check the compute_new_metrics function"

            ########################################
            # Save new summary file with new metrics
            ########################################
            new_summary_file = summary_file.replace(".csv", ".new.csv")
            df_summary_new.to_csv(new_summary_file, index=False)
            logger.info(f"New metrics saved to file: {new_summary_file}")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.error(traceback.format_exc())
            return 1

        logger.info(f"Experiment end: {experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
