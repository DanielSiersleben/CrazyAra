# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 22:16:11 2021

@author: Daniel
"""
import os
from subprocess import Popen, check_output, check_call, PIPE, call, STDOUT
import re
import nni
import logging

from tournament_config import tournament_config

logger = logging.getLogger()
logging.basicConfig()


def run_tournament(input_dict):
    contender_arg = ["-engine", tournament_config["con_name"], tournament_config["con_cmd"],
                     tournament_config["con_dir"]]
    contender_arg.extend(tournament_config["con_model_dir"])
    baseline_arg = ["-engine", tournament_config["base_name"], tournament_config["base_cmd"],
                    tournament_config["base_dir"]]
    baseline_arg.extend(tournament_config["base_model_dir"])
    each_arg = ["-each", "option.Context=gpu", tournament_config["proto"], "tc=50/10", tournament_config["batch_size"],
                tournament_config["threads"], tournament_config["move_overhead"], tournament_config["mcts_solver"],
                tournament_config["multipv"]]
    for key in input_dict.keys():
        if key in mpv_options:
            if tournament_config["mpv_engine"]:
                if tournament_config["mpv_engine"] == "both":
                    each_arg.append(input_dict[key])
                if tournament_config["mpv_engine"] == "c":
                    contender_arg.append(input_dict[key])
                if tournament_config["mpv_engine"] == "b":
                    baseline_arg.append(input_dict[key])
        else:
            each_arg.append(input_dict[key])
    etc_arg = ["-games", tournament_config["games"], "-rounds", tournament_config["rounds"], "-variant",
               tournament_config["variant"], "-openings", tournament_config["file"], tournament_config["file_format"],
               tournament_config["order"], "-epdout", tournament_config["epd_out"], "-pgnout", tournament_config["pgn_out"]]
    arguments = [tournament_config["cli_dir"]] + contender_arg + baseline_arg + each_arg + etc_arg

    process = Popen(arguments, stdout=PIPE, stderr=STDOUT, shell=True, text=True)
    ccout, ccerr = process.communicate()

    if not os.path.isfile(tournament_config["save_file_path"]):
        with open(tournament_config["save_file_path"], 'w') as out_file:
            out_file.write("CuteChess output and error log \n\n")

    with open(tournament_config["save_file_path"], "a") as save_file:
        if ccout:
            save_file.write("out:\n")
            save_file.writelines(ccout)
        if ccerr:
            save_file.write("err:\n")
            save_file.writelines(ccerr)
        save_file.write("\n")

    logger.info(ccout)
    logger.info(ccerr)
    output = re.findall(r"\s[-]*(?:\d+[.]\d|inf)\s\+/\-\s[-]*(?:\d+[.]\d|nan)", ccout)
    logger.info(output)
    if not output:
        raise Exception("invalid cute chess output")
    else:
        output = output[0].split(r" +/- ")

    elo = float(output[0])
    var = float(output[1])

    metric = {'elo': elo, 'variance': var}
    nni.report_final_result(metric)
    logger.info("final result is %f", elo)


mpv_options = [
    "LargeNetThreshold",
    "LargeNetPolicyWeight",
    "LargeNetStartPhase",
    "Expected_Strength_disparity"
]


def param_to_cli_format(params):
    output_dict = {}
    for k, v in params.items():
        if isinstance(v, str):
            output_dict.update({k: r"option." + k + "=%s" % v})
        else:
            if isinstance(v, bool):
                output_dict.update({k: r"option." + k + "=%s" % bool(v)})
            output_dict.update({k: r"option." + k + "=%i" % int(v)})
    return output_dict


if __name__ == '__main__':
    try:
        RECEIVED_PARAMS = nni.get_next_parameter()

        # RECEIVED_PARAMS = {'largeNetThreshold': 100.0, 'Use_Transposition_Table': 'true', 'Reuse_Tree': 'true', 'Fixed_Movetime': 100.0}

        logger.info(RECEIVED_PARAMS)
        input_params = param_to_cli_format(RECEIVED_PARAMS)
        logger.info(input_params)
        run_tournament(input_params)
    except Exception as exception:
        logger.exception(exception)
        raise
