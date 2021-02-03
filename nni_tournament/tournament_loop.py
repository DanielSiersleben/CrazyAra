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

cli_dir = r"D:\Program Files (x86)\Cute Chess\cutechess-cli.exe"
# Baseline:
base_name = "name=Baseline"
base_cmd = "cmd=CrazyAra.exe"
base_dir = "dir=C:/Users/Daniel/MPVCrazyAra/CrazyAra/builds/buildSmallNet"

# Contender:
con_name = "name=Contender"
con_cmd = "cmd=CrazyAra.exe"
con_dir = r"dir=C:\Users\Daniel\MPVCrazyAra\CrazyAra\builds\buildLargeNet"

# each:
proto = "proto=uci"
batch_size = "option.Batch_Size=16"
threads = "option.Threads=2"
move_overhead = "option.Move_Overhead=0"
mcts_solver = "option.MCTS_Solver=true"
multipv = "option.MultiPV=1"

mpv_engine = "" ## "", c (contender), b (baseline), both

variant = "crazyhouse"
games = "500"
rounds = "2"
repeat = "2"

# openings
file = "file=D:/Program Files (x86)/Cute Chess/books-master/crazyhouse_mix_cp_130.epd"
file_format = "format=epd"
order = "order=random"

save_file_path = r"CuteChessOut.txt"

logger = logging.getLogger()
logging.basicConfig()


def run_tournament(input_dict):
    contender_arg = ["-engine", con_name, con_cmd, con_dir]
    baseline_arg = ["-engine", base_name, base_cmd, base_dir]
    each_arg = ["-each", "option.Context=gpu", proto, "tc=50/10", batch_size, threads, move_overhead, mcts_solver, multipv]
    for key in input_dict.keys():
        if key in mpv_options:
            if mpv_engine:
                if mpv_engine == "both":
                    each_arg.append(input_dict[key])
                if mpv_engine == "c":
                    contender_arg.append(input_dict[key])
                if mpv_engine == "b":
                    baseline_arg.append(input_dict[key])
        else:
            each_arg.append(input_dict[key])
    etc_arg = ["-games", games, "-rounds", rounds, "-variant", variant, "-openings", file, file_format, order]
    arguments = [cli_dir] + contender_arg + baseline_arg + each_arg + etc_arg

    process = Popen(arguments, stdout=PIPE, stderr=STDOUT, shell=True, text=True)
    ccout, ccerr = process.communicate()

    if not os.path.isfile(save_file_path):
        with open(save_file_path, 'w') as out_file:
            out_file.write("CuteChess output and error log \n\n")

    with open(save_file_path, "a") as save_file:
        if ccout:
            save_file.write("out:\n")
            save_file.writelines(ccout)
        if ccerr:
            save_file.write("err:\n")
            save_file.writelines(ccerr)
        save_file.write("\n")

    logger.info(ccout)
    logger.info(ccerr)
    output = re.findall("\s[-]*(?:\d+[.]\d|inf)\s\+/\-\s[-]*(?:\d+[.]\d|nan)", ccout)
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
    "largeNetThreshold",
    "sortPolicy",
    "startLarge",
    "largeQValWeight"
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

        #RECEIVED_PARAMS = {'largeNetThreshold': 100.0, 'Use_Transposition_Table': 'true', 'Reuse_Tree': 'true', 'Fixed_Movetime': 100.0}

        logger.info(RECEIVED_PARAMS)
        input_params = param_to_cli_format(RECEIVED_PARAMS)
        logger.info(input_params)
        run_tournament(input_params)
    except Exception as exception:
        logger.exception(exception)
        raise
