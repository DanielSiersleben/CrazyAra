tournament_config = {
    "cli_dir": r"D:\Program Files (x86)\Cute Chess\cutechess-cli.exe",
    # Baseline:
    "base_name": "name=Baseline",
    "base_cmd": "cmd=CrazyAra.exe",
    "base_model_dir": ["option.Model_Directory=model/standard_net"],
    "base_dir": "dir=C:/Users/Daniel/MPVCrazyAra/CrazyAra/builds/buildStandard",

    # Contender:
    "con_name": "name=Contender",
    "con_cmd": "cmd=CrazyAra.exe",
    "con_model_dir":  ["option.Small_Model_Directory=model/standard_net", "option.Large_Model_Directory=model/large_net"],
    "con_dir": r"dir=C:\Users\Daniel\MPVCrazyAra\CrazyAra\builds\buildMPV",

    # each:
    "proto": "proto=uci",
    "batch_size": "option.Batch_Size=16",
    "threads": "option.Threads=2",
    "move_overhead": "option.Move_Overhead=0",
    "mcts_solver": "option.MCTS_Solver=true",
    "multipv": "option.MultiPV=1",

    "mpv_engine": "c",  # "" (None), "c" (contender), "b" (baseline), "both" (both)

    "epd_out": r"D:\Program Files (x86)\Cute Chess\Tournaments\nni_epd.epd",
    "pgn_out": r"D:\Program Files (x86)\Cute Chess\Tournaments\nni_pgn.pgn",
    "variant": "crazyhouse",
    "games": "50",
    "rounds": "2",
    "repeat": "2",

    # openings
    "file": r"file=D:/Program Files (x86)/Cute Chess/books-master/crazyhouse_mix_cp_130.epd",
    "file_format": "format=epd",
    "order": "order=random",

    "save_file_path": r"CuteChessOut.txt"
}
