"""
@file: plane_policy_representation.py
Created on 18.04.19
@project: CrazyAra
@author: Moritz Willig and later updated by queensgambit

Functionality for representing all possible moves in the policy feature maps.
Note, most of the entries in the policy feature map are unusable because the represent illegal moves
which would go beyond the board.
Most of the functions are not 100% optimal in terms of performance, but there are only used to create a static look-up
table for retrieving the move probability from the policy feature maps, so this doesn't play a factor.

"""
import numpy as np
import chess
from DeepCrazyhouse.src.domain.crazyhouse.constants import BOARD_WIDTH, BOARD_HEIGHT, LABELS, P_MAP
from DeepCrazyhouse.src.domain.util import get_row_col


def get_movement_vector(move: chess.Move):
    """
    Returns a vector representation for a chess-move.
    The first coordinate describes the row offset and the second the column offset.
    :param move: Chess move object
    :return: Offset of the move [y-offset, x-offset]
    """
    from_row, from_col = get_row_col(move.from_square)
    to_row, to_col = get_row_col(move.to_square)
    return [to_row - from_row, to_col - from_col]


def get_plane_index_queen_move(movement_vector) -> int:
    """
    Returns the plane index for the given queen like move. Possible return values 0..55.
    These are returned as {N, NE, E, SE, S, SW, W, NW} in which the different move lengths are sorted after each
    direction. Bishop, rook, pawn, king and queen movements can be expressed like this.
    :param movement_vector: Movement offset vector as returned by get_movement_vector().
                            It's assumed to be a queen move.
    :return: Plane index
    """
    #
    if BOARD_WIDTH != BOARD_HEIGHT:
        raise Exception("This function only assumes a quadratic board.")

    # vector components range from -7 to +7
    x = movement_vector[1]
    y = movement_vector[0]

    # length: min 1 field max 7 fields -> correct to 0 based: 0..6
    length = max(abs(x), abs(y)) - 1

    # {N, NE, E, SE, S, SW, W, NW}
    # \|/    701
    # -Q- -> 6 2
    # /|\    543

    direction_cases = [
        (x == 0 and y > 0),  # 0: North
        (x > 0 and y > 0),  # 1: North-East
        (x > 0 and y == 0),  # 2: East
        (y < 0 < x),  # 3: South-East
        (x == 0 and y < 0),  # 4: South
        (x < 0 and y < 0),  # 5: South-West
        (x < 0 and y == 0),  # 6: West
        (x < 0 < y),  # 7: North-West
    ]

    direction = None
    # iterate until the first True case is fulfilled
    for idx, direction_case in enumerate(direction_cases):
        if direction_case:
            direction = idx
    if direction is None:
        raise Exception("Given Queenmove:", movement_vector, "is invalid")

    # fist all moves in a certain direction are used followed by different amount of lengths
    offset = direction * (BOARD_WIDTH - 1) + length
    return offset


def get_plane_index_knight_move(movement_vector) -> int:
    """
    Returns the plane index for the given knight move. Possible return values 0..55.
    :param movement_vector: Movement-vector as returned as get_movement_vector(). Expected to be a knight move
    :return: Plane index
    """

    #    7 0
    #   6   1
    #     N
    #   5   2
    #    4 3

    # movement cases: row / col
    mv_cases = [
        [+2, +1],  # case 0
        [+1, +2],  # case 1
        [-1, +2],  # case 2
        [-2, +1],  # case 3
        [-2, -1],  # case 4
        [-1, -2],  # case 5
        [+1, -2],  # case 6
        [+2, -1],  # case 7
    ]

    movement_offset = None
    for idx, mv_case in enumerate(mv_cases):
        if movement_vector == mv_case:
            movement_offset = idx
            break
    if movement_offset is None:
        raise Exception("Given Knightmove:", movement_vector, "is invalid")

    move_type_offset = 56
    board_offset = move_type_offset + movement_offset
    return board_offset


def get_plane_index_promotion_move(piece_type, movement_vector) -> int:
    """
    Returns the plane index for a promotion move (Queen promotion included).
    :param piece_type: Piece type integer value as described in python-chess
    :param movement_vector:  Movement-vector as returned as get_movement_vector(). Expected to be a promotion pawn move.
    :return: Plane index
    """

    # a pawn can only move forward or capture to the left or right
    # => we only have to inspect the x axis
    # python-chess starts counting at 1
    if type(piece_type) is str:
        piece_id = P_MAP[piece_type] % 6
    else:
        # python-chess starts counting at 1
        piece_id = (piece_type - 1) % 6
    # promotion
    # 1=knight, 2=bishop, 3=rook, 4=queen
    if 1 <= piece_id <= 4:
        move_type_offset = 64
        piece_offset = piece_id - 1  # map rook,knight,bishop -> 0..2
        movement_offset = movement_vector[1] + 1  # pawn move: left, forward, right -> 0..2
        board_offset = move_type_offset + (piece_offset * 3) + movement_offset
        return board_offset
    raise RuntimeError("invalid piece promotion")


def get_plane_index_drop_move(piece_type) -> int:
    """
    Returns the plane index for a drop move (ordering: pawn, knight, bishop, rook, queen).
    :param piece_type: Piece type integer value as described in python-chess
    :return: Plane index
    """

    # python-chess starts counting at 1
    if type(piece_type) is str:
        piece_id = P_MAP[piece_type] % 6
    else:
        # python-chess starts counting at 1
        piece_id = (piece_type - 1) % 6
    move_type_offset = 76
    piece_offset = piece_id
    board_offset = move_type_offset + piece_offset
    return board_offset


def get_plane_index_from_move(move) -> [int, int, int]:
    """
    Computes the index in the move policy matrix for a given move.

    Queen moves  | 56     ->  0..55
    Knight moves | 8      -> 56..63
    Promotions   | 12     -> 64..75
    Drop         | 5      -> 76..80
    ----------------------------
    Total 81
    :param move: Chess move object
    :return: Plane index

    """

    if move.drop:
        # dropping a piece
        piece_type = move.drop
        to_row, to_col = get_row_col(move.to_square)
        board_offset = get_plane_index_drop_move(piece_type)
        return board_offset, to_row, to_col
    if move.promotion:
        piece_type = move.promotion
        movement_vector = get_movement_vector(move)
        # a pawn can only move forward or capture to the left or right
        # => we only have to inspect the x axis
        # python-chess starts counting at 1
        from_row, from_col = get_row_col(move.from_square)
        board_offset = get_plane_index_promotion_move(piece_type, movement_vector)
        return board_offset, from_row, from_col

    # normal move
    movement_vector = get_movement_vector(move)
    from_row, from_col = get_row_col(move.from_square)

    absolute_movement_vector = np.abs(movement_vector)
    # a knight move can be identified by its special movement behaviour
    # only the knight has a '1' and a '2' in its movement vector
    is_knight_move = (min(absolute_movement_vector) == 1) and (max(absolute_movement_vector) == 2)
    if is_knight_move:
        board_offset = get_plane_index_knight_move(movement_vector)
        return board_offset, from_row, from_col
    move_type_offset = 0
    movement_offset = get_plane_index_queen_move(movement_vector)
    board_offset = move_type_offset + movement_offset
    return board_offset, from_row, from_col


def get_move_planes(move):
    """
    Maps a move to the networks output representation.

    Queen moves | 56     ->  0..55
    Knight moves | 8     -> 56..63
    Promotions | 12  -> 64..75
    Drop | 5             -> 76..80
    ----------------------------
    Total 81
    """
    board = np.zeros((81, 8, 8), dtype="bool")
    plane, row, column = get_plane_index_from_move(move)
    board[plane, row, column] = 1
    return board


# generated conversion list for the flattened policy map representation
# which describes the corresponding index for each move in the policy LABELS (constants.py) list
FLAT_PLANE_IDX = [
    896,
    960,
    1024,
    1088,
    1152,
    1216,
    1280,
    0,
    64,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    576,
    640,
    704,
    768,
    832,
    3648,
    3584,
    904,
    968,
    1032,
    1096,
    1160,
    1224,
    1288,
    1800,
    8,
    72,
    136,
    200,
    264,
    328,
    456,
    520,
    584,
    648,
    712,
    776,
    1352,
    3720,
    3656,
    3592,
    912,
    976,
    1040,
    1104,
    1168,
    1232,
    1296,
    1872,
    1808,
    16,
    80,
    144,
    208,
    272,
    464,
    528,
    592,
    656,
    720,
    1360,
    1424,
    3792,
    3728,
    3664,
    3600,
    920,
    984,
    1048,
    1112,
    1176,
    1240,
    1304,
    1944,
    1880,
    1816,
    24,
    88,
    152,
    216,
    472,
    536,
    600,
    664,
    1368,
    1432,
    1496,
    3800,
    3736,
    3672,
    3608,
    928,
    992,
    1056,
    1120,
    1184,
    1248,
    1312,
    2016,
    1952,
    1888,
    1824,
    32,
    96,
    160,
    480,
    544,
    608,
    1376,
    1440,
    1504,
    1568,
    3808,
    3744,
    3680,
    3616,
    936,
    1000,
    1064,
    1128,
    1192,
    1256,
    1320,
    2088,
    2024,
    1960,
    1896,
    1832,
    40,
    104,
    488,
    552,
    1384,
    1448,
    1512,
    1576,
    1640,
    3816,
    3752,
    3688,
    3624,
    944,
    1008,
    1072,
    1136,
    1200,
    1264,
    1328,
    2160,
    2096,
    2032,
    1968,
    1904,
    1840,
    48,
    496,
    1392,
    1456,
    1520,
    1584,
    1648,
    1712,
    3824,
    3760,
    3696,
    952,
    1016,
    1080,
    1144,
    1208,
    1272,
    1336,
    2232,
    2168,
    2104,
    2040,
    1976,
    1912,
    1848,
    1400,
    1464,
    1528,
    1592,
    1656,
    1720,
    1784,
    3832,
    3768,
    2689,
    897,
    961,
    1025,
    1089,
    1153,
    1217,
    1,
    65,
    129,
    193,
    257,
    321,
    385,
    449,
    513,
    577,
    641,
    705,
    769,
    3137,
    4033,
    3649,
    3585,
    2697,
    905,
    969,
    1033,
    1097,
    1161,
    1225,
    1801,
    9,
    73,
    137,
    201,
    265,
    329,
    2249,
    457,
    521,
    585,
    649,
    713,
    777,
    3145,
    1353,
    3721,
    4041,
    3657,
    3593,
    2705,
    913,
    977,
    1041,
    1105,
    1169,
    1233,
    1873,
    1809,
    17,
    81,
    145,
    209,
    273,
    2257,
    465,
    529,
    593,
    657,
    721,
    3153,
    1361,
    1425,
    3857,
    3793,
    3729,
    4049,
    3665,
    3601,
    2713,
    921,
    985,
    1049,
    1113,
    1177,
    1241,
    1945,
    1881,
    1817,
    25,
    89,
    153,
    217,
    2265,
    473,
    537,
    601,
    665,
    3161,
    1369,
    1433,
    1497,
    3865,
    3801,
    3737,
    4057,
    3673,
    3609,
    2721,
    929,
    993,
    1057,
    1121,
    1185,
    1249,
    2017,
    1953,
    1889,
    1825,
    33,
    97,
    161,
    2273,
    481,
    545,
    609,
    3169,
    1377,
    1441,
    1505,
    1569,
    3873,
    3809,
    3745,
    4065,
    3681,
    3617,
    2729,
    937,
    1001,
    1065,
    1129,
    1193,
    1257,
    2089,
    2025,
    1961,
    1897,
    1833,
    41,
    105,
    2281,
    489,
    553,
    3177,
    1385,
    1449,
    1513,
    1577,
    1641,
    3881,
    3817,
    3753,
    4073,
    3689,
    3625,
    2737,
    945,
    1009,
    1073,
    1137,
    1201,
    1265,
    2161,
    2097,
    2033,
    1969,
    1905,
    1841,
    49,
    2289,
    497,
    3185,
    1393,
    1457,
    1521,
    1585,
    1649,
    1713,
    3889,
    3825,
    3761,
    3697,
    2745,
    953,
    1017,
    1081,
    1145,
    1209,
    1273,
    2233,
    2169,
    2105,
    2041,
    1977,
    1913,
    1849,
    2297,
    1401,
    1465,
    1529,
    1593,
    1657,
    1721,
    3897,
    3833,
    3769,
    2754,
    2690,
    898,
    962,
    1026,
    1090,
    1154,
    2,
    66,
    130,
    194,
    258,
    322,
    386,
    450,
    514,
    578,
    642,
    706,
    3202,
    3138,
    3970,
    4034,
    3650,
    3586,
    2762,
    2698,
    906,
    970,
    1034,
    1098,
    1162,
    1802,
    10,
    74,
    138,
    202,
    266,
    330,
    2250,
    458,
    522,
    586,
    650,
    714,
    3210,
    3146,
    1354,
    3914,
    3978,
    3722,
    4042,
    3658,
    3594,
    2770,
    2706,
    914,
    978,
    1042,
    1106,
    1170,
    1874,
    1810,
    18,
    82,
    146,
    210,
    274,
    2322,
    2258,
    466,
    530,
    594,
    658,
    722,
    3218,
    3154,
    1362,
    1426,
    3922,
    3858,
    3986,
    3794,
    3730,
    4050,
    3666,
    3602,
    2778,
    2714,
    922,
    986,
    1050,
    1114,
    1178,
    1946,
    1882,
    1818,
    26,
    90,
    154,
    218,
    2330,
    2266,
    474,
    538,
    602,
    666,
    3226,
    3162,
    1370,
    1434,
    1498,
    3930,
    3866,
    3994,
    3802,
    3738,
    4058,
    3674,
    3610,
    2786,
    2722,
    930,
    994,
    1058,
    1122,
    1186,
    2018,
    1954,
    1890,
    1826,
    34,
    98,
    162,
    2338,
    2274,
    482,
    546,
    610,
    3234,
    3170,
    1378,
    1442,
    1506,
    1570,
    3938,
    3874,
    4002,
    3810,
    3746,
    4066,
    3682,
    3618,
    2794,
    2730,
    938,
    1002,
    1066,
    1130,
    1194,
    2090,
    2026,
    1962,
    1898,
    1834,
    42,
    106,
    2346,
    2282,
    490,
    554,
    3242,
    3178,
    1386,
    1450,
    1514,
    1578,
    1642,
    3946,
    3882,
    4010,
    3818,
    3754,
    4074,
    3690,
    3626,
    2802,
    2738,
    946,
    1010,
    1074,
    1138,
    1202,
    2162,
    2098,
    2034,
    1970,
    1906,
    1842,
    50,
    2354,
    2290,
    498,
    3186,
    1394,
    1458,
    1522,
    1586,
    1650,
    3954,
    3890,
    4018,
    3826,
    3762,
    3698,
    2810,
    2746,
    954,
    1018,
    1082,
    1146,
    1210,
    2234,
    2170,
    2106,
    2042,
    1978,
    1914,
    1850,
    2362,
    2298,
    1402,
    1466,
    1530,
    1594,
    1658,
    3962,
    3898,
    3834,
    3770,
    2819,
    2755,
    2691,
    899,
    963,
    1027,
    1091,
    3,
    67,
    131,
    195,
    259,
    323,
    387,
    451,
    515,
    579,
    643,
    3267,
    3203,
    3139,
    3971,
    4035,
    3651,
    3587,
    2827,
    2763,
    2699,
    907,
    971,
    1035,
    1099,
    1803,
    11,
    75,
    139,
    203,
    267,
    331,
    2251,
    459,
    523,
    587,
    651,
    3275,
    3211,
    3147,
    1355,
    3915,
    3979,
    3723,
    4043,
    3659,
    3595,
    2835,
    2771,
    2707,
    915,
    979,
    1043,
    1107,
    1875,
    1811,
    19,
    83,
    147,
    211,
    275,
    2323,
    2259,
    467,
    531,
    595,
    659,
    3283,
    3219,
    3155,
    1363,
    1427,
    3923,
    3859,
    3987,
    3795,
    3731,
    4051,
    3667,
    3603,
    2843,
    2779,
    2715,
    923,
    987,
    1051,
    1115,
    1947,
    1883,
    1819,
    27,
    91,
    155,
    219,
    2395,
    2331,
    2267,
    475,
    539,
    603,
    667,
    3291,
    3227,
    3163,
    1371,
    1435,
    1499,
    3931,
    3867,
    3995,
    3803,
    3739,
    4059,
    3675,
    3611,
    2851,
    2787,
    2723,
    931,
    995,
    1059,
    1123,
    2019,
    1955,
    1891,
    1827,
    35,
    99,
    163,
    2403,
    2339,
    2275,
    483,
    547,
    611,
    3299,
    3235,
    3171,
    1379,
    1443,
    1507,
    1571,
    3939,
    3875,
    4003,
    3811,
    3747,
    4067,
    3683,
    3619,
    2859,
    2795,
    2731,
    939,
    1003,
    1067,
    1131,
    2091,
    2027,
    1963,
    1899,
    1835,
    43,
    107,
    2411,
    2347,
    2283,
    491,
    555,
    3243,
    3179,
    1387,
    1451,
    1515,
    1579,
    3947,
    3883,
    4011,
    3819,
    3755,
    4075,
    3691,
    3627,
    2867,
    2803,
    2739,
    947,
    1011,
    1075,
    1139,
    2163,
    2099,
    2035,
    1971,
    1907,
    1843,
    51,
    2419,
    2355,
    2291,
    499,
    3187,
    1395,
    1459,
    1523,
    1587,
    3955,
    3891,
    4019,
    3827,
    3763,
    3699,
    2875,
    2811,
    2747,
    955,
    1019,
    1083,
    1147,
    2235,
    2171,
    2107,
    2043,
    1979,
    1915,
    1851,
    2427,
    2363,
    2299,
    1403,
    1467,
    1531,
    1595,
    3963,
    3899,
    3835,
    3771,
    2884,
    2820,
    2756,
    2692,
    900,
    964,
    1028,
    4,
    68,
    132,
    196,
    260,
    324,
    388,
    452,
    516,
    580,
    3332,
    3268,
    3204,
    3140,
    3972,
    4036,
    3652,
    3588,
    2892,
    2828,
    2764,
    2700,
    908,
    972,
    1036,
    1804,
    12,
    76,
    140,
    204,
    268,
    332,
    2252,
    460,
    524,
    588,
    3340,
    3276,
    3212,
    3148,
    1356,
    3916,
    3980,
    3724,
    4044,
    3660,
    3596,
    2900,
    2836,
    2772,
    2708,
    916,
    980,
    1044,
    1876,
    1812,
    20,
    84,
    148,
    212,
    276,
    2324,
    2260,
    468,
    532,
    596,
    3348,
    3284,
    3220,
    3156,
    1364,
    1428,
    3924,
    3860,
    3988,
    3796,
    3732,
    4052,
    3668,
    3604,
    2908,
    2844,
    2780,
    2716,
    924,
    988,
    1052,
    1948,
    1884,
    1820,
    28,
    92,
    156,
    220,
    2396,
    2332,
    2268,
    476,
    540,
    604,
    3356,
    3292,
    3228,
    3164,
    1372,
    1436,
    1500,
    3932,
    3868,
    3996,
    3804,
    3740,
    4060,
    3676,
    3612,
    2916,
    2852,
    2788,
    2724,
    932,
    996,
    1060,
    2020,
    1956,
    1892,
    1828,
    36,
    100,
    164,
    2468,
    2404,
    2340,
    2276,
    484,
    548,
    612,
    3300,
    3236,
    3172,
    1380,
    1444,
    1508,
    3940,
    3876,
    4004,
    3812,
    3748,
    4068,
    3684,
    3620,
    2924,
    2860,
    2796,
    2732,
    940,
    1004,
    1068,
    2092,
    2028,
    1964,
    1900,
    1836,
    44,
    108,
    2476,
    2412,
    2348,
    2284,
    492,
    556,
    3244,
    3180,
    1388,
    1452,
    1516,
    3948,
    3884,
    4012,
    3820,
    3756,
    4076,
    3692,
    3628,
    2932,
    2868,
    2804,
    2740,
    948,
    1012,
    1076,
    2164,
    2100,
    2036,
    1972,
    1908,
    1844,
    52,
    2484,
    2420,
    2356,
    2292,
    500,
    3188,
    1396,
    1460,
    1524,
    3956,
    3892,
    4020,
    3828,
    3764,
    3700,
    2940,
    2876,
    2812,
    2748,
    956,
    1020,
    1084,
    2236,
    2172,
    2108,
    2044,
    1980,
    1916,
    1852,
    2492,
    2428,
    2364,
    2300,
    1404,
    1468,
    1532,
    3964,
    3900,
    3836,
    3772,
    2949,
    2885,
    2821,
    2757,
    2693,
    901,
    965,
    5,
    69,
    133,
    197,
    261,
    325,
    389,
    453,
    517,
    3397,
    3333,
    3269,
    3205,
    3141,
    3973,
    4037,
    3653,
    3589,
    2957,
    2893,
    2829,
    2765,
    2701,
    909,
    973,
    1805,
    13,
    77,
    141,
    205,
    269,
    333,
    2253,
    461,
    525,
    3405,
    3341,
    3277,
    3213,
    3149,
    1357,
    3917,
    3981,
    3725,
    4045,
    3661,
    3597,
    2965,
    2901,
    2837,
    2773,
    2709,
    917,
    981,
    1877,
    1813,
    21,
    85,
    149,
    213,
    277,
    2325,
    2261,
    469,
    533,
    3413,
    3349,
    3285,
    3221,
    3157,
    1365,
    1429,
    3925,
    3861,
    3989,
    3797,
    3733,
    4053,
    3669,
    3605,
    2973,
    2909,
    2845,
    2781,
    2717,
    925,
    989,
    1949,
    1885,
    1821,
    29,
    93,
    157,
    221,
    2397,
    2333,
    2269,
    477,
    541,
    3357,
    3293,
    3229,
    3165,
    1373,
    1437,
    3933,
    3869,
    3997,
    3805,
    3741,
    4061,
    3677,
    3613,
    2981,
    2917,
    2853,
    2789,
    2725,
    933,
    997,
    2021,
    1957,
    1893,
    1829,
    37,
    101,
    165,
    2469,
    2405,
    2341,
    2277,
    485,
    549,
    3301,
    3237,
    3173,
    1381,
    1445,
    3941,
    3877,
    4005,
    3813,
    3749,
    4069,
    3685,
    3621,
    2989,
    2925,
    2861,
    2797,
    2733,
    941,
    1005,
    2093,
    2029,
    1965,
    1901,
    1837,
    45,
    109,
    2541,
    2477,
    2413,
    2349,
    2285,
    493,
    557,
    3245,
    3181,
    1389,
    1453,
    3949,
    3885,
    4013,
    3821,
    3757,
    4077,
    3693,
    3629,
    2997,
    2933,
    2869,
    2805,
    2741,
    949,
    1013,
    2165,
    2101,
    2037,
    1973,
    1909,
    1845,
    53,
    2549,
    2485,
    2421,
    2357,
    2293,
    501,
    3189,
    1397,
    1461,
    3957,
    3893,
    4021,
    3829,
    3765,
    3701,
    3005,
    2941,
    2877,
    2813,
    2749,
    957,
    1021,
    2237,
    2173,
    2109,
    2045,
    1981,
    1917,
    1853,
    2557,
    2493,
    2429,
    2365,
    2301,
    1405,
    1469,
    3965,
    3901,
    3837,
    3773,
    3014,
    2950,
    2886,
    2822,
    2758,
    2694,
    902,
    6,
    70,
    134,
    198,
    262,
    326,
    390,
    454,
    3462,
    3398,
    3334,
    3270,
    3206,
    3142,
    3974,
    4038,
    3590,
    3022,
    2958,
    2894,
    2830,
    2766,
    2702,
    910,
    1806,
    14,
    78,
    142,
    206,
    270,
    334,
    2254,
    462,
    3470,
    3406,
    3342,
    3278,
    3214,
    3150,
    1358,
    3918,
    3982,
    4046,
    3598,
    3030,
    2966,
    2902,
    2838,
    2774,
    2710,
    918,
    1878,
    1814,
    22,
    86,
    150,
    214,
    278,
    2326,
    2262,
    470,
    3414,
    3350,
    3286,
    3222,
    3158,
    1366,
    3926,
    3862,
    3990,
    3798,
    4054,
    3606,
    3038,
    2974,
    2910,
    2846,
    2782,
    2718,
    926,
    1950,
    1886,
    1822,
    30,
    94,
    158,
    222,
    2398,
    2334,
    2270,
    478,
    3358,
    3294,
    3230,
    3166,
    1374,
    3934,
    3870,
    3998,
    3806,
    4062,
    3614,
    3046,
    2982,
    2918,
    2854,
    2790,
    2726,
    934,
    2022,
    1958,
    1894,
    1830,
    38,
    102,
    166,
    2470,
    2406,
    2342,
    2278,
    486,
    3302,
    3238,
    3174,
    1382,
    3942,
    3878,
    4006,
    3814,
    4070,
    3622,
    3054,
    2990,
    2926,
    2862,
    2798,
    2734,
    942,
    2094,
    2030,
    1966,
    1902,
    1838,
    46,
    110,
    2542,
    2478,
    2414,
    2350,
    2286,
    494,
    3246,
    3182,
    1390,
    3950,
    3886,
    4014,
    3822,
    4078,
    3630,
    3062,
    2998,
    2934,
    2870,
    2806,
    2742,
    950,
    2166,
    2102,
    2038,
    1974,
    1910,
    1846,
    54,
    2614,
    2550,
    2486,
    2422,
    2358,
    2294,
    502,
    3190,
    1398,
    3958,
    3894,
    4022,
    3830,
    3070,
    3006,
    2942,
    2878,
    2814,
    2750,
    958,
    2238,
    2174,
    2110,
    2046,
    1982,
    1918,
    1854,
    2622,
    2558,
    2494,
    2430,
    2366,
    2302,
    1406,
    3966,
    3902,
    3838,
    3079,
    3015,
    2951,
    2887,
    2823,
    2759,
    2695,
    7,
    71,
    135,
    199,
    263,
    327,
    391,
    3527,
    3463,
    3399,
    3335,
    3271,
    3207,
    3143,
    3975,
    4039,
    3087,
    3023,
    2959,
    2895,
    2831,
    2767,
    2703,
    1807,
    15,
    79,
    143,
    207,
    271,
    335,
    2255,
    3471,
    3407,
    3343,
    3279,
    3215,
    3151,
    3919,
    3983,
    4047,
    3095,
    3031,
    2967,
    2903,
    2839,
    2775,
    2711,
    1879,
    1815,
    23,
    87,
    151,
    215,
    279,
    2327,
    2263,
    3415,
    3351,
    3287,
    3223,
    3159,
    3927,
    3863,
    3991,
    4055,
    3103,
    3039,
    2975,
    2911,
    2847,
    2783,
    2719,
    1951,
    1887,
    1823,
    31,
    95,
    159,
    223,
    2399,
    2335,
    2271,
    3359,
    3295,
    3231,
    3167,
    3935,
    3871,
    3999,
    4063,
    3111,
    3047,
    2983,
    2919,
    2855,
    2791,
    2727,
    2023,
    1959,
    1895,
    1831,
    39,
    103,
    167,
    2471,
    2407,
    2343,
    2279,
    3303,
    3239,
    3175,
    3943,
    3879,
    4007,
    4071,
    3119,
    3055,
    2991,
    2927,
    2863,
    2799,
    2735,
    2095,
    2031,
    1967,
    1903,
    1839,
    47,
    111,
    2543,
    2479,
    2415,
    2351,
    2287,
    3247,
    3183,
    3951,
    3887,
    4015,
    4079,
    3127,
    3063,
    2999,
    2935,
    2871,
    2807,
    2743,
    2167,
    2103,
    2039,
    1975,
    1911,
    1847,
    55,
    2615,
    2551,
    2487,
    2423,
    2359,
    2295,
    3191,
    3959,
    3895,
    4023,
    3135,
    3071,
    3007,
    2943,
    2879,
    2815,
    2751,
    2239,
    2175,
    2111,
    2047,
    1983,
    1919,
    1855,
    2687,
    2623,
    2559,
    2495,
    2431,
    2367,
    2303,
    3967,
    3903,
    4744,
    4784,
    4808,
    4848,
    4552,
    4592,
    4616,
    4656,
    4360,
    4400,
    4424,
    4464,
    4168,
    4208,
    4232,
    4272,
    4745,
    4785,
    4681,
    4721,
    4809,
    4849,
    4553,
    4593,
    4489,
    4529,
    4617,
    4657,
    4361,
    4401,
    4297,
    4337,
    4425,
    4465,
    4169,
    4209,
    4105,
    4145,
    4233,
    4273,
    4746,
    4786,
    4682,
    4722,
    4810,
    4850,
    4554,
    4594,
    4490,
    4530,
    4618,
    4658,
    4362,
    4402,
    4298,
    4338,
    4426,
    4466,
    4170,
    4210,
    4106,
    4146,
    4234,
    4274,
    4747,
    4787,
    4683,
    4723,
    4811,
    4851,
    4555,
    4595,
    4491,
    4531,
    4619,
    4659,
    4363,
    4403,
    4299,
    4339,
    4427,
    4467,
    4171,
    4211,
    4107,
    4147,
    4235,
    4275,
    4748,
    4788,
    4684,
    4724,
    4812,
    4852,
    4556,
    4596,
    4492,
    4532,
    4620,
    4660,
    4364,
    4404,
    4300,
    4340,
    4428,
    4468,
    4172,
    4212,
    4108,
    4148,
    4236,
    4276,
    4749,
    4789,
    4685,
    4725,
    4813,
    4853,
    4557,
    4597,
    4493,
    4533,
    4621,
    4661,
    4365,
    4405,
    4301,
    4341,
    4429,
    4469,
    4173,
    4213,
    4109,
    4149,
    4237,
    4277,
    4750,
    4790,
    4686,
    4726,
    4814,
    4854,
    4558,
    4598,
    4494,
    4534,
    4622,
    4662,
    4366,
    4406,
    4302,
    4342,
    4430,
    4470,
    4174,
    4214,
    4110,
    4150,
    4238,
    4278,
    4751,
    4791,
    4687,
    4727,
    4559,
    4599,
    4495,
    4535,
    4367,
    4407,
    4303,
    4343,
    4175,
    4215,
    4111,
    4151,
    4928,
    4992,
    5056,
    5120,
    4872,
    4936,
    5000,
    5064,
    5128,
    4880,
    4944,
    5008,
    5072,
    5136,
    4888,
    4952,
    5016,
    5080,
    5144,
    4896,
    4960,
    5024,
    5088,
    5152,
    4904,
    4968,
    5032,
    5096,
    5160,
    4912,
    4976,
    5040,
    5104,
    5168,
    4984,
    5048,
    5112,
    5176,
    4929,
    4993,
    5057,
    5121,
    4873,
    4937,
    5001,
    5065,
    5129,
    4881,
    4945,
    5009,
    5073,
    5137,
    4889,
    4953,
    5017,
    5081,
    5145,
    4897,
    4961,
    5025,
    5089,
    5153,
    4905,
    4969,
    5033,
    5097,
    5161,
    4913,
    4977,
    5041,
    5105,
    5169,
    4985,
    5049,
    5113,
    5177,
    4930,
    4994,
    5058,
    5122,
    4874,
    4938,
    5002,
    5066,
    5130,
    4882,
    4946,
    5010,
    5074,
    5138,
    4890,
    4954,
    5018,
    5082,
    5146,
    4898,
    4962,
    5026,
    5090,
    5154,
    4906,
    4970,
    5034,
    5098,
    5162,
    4914,
    4978,
    5042,
    5106,
    5170,
    4986,
    5050,
    5114,
    5178,
    4931,
    4995,
    5059,
    5123,
    4875,
    4939,
    5003,
    5067,
    5131,
    4883,
    4947,
    5011,
    5075,
    5139,
    4891,
    4955,
    5019,
    5083,
    5147,
    4899,
    4963,
    5027,
    5091,
    5155,
    4907,
    4971,
    5035,
    5099,
    5163,
    4915,
    4979,
    5043,
    5107,
    5171,
    4987,
    5051,
    5115,
    5179,
    4932,
    4996,
    5060,
    5124,
    4876,
    4940,
    5004,
    5068,
    5132,
    4884,
    4948,
    5012,
    5076,
    5140,
    4892,
    4956,
    5020,
    5084,
    5148,
    4900,
    4964,
    5028,
    5092,
    5156,
    4908,
    4972,
    5036,
    5100,
    5164,
    4916,
    4980,
    5044,
    5108,
    5172,
    4988,
    5052,
    5116,
    5180,
    4933,
    4997,
    5061,
    5125,
    4877,
    4941,
    5005,
    5069,
    5133,
    4885,
    4949,
    5013,
    5077,
    5141,
    4893,
    4957,
    5021,
    5085,
    5149,
    4901,
    4965,
    5029,
    5093,
    5157,
    4909,
    4973,
    5037,
    5101,
    5165,
    4917,
    4981,
    5045,
    5109,
    5173,
    4989,
    5053,
    5117,
    5181,
    4934,
    4998,
    5062,
    5126,
    4878,
    4942,
    5006,
    5070,
    5134,
    4886,
    4950,
    5014,
    5078,
    5142,
    4894,
    4958,
    5022,
    5086,
    5150,
    4902,
    4966,
    5030,
    5094,
    5158,
    4910,
    4974,
    5038,
    5102,
    5166,
    4918,
    4982,
    5046,
    5110,
    5174,
    4990,
    5054,
    5118,
    5182,
    4935,
    4999,
    5063,
    5127,
    4879,
    4943,
    5007,
    5071,
    5135,
    4887,
    4951,
    5015,
    5079,
    5143,
    4895,
    4959,
    5023,
    5087,
    5151,
    4903,
    4967,
    5031,
    5095,
    5159,
    4911,
    4975,
    5039,
    5103,
    5167,
    4919,
    4983,
    5047,
    5111,
    5175,
    4991,
    5055,
    5119,
    5183,
]


if __name__ == "__main__":
    LABEL_PLANE_IDX = []
    print("FLAT_PLANE_IDX = [")
    for mv_uci in LABELS:
        move = chess.Move.from_uci(mv_uci)
        board_offset, from_row, from_col = get_plane_index_from_move(move)
        flat_plane_idx = board_offset * BOARD_WIDTH * BOARD_HEIGHT + from_row * BOARD_WIDTH + from_col
        print("   " + str(flat_plane_idx) + ",")
        LABEL_PLANE_IDX.append(flat_plane_idx)
    print("]")