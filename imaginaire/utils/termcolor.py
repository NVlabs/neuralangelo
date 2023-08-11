'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import pprint

import termcolor


def red(x): return termcolor.colored(str(x), color="red")
def green(x): return termcolor.colored(str(x), color="green")
def blue(x): return termcolor.colored(str(x), color="blue")
def cyan(x): return termcolor.colored(str(x), color="cyan")
def yellow(x): return termcolor.colored(str(x), color="yellow")
def magenta(x): return termcolor.colored(str(x), color="magenta")
def grey(x): return termcolor.colored(str(x), color="grey")


COLORS = {
    'red': red, 'green': green, 'blue': blue, 'cyan': cyan, 'yellow': yellow, 'magenta': magenta, 'grey': grey
}


def PP(x):
    string = pprint.pformat(x, indent=2)
    if isinstance(x, dict):
        string = '{\n ' + string[1:-1] + '\n}'
    return string


def alert(x, color='red'):
    color = COLORS[color]
    print(color('-' * 32))
    print(color(f'* {x}'))
    print(color('-' * 32))
