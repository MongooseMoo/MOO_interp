import os

from lark import Lark
import pathlib

path = pathlib.Path(__file__).parent.absolute()
grammar = open(path / "parser.lark", "r")
parser = Lark(grammar)
