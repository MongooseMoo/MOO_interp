# moo_interp

A Python bytecode interpreter for the MOO programming language.

## Features

- Complete MOO language parser (Lark/Earley)
- Bytecode compiler
- Stack-based virtual machine
- 100+ builtin functions
- 1-based indexing types (MOOString, MOOList, MOOMap)

## Usage

```python
from moo_interp.moo_ast import compile, parse, run

code = 'return 1 + 2;'
result = run(compile(parse(code)))
print(result)  # 3
```

## Requirements

- Python 3.8+
- attrs
- lark
