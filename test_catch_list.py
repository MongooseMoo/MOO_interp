from moo_interp.moo_ast import parse, compile as compile_moo
from moo_interp.vm import VM
import logging

logging.basicConfig(level=logging.DEBUG)

# Test: try-catch in list
code = '''
return {"e", `1/0 ! ANY => 0'};
'''

print("Code:", repr(code))
ast = parse(code)
print("AST:", ast)
frame = compile_moo(ast)
print("Bytecode:", frame.stack)
vm = VM(db=None, bi_funcs=None)
vm.call_stack = [frame]

try:
    for _ in vm.run():
        pass
    print(f"Result: {vm.result}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
