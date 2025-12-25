#!/usr/bin/env python3
"""Fix three builtin function issues for iteration 003."""

import sys

def fix_builtin_functions():
    filepath = "moo_interp/builtin_functions.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Add shift builtin (after random_bytes function around line 1043)
    shift_function = '''
    def shift(self, n, count):
        """Shift n by count bits. Positive count=left shift, negative=right shift.
        
        shift(1, 3) => 8 (1 << 3)
        shift(8, -3) => 1 (8 >> 3)
        """
        if count >= 0:
            return n << count
        else:
            return n >> (-count)
'''
    
    # Insert shift after reseed_random (line ~1043)
    marker = "    def reseed_random(self):\n        \"\"\"Reseed the random number generator.\"\"\"\n        random.seed()\n        return 0"
    if marker in content:
        content = content.replace(marker, marker + shift_function)
        print("[OK] Added shift() builtin")
    else:
        print("[FAIL] Could not find reseed_random marker")
        
    # Fix 2: Change eval signature from (self, x) to (self, *args) with concatenation
    old_eval_sig = "    def eval(self, x):"
    new_eval_sig = "    def eval(self, *args):"
    
    old_eval_body = '''        """
        from .moo_ast import compile, run
        try:
            compiled = compile(x)
            compiled.debug = True
            compiled.this = -1
            compiled.verb = ""
            result = run(x)
        except Exception as e:
            return MOOList([False, MOOList([e])])
        return MOOList([True, result.result])'''
    
    new_eval_body = '''
        eval() accepts multiple string arguments and concatenates them before evaluation.
        """
        # Check for no arguments
        if len(args) == 0:
            raise MOOException(MOOError.E_ARGS, "eval() requires at least one argument")

        # Check all arguments are strings
        for arg in args:
            if not isinstance(arg, (str, MOOString)):
                raise MOOException(MOOError.E_TYPE, "eval() requires string arguments")

        # Concatenate all string arguments
        code = ''.join(str(arg.data if hasattr(arg, 'data') else arg) for arg in args)

        from .moo_ast import compile, run
        try:
            compiled = compile(code)
            compiled.debug = True
            compiled.this = -1
            compiled.verb = ""
            result = run(code)
        except Exception as e:
            return MOOList([False, MOOList([e])])
        return MOOList([True, result.result])'''
    
    if old_eval_sig in content:
        content = content.replace(old_eval_sig, new_eval_sig)
        content = content.replace(old_eval_body, new_eval_body)
        print("[OK] Fixed eval() signature to accept *args")
    else:
        print("[FAIL] Could not find eval signature")
        
    # Fix 3: Fix toobj to return ObjNum instead of int
    old_toobj = '''    def toobj(self, x):
        """Convert a value to an object reference."""
        if isinstance(x, MooObject):
            return x
        elif isinstance(x, int):
            return x  # Return int as object number for now
        elif isinstance(x, (str, MOOString)):
            s = str(x).strip()
            if s.startswith('#'):
                s = s[1:]
            try:
                return int(s)
            except ValueError:
                return -1
        elif isinstance(x, float):
            return int(x)
        else:
            return -1'''
    
    new_toobj = '''    def toobj(self, x):
        """Convert a value to an object reference."""
        if isinstance(x, ObjNum):
            return x
        elif isinstance(x, MooObject):
            return ObjNum(x.id)
        elif isinstance(x, int):
            return ObjNum(x)
        elif isinstance(x, (str, MOOString)):
            s = str(x).strip()
            if s.startswith('#'):
                s = s[1:]
            try:
                obj_id = int(s)
                return ObjNum(obj_id)
            except ValueError:
                return ObjNum(0)  # Invalid strings return #0 per MOO spec
        elif isinstance(x, float):
            return ObjNum(int(x))
        else:
            return ObjNum(0)'''
    
    if old_toobj in content:
        content = content.replace(old_toobj, new_toobj)
        print("[OK] Fixed toobj() to return ObjNum")
    else:
        print("[FAIL] Could not find toobj function")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n[DONE] All fixes applied to", filepath)

if __name__ == '__main__':
    fix_builtin_functions()
