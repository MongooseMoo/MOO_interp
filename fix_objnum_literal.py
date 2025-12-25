"""Fix ObjnumLiteral to emit ObjNum instead of plain int."""

filepath = "moo_interp/moo_ast.py"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the ObjnumLiteral.to_bytecode method
in_objnum = False
fixed = False

for i, line in enumerate(lines):
    if 'class ObjnumLiteral' in line:
        in_objnum = True
    elif in_objnum and 'def to_bytecode' in line:
        # Found the method - need to fix the next line
        # Look for the return statement
        for j in range(i+1, min(i+10, len(lines))):
            if 'return [self.emit_byte(Opcode.OP_IMM, self.value)]' in lines[j]:
                # Replace with ObjNum-wrapped version
                indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                lines[j] = indent + 'from lambdamoo_db.database import ObjNum\n'
                lines.insert(j+1, indent + 'return [self.emit_byte(Opcode.OP_IMM, ObjNum(self.value))]\n')
                fixed = True
                print(f"[OK] Fixed ObjnumLiteral.to_bytecode() at line {j+1}")
                break
        break

if not fixed:
    print("[FAIL] Could not find ObjnumLiteral.to_bytecode()")
    exit(1)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("[DONE] ObjnumLiteral now wraps value in ObjNum")
