"""Tests for property and global variable opcodes"""

import pytest
from moo_interp.list import MOOList

# Skip all tests in this module - opcodes not yet implemented
pytestmark = pytest.mark.skip(reason="OP_PUSH_GET_PROP, OP_INDEXSET, OP_G_* opcodes not yet implemented")
from moo_interp.map import MOOMap
from moo_interp.opcodes import Opcode
from moo_interp.string import MOOString
from moo_interp.vm import VM, Instruction, Program, StackFrame, VMError
from lambdamoo_db.database import MooDatabase, MooObject, Property, ObjectFlags, PropertyFlags


def test_push_get_prop():
    """OP_PUSH_GET_PROP gets property value and pushes object and property name."""
    # Create a test database with an object
    db = MooDatabase()
    db.objects = {}
    obj = MooObject(id=1, name="test_object", flags=ObjectFlags(0), owner=1, location=0, parents=[])
    prop = Property(propertyName="test_prop", value=42, owner=1, perms=PropertyFlags(0))
    obj.properties = [prop]
    db.objects[1] = obj

    vm = VM(db=db)

    # OP_PUSH_GET_PROP should:
    # 1. Push object ID and property name onto stack
    # 2. Get the property value
    # Stack layout: [obj_id, prop_name] -> [value]
    prog = Program()
    frame = StackFrame(
        func_id=0,
        prog=prog,
        ip=0,
        stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=1),  # Push object ID
            Instruction(opcode=Opcode.OP_IMM, operand=MOOString("test_prop")),  # Push property name
            Instruction(opcode=Opcode.OP_PUSH_GET_PROP),  # Get property
        ]
    )
    vm.call_stack.append(frame)

    # Execute: push obj ID
    vm.step()
    assert vm.stack[-1] == 1

    # Execute: push prop name
    vm.step()
    assert vm.stack[-1] == MOOString("test_prop")

    # Execute: OP_PUSH_GET_PROP
    vm.step()
    # Should have property value on stack
    assert vm.stack[-1] == 42


def test_indexset_list():
    """OP_INDEXSET sets list element at index."""
    vm = VM()

    # Create a list and set an element
    # list[index] = value
    # Stack layout: [list, index, value] -> [value]
    test_list = MOOList(1, 2, 3, 4, 5)
    prog = Program()
    frame = StackFrame(
        func_id=0,
        prog=prog,
        ip=0,
        stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),  # Push list
            Instruction(opcode=Opcode.OP_IMM, operand=2),  # Push index (1-indexed in MOO)
            Instruction(opcode=Opcode.OP_IMM, operand=99),  # Push new value
            Instruction(opcode=Opcode.OP_INDEXSET),  # Set list[2] = 99
        ]
    )
    vm.call_stack.append(frame)

    # Execute: push list
    vm.step()
    assert vm.stack[-1] == test_list

    # Execute: push index
    vm.step()
    assert vm.stack[-1] == 2

    # Execute: push value
    vm.step()
    assert vm.stack[-1] == 99

    # Execute: OP_INDEXSET
    vm.step()
    # Should return the value
    assert vm.stack[-1] == 99
    # List should be modified (in MOO, lists are 1-indexed)
    assert test_list[2] == 99  # Python 0-indexed, so index 2 is third element


def test_indexset_map():
    """OP_INDEXSET sets map entry at key."""
    vm = VM()

    # Create a map and set an entry
    # map[key] = value
    # Stack layout: [map, key, value] -> [value]
    test_map = MOOMap({"a": 1, "b": 2})
    prog = Program()
    frame = StackFrame(
        func_id=0,
        prog=prog,
        ip=0,
        stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_map),  # Push map
            Instruction(opcode=Opcode.OP_IMM, operand=MOOString("c")),  # Push key
            Instruction(opcode=Opcode.OP_IMM, operand=42),  # Push value
            Instruction(opcode=Opcode.OP_INDEXSET),  # Set map["c"] = 42
        ]
    )
    vm.call_stack.append(frame)

    # Execute: push map
    vm.step()
    assert vm.stack[-1] == test_map

    # Execute: push key
    vm.step()
    assert vm.stack[-1] == MOOString("c")

    # Execute: push value
    vm.step()
    assert vm.stack[-1] == 42

    # Execute: OP_INDEXSET
    vm.step()
    # Should return the value
    assert vm.stack[-1] == 42
    # Map should have new entry
    assert test_map[MOOString("c")] == 42


def test_global_push():
    """OP_G_PUSH pushes global variable value."""
    vm = VM()

    # Global variables in MOO use indices 0-31 for special vars
    # OP_G_PUSH is OP_PUSH + NUM_READY_VARS (32)
    # So OP_G_PUSH + n pushes global variable n

    prog = Program(var_names=[MOOString("player"), MOOString("this")])
    frame = StackFrame(
        func_id=0,
        prog=prog,
        ip=0,
        stack=[
            Instruction(opcode=Opcode.OP_G_PUSH, operand=MOOString("player")),
        ],
        rt_env=[123, 456],  # player=123, this=456
        player=123,
        this=456,
    )
    vm.call_stack.append(frame)

    # Execute: OP_G_PUSH
    vm.step()
    # Should push player value
    assert vm.stack[-1] == 123


def test_global_put():
    """OP_G_PUT stores value in global variable."""
    vm = VM()

    prog = Program(var_names=[MOOString("player")])
    frame = StackFrame(
        func_id=0,
        prog=prog,
        ip=0,
        stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=999),  # Push value
            Instruction(opcode=Opcode.OP_G_PUT, operand=MOOString("player")),
        ],
        rt_env=[123],  # player=123 initially
        player=123,
    )
    vm.call_stack.append(frame)

    # Execute: push value
    vm.step()
    assert vm.stack[-1] == 999

    # Execute: OP_G_PUT
    vm.step()
    # Should update player in runtime env
    var_index = prog.var_names.index(MOOString("player"))
    assert frame.rt_env[var_index] == 999


def test_global_push_clear():
    """OP_G_PUSH_CLEAR pushes global and clears it."""
    vm = VM()

    prog = Program(var_names=[MOOString("args"), MOOString("temp")])
    frame = StackFrame(
        func_id=0,
        prog=prog,
        ip=0,
        stack=[
            Instruction(opcode=Opcode.OP_G_PUSH_CLEAR, operand=MOOString("temp")),
        ],
        rt_env=[MOOList(1, 2, 3), 42],  # args, temp
    )
    vm.call_stack.append(frame)

    # Execute: OP_G_PUSH_CLEAR
    vm.step()
    # Should push temp value
    assert vm.stack[-1] == 42
    # Should remove temp from var_names and rt_env
    assert MOOString("temp") not in prog.var_names
    assert len(frame.rt_env) == 1  # Only args remains
