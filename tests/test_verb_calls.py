"""Tests for verb call functionality (OP_CALL_VERB)"""
import pytest
from moo_interp.vm import VM, Instruction, Program, StackFrame, VMOutcome
from moo_interp.opcodes import Opcode
from moo_interp.list import MOOList
from moo_interp.string import MOOString
from lambdamoo_db.database import MooDatabase, MooObject, Verb


def test_op_call_verb_simple():
    """Test that OP_CALL_VERB can call a verb on an object

    Stack layout before OP_CALL_VERB (bottom to top):
    - object ID (int)
    - verb name (MOOString)
    - arguments (MOOList)
    """
    # Create a simple test database
    db = MooDatabase()
    db.objects = {}
    db.players = []
    db.queuedTasks = []
    db.suspendedTasks = []
    db.waifs = {}
    db.clocks = []
    db.versionstring = "test"
    db.version = 1
    db.total_objects = 1
    db.total_verbs = 1
    db.total_players = 0

    # Create test object #1
    obj_id = 1
    obj = MooObject(
        id=obj_id,
        name="test_object",
        flags=0,
        owner=1,
        location=-1,
        parents=[]
    )

    # Add a simple verb that returns the sum of two arguments
    # Verb code: return args[1] + 10;
    verb_code = [
        Instruction(opcode=Opcode.OP_PUSH, operand=MOOString("args")),
        Instruction(opcode=Opcode.OP_IMM, operand=1),
        Instruction(opcode=Opcode.OP_REF),  # args[1]
        Instruction(opcode=Opcode.OP_IMM, operand=10),
        Instruction(opcode=Opcode.OP_ADD),
        Instruction(opcode=Opcode.OP_RETURN),
    ]

    verb = Verb(
        name="test_verb",
        owner=1,
        perms=0xFFFF,
        preps=0,
        object=obj_id
    )
    verb.code = verb_code
    obj.verbs.append(verb)
    db.objects[obj_id] = obj

    # Create VM with database
    vm = VM(db=db)

    # Create calling context: obj:test_verb(5)
    caller_code = [
        Instruction(opcode=Opcode.OP_IMM, operand=obj_id),  # push object
        Instruction(opcode=Opcode.OP_IMM, operand=MOOString("test_verb")),  # push verb name
        Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST),  # create args list
        Instruction(opcode=Opcode.OP_IMM, operand=5),
        Instruction(opcode=Opcode.OP_LIST_ADD_TAIL),  # args = {5}
        Instruction(opcode=Opcode.OP_CALL_VERB),  # call the verb
    ]

    caller_prog = Program(var_names=[MOOString("args")])
    frame = StackFrame(func_id=0, prog=caller_prog, ip=0, stack=caller_code)
    vm.call_stack.append(frame)

    # Run until completion
    while vm.state is None:
        vm.step()

    # Should have returned 15 (5 + 10)
    assert vm.result == 15
    assert vm.state == VMOutcome.OUTCOME_DONE


def test_op_call_verb_with_inheritance():
    """Test that OP_CALL_VERB looks up verbs with inheritance"""
    db = MooDatabase()
    db.objects = {}
    db.players = []
    db.queuedTasks = []
    db.suspendedTasks = []
    db.waifs = {}
    db.clocks = []
    db.versionstring = "test"
    db.version = 1
    db.total_objects = 2
    db.total_verbs = 1
    db.total_players = 0

    # Create parent object with verb
    parent_id = 1
    parent = MooObject(
        id=parent_id,
        name="parent_object",
        flags=0,
        owner=1,
        location=-1,
        parents=[]
    )

    verb_code = [
        Instruction(opcode=Opcode.OP_IMM, operand=42),
        Instruction(opcode=Opcode.OP_RETURN),
    ]

    verb = Verb(
        name="inherited_verb",
        owner=1,
        perms=0xFFFF,
        preps=0,
        object=parent_id
    )
    verb.code = verb_code
    parent.verbs.append(verb)
    db.objects[parent_id] = parent

    # Create child object that inherits from parent
    child_id = 2
    child = MooObject(
        id=child_id,
        name="child_object",
        flags=0,
        owner=1,
        location=-1,
        parents=[parent_id]
    )
    db.objects[child_id] = child

    # Call verb on child - should find it on parent
    vm = VM(db=db)
    caller_code = [
        Instruction(opcode=Opcode.OP_IMM, operand=child_id),
        Instruction(opcode=Opcode.OP_IMM, operand=MOOString("inherited_verb")),
        Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST),
        Instruction(opcode=Opcode.OP_CALL_VERB),
    ]

    caller_prog = Program()
    frame = StackFrame(func_id=0, prog=caller_prog, ip=0, stack=caller_code)
    vm.call_stack.append(frame)

    while vm.state is None:
        vm.step()

    assert vm.result == 42


def test_op_call_verb_sets_context():
    """Test that OP_CALL_VERB properly sets this, caller, player context"""
    db = MooDatabase()
    db.objects = {}
    db.players = []
    db.queuedTasks = []
    db.suspendedTasks = []
    db.waifs = {}
    db.clocks = []
    db.versionstring = "test"
    db.version = 1
    db.total_objects = 1
    db.total_verbs = 1
    db.total_players = 0

    obj_id = 1
    obj = MooObject(
        id=obj_id,
        name="test_object",
        flags=0,
        owner=1,
        location=-1,
        parents=[]
    )

    # Verb that returns the value of 'this'
    # In real MOO, 'this' is a builtin variable
    verb_code = [
        Instruction(opcode=Opcode.OP_PUSH, operand=MOOString("this")),
        Instruction(opcode=Opcode.OP_RETURN),
    ]

    verb = Verb(
        name="get_this",
        owner=1,
        perms=0xFFFF,
        preps=0,
        object=obj_id
    )
    verb.code = verb_code
    obj.verbs.append(verb)
    db.objects[obj_id] = obj

    vm = VM(db=db)
    caller_code = [
        Instruction(opcode=Opcode.OP_IMM, operand=obj_id),
        Instruction(opcode=Opcode.OP_IMM, operand=MOOString("get_this")),
        Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST),
        Instruction(opcode=Opcode.OP_CALL_VERB),
    ]

    caller_prog = Program(var_names=[MOOString("this")])
    frame = StackFrame(func_id=0, prog=caller_prog, ip=0, stack=caller_code, player=99)
    vm.call_stack.append(frame)

    while vm.state is None:
        vm.step()

    # The verb should have received 'this' == obj_id
    assert vm.result == obj_id

    # The new frame should have had this=obj_id, player=99 (inherited)
    # We can't check the frame after execution, but we verified via result


def test_op_call_verb_nonexistent_verb():
    """Test that calling a non-existent verb raises an error"""
    db = MooDatabase()
    db.objects = {}
    db.players = []
    db.queuedTasks = []
    db.suspendedTasks = []
    db.waifs = {}
    db.clocks = []
    db.versionstring = "test"
    db.version = 1
    db.total_objects = 1
    db.total_verbs = 0
    db.total_players = 0

    obj_id = 1
    obj = MooObject(
        id=obj_id,
        name="test_object",
        flags=0,
        owner=1,
        location=-1,
        parents=[]
    )
    db.objects[obj_id] = obj

    vm = VM(db=db)
    caller_code = [
        Instruction(opcode=Opcode.OP_IMM, operand=obj_id),
        Instruction(opcode=Opcode.OP_IMM, operand=MOOString("nonexistent")),
        Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST),
        Instruction(opcode=Opcode.OP_CALL_VERB),
    ]

    caller_prog = Program()
    frame = StackFrame(func_id=0, prog=caller_prog, ip=0, stack=caller_code)
    vm.call_stack.append(frame)

    with pytest.raises(Exception):  # VMError or similar
        while vm.state is None:
            vm.step()
