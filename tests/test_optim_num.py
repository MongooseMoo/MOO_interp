"""Test OPTIM_NUM optimized immediate number opcodes."""

import pytest
from moo_interp.opcodes import Opcode
from moo_interp.vm import VM, Instruction, StackFrame, Program


def test_optim_num_formula():
    """Test that OPTIM_NUM opcodes produce correct immediate values.
    
    According to C reference:
    OPTIM_NUM_LOW = -10
    OPTIM_NUM_START = 53
    value = opcode - OPTIM_NUM_START + OPTIM_NUM_LOW
    
    So:
    - opcode 53 -> -10
    - opcode 63 -> 0
    - opcode 64 -> 1
    - opcode 114 -> 51
    - opcode 115 -> 52
    """
    test_cases = [
        (53, -10),   # First optimized number
        (63, 0),     # Zero
        (64, 1),     # One
        (114, 51),   # From the problem description
        (115, 52),   # From the problem description
        (123, 60),   # Another example
        (255, 192),  # Last possible opcode
    ]
    
    for opcode, expected_value in test_cases:
        # Create a minimal VM with a frame containing just this opcode
        vm = VM(db=None, bi_funcs={})
        
        # Create a frame with the opcode instruction
        frame = StackFrame(
            func_id=0,
            prog=Program(),
            ip=0,
            stack=[Instruction(opcode=opcode, operand=None)]
        )
        vm.call_stack = [frame]
        
        # Execute one step
        vm.step()
        
        # Check that the value was pushed onto the stack
        assert len(vm.stack) == 1, f"Expected 1 item on stack for opcode {opcode}"
        assert vm.stack[0] == expected_value, \
            f"Opcode {opcode} should push {expected_value}, got {vm.stack[0]}"


def test_optim_num_in_compiled_code():
    """Test that MOO code compiles to use OPTIM_NUM and executes correctly."""
    from moo_interp.moo_ast import parse_and_compile
    
    # Small integers should compile to OPTIM_NUM opcodes
    code = """
    x = 0;
    y = 1;
    z = -5;
    return x + y + z;
    """
    
    bytecode = parse_and_compile(code)
    
    # Check that we have some OPTIM_NUM opcodes (integers >= 53)
    optim_num_opcodes = [
        instr for instr in bytecode 
        if isinstance(instr.opcode, int) and instr.opcode >= 53
    ]
    
    # We should have compiled some small integers to OPTIM_NUM
    assert len(optim_num_opcodes) > 0, \
        "Expected some OPTIM_NUM opcodes for small integers"
    
    # Now execute and verify the result
    vm = VM(db=None, bi_funcs={})
    frame = StackFrame(
        func_id=0,
        prog=Program(),
        ip=0,
        stack=bytecode
    )
    vm.call_stack = [frame]
    
    # Run until completion
    for _ in vm.run():
        pass
    
    # Result should be 0 + 1 + (-5) = -4
    assert vm.result == -4, f"Expected result -4, got {vm.result}"


def test_optim_num_range():
    """Test the full range of OPTIM_NUM values."""
    from moo_interp.opcodes import Opcode
    
    OPTIM_NUM_START = 53
    OPTIM_NUM_LOW = -10
    Last_Opcode = 255
    OPTIM_NUM_HI = Last_Opcode - OPTIM_NUM_START + OPTIM_NUM_LOW
    
    # Test boundary values
    test_cases = [
        (OPTIM_NUM_START, OPTIM_NUM_LOW),  # First: 53 -> -10
        (255, OPTIM_NUM_HI),                # Last: 255 -> 192
    ]
    
    for opcode, expected_value in test_cases:
        vm = VM(db=None, bi_funcs={})
        frame = StackFrame(
            func_id=0,
            prog=Program(),
            ip=0,
            stack=[Instruction(opcode=opcode, operand=None)]
        )
        vm.call_stack = [frame]
        
        vm.step()
        
        assert len(vm.stack) == 1
        assert vm.stack[0] == expected_value, \
            f"Opcode {opcode} should produce {expected_value}, got {vm.stack[0]}"
