import paddle.distributed.fleet as fleet
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import compat as cpt
fluid.framework
used_var_set = set()
def op_inputs(op, src_block, dst_block):
    global used_var_set
    inputs = {}
    for i in range(0, len(op.input_names)):
        param_name = op.input_names[i]
        inputs[param_name] = []
        var_names = op.input(param_name)
        for var_name in var_names:
            #if not dst_block.has_var(var_name):
            #    create_var([var_name], src_block, dst_block)
            #    print(var_name, "is created")
            if dst_block.has_var(var_name) and var_name not in used_var_set:
                print("remove_var:", var_name)
                dst_block._remove_var(var_name)
            if not dst_block.has_var(var_name):
                print("create_var:", var_name)
                create_var([var_name], src_block, dst_block)
            # print(var_name, "is created")
            val = dst_block.var(var_name)
            # print("var_shape:", val.shape)
            inputs[param_name].append(val)
    # print(inputs)
    return inputs

def op_outputs(op, src_block, dst_block):
    global used_var_set
    outputs = {}
    for i in range(0, len(op.output_names)):
        param_name = op.output_names[i]
        outputs[param_name] = []
        var_names = op.output(param_name)
        for var_name in var_names:
            #if not dst_block.has_var(var_name):
            #    create_var([var_name], src_block, dst_block)
            #    print(var_name, "is created")
            if dst_block.has_var(var_name) and var_name not in used_var_set:
                print("remove_var:", var_name)
                dst_block._remove_var(var_name)
            if not dst_block.has_var(var_name):
                print("create_var:", var_name)
                create_var([var_name], src_block, dst_block)
            val = dst_block.var(var_name)
            outputs[param_name].append(val)

    return outputs

def create_var(vars, src_block, dst_block, should_rename=False):
    for var in vars:
        # print(var, "in used_var_set", "True" if var in used_var_set else "False")
        if var in used_var_set or "_blocking_queue" in var:continue
        used_var_set.add(var)
        if dst_block.has_var(var):continue
        source_var = src_block._var_recursive(var)
        if source_var.type == core.VarDesc.VarType.READER:
            dst_var= dst_block.create_var(name=var, type=core.VarDesc.VarType.READER, persistable=source_var.persistable)
        else:
            dst_var = dst_block._clone_variable(source_var, False)
        if should_rename:
            for op in src_block.ops:
                new_var_name = var + ".tmp"
                print("rename {} to {}".format(var, new_var_name))
                src_input_vars = op.desc.input_arg_names()
                dst_input_vars = op.desc.output_arg_names()
                all_vars = src_input_vars + dst_input_vars
                if var in all_vars:
                    op._rename_input(var, new_var_name)
                    op._rename_output(var, new_var_name)
        # print("created")
        dst_var.stop_gradient = source_var.stop_gradient


def find_var(block):
    varnames = block.vars
    for var_name in varnames:
        if var_name.startswith("_generated_var"):
            continue
        if block.has_var(var_name):
            if list(block.var(var_name).shape) == [-1, 16, 1, 2]:
                print(var_name, "+++++++++++++++++++++++", list(block.var(var_name).shape))


# 用于将一个src_program的op赋值给另一个dst_program
def replace(src_program, dst_program, \
    src_block_id=0, dst_block_id=1, \
    src_block_start_op_idx=12, \
    src_block_end_op_idx=None, \
    dst_block_start_op_idx=None, \
    dst_block_end_op_idx=None):
    # global used_var_set
    src_block = src_program.block(src_block_id)
    dst_block = dst_program.block(dst_block_id)
    src_ops = src_block.ops
    dst_ops = dst_block.ops
    if src_block_start_op_idx is None:
        src_block_start_op_idx = 0
    if src_block_end_op_idx is None:
        src_block_end_op_idx = len(src_ops)
    if dst_block_start_op_idx is None:
        dst_block_start_op_idx = 0
    if dst_block_end_op_idx is None:
        dst_block_end_op_idx = len(dst_ops)
    import pdb
    pdb.set_trace()
    with open("test1.txt", "w") as f:
        f.write(str(src_program))
    
    # find_var(src_block)
    # find_var(src_block)
    # find_var(dst_block)
    print("&&&&&&&&&&&&&&&&&&&&&&&&")
    # if src_block.var("matmul_0.tmp_0").shape == (-1, 16, 1, 2):
    #     import pdb
    #     pdb.set_trace()
    # 修改src_block的名字
    dst_op_pointer = dst_block_start_op_idx
    op_count = 0
    for i in range(src_block_start_op_idx, src_block_end_op_idx):
        src_op = src_ops[i]
        # print(str(i), src_op.type)
        if str(src_op.type) in ["c_sync_calc_stream", "c_sync_comm_stream", "c_broadcast"]:
            continue       
        dst_op = dst_ops[dst_op_pointer]
        if src_op.type == "fill_constant" and "BroadCast" in src_op.desc.output_arg_names()[0]:
            continue
        #FIXME:
        if src_op.type != dst_op.type:           
            break
        assert src_op.type == dst_op.type, "src_op: {}, dst_op: {}, src_i: {}, dst_i: {}".format(src_op.type, dst_op.type, str(i), str(dst_op_pointer))
        src_input_vars = src_op.desc.input_arg_names()
        src_output_vars = src_op.desc.output_arg_names()
        dst_input_vars = dst_op.desc.input_arg_names()
        dst_output_vars = dst_op.desc.output_arg_names()
        
        for (src_var, dst_var) in zip(src_input_vars, dst_input_vars):
            # if str(dst_op_pointer) == "402":
            #     print(src_var, dst_var)
            #     print(src_block.has_var(src_var))
            if not src_block.has_var(dst_var):
                # 因为修改的是src
                create_var([dst_var], dst_block, src_block, should_rename=False)
            else:
                create_var([dst_var], dst_block, src_block, should_rename=True)
            src_op._rename_input(src_var, dst_var)
        
        for (src_var, dst_var) in zip(src_output_vars, dst_output_vars):
            if not src_block.has_var(dst_var):
                create_var([dst_var], dst_block, src_block, should_rename=False)
                # 因为修改的是src
            else:
                create_var([dst_var], dst_block, src_block, should_rename=True)
            src_op._rename_output(src_var, dst_var)
        
        dst_op_pointer += 1
        op_count += 1
    global used_var_set
    used_var_set = set()
    # find_var(src_block)
    # find_var(dst_block)
    import pdb
    pdb.set_trace()       
    with open("src_program_org.txt", "w") as f:
        f.write(str(src_program))
    with open("dst_program_org.txt", "w") as f:
        f.write(str(dst_program))

    for i in range(dst_op_pointer-1, dst_block_start_op_idx-1, -1):
        dst_block._remove_op(i)
    find_var(src_block)
    print("))))))))00000000000")
    print("dst_block")
    find_var(dst_block)
    print("+++++++++++++++++++++++++")

    for i in range(src_block_start_op_idx, src_block_start_op_idx+op_count-5):
        # print(i)
        # if i == 56:
        #     import pdb
        #     pdb.set_trace()
        src_op = src_ops[i]
        # print(src_op)
        find_var(dst_block)
        dst_block._insert_op(dst_block_start_op_idx, 
        type=src_op.type, 
        attrs=src_op.all_attrs(), 
        inputs=op_inputs(src_op, src_block, dst_block), 
        outputs=op_outputs(src_op, src_block, dst_block))
        dst_block_start_op_idx += 1
    
    dst_block._sync_with_cpp()
    # 删除dst_block里面所有没有被用到的var
    vars = list(dst_block.vars.keys())
    for var in vars:
        if var not in used_var_set:
            dst_block._remove_var(var)

    with open("src_program.txt", "w") as f:
        f.write(str(src_program))
    with open("dst_program.txt", "w") as f:
        f.write(str(dst_program))
    
    return
    

        

