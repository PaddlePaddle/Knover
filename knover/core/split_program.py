import paddle.distributed.fleet as fleet
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import paddle.fluid.core as core

# 取出src_block里的op所有vars，dst_block里面的var如果不在vars就去除该op
def clean_redundancy(src_block, dst_block):
    effective_var = []
    for i in range(src_block.num_blocks):
        block = src_block.block(i)
        for op in block.ops:
            var_names= op.desc.input_arg_names() + op.desc.output_arg_names()
            effective_var.extend(var_names)
    for i in range(dst_block.num_blocks):
        block = dst_block.block(i)   
        for indx, op in reversed(list(enumerate(block.ops))):
            if op.type in ["c_gen_nccl_id", "c_comm_init"]:
                continue
            var_names = op.desc.input_arg_names() + op.desc.output_arg_names()
            for var_name in var_names:
                if var_name not in effective_var:
                    block._remove_op(indx)

def find_op_idx(block, var_name, as_input=False):
    for idx, op in enumerate(list(block.ops)):
        in_names = op.desc.input_arg_names()
        out_names = op.desc.output_arg_names()
        if as_input and var_name in in_names: return idx
        if not as_input and var_name in out_names: return idx
    raise ValueError("Cannot find a op in block which takes {} as input or output.".format(var_name))

def op_inputs(op, src_block, dst_block):
    inputs = {}
    for i in range(0, len(op.input_names)):
        param_name = op.input_names[i]
        inputs[param_name] = []
        var_names = op.input(param_name)
        for var_name in var_names:
            if not dst_block._find_var_recursive(var_name):
                create_var([var_name], src_block, dst_block)
            val = dst_block._var_recursive(var_name)
            inputs[param_name].append(val)
    return inputs

def op_outputs(op, src_block, dst_block):
    outputs = {}
    for i in range(0, len(op.output_names)):
        param_name = op.output_names[i]
        outputs[param_name] = []
        var_names = op.output(param_name)
        for var_name in var_names:
            if not dst_block._find_var_recursive(var_name):
                create_var([var_name], src_block, dst_block)
            val = dst_block._var_recursive(var_name)
            outputs[param_name].append(val)

    return outputs

def create_var(vars, src_block, dst_block, should_rename=False):
    for var in vars:
        source_var = src_block._var_recursive(var)
        if source_var.type == core.VarDesc.VarType.READER:
            dst_var= dst_block.create_var(name=var, type=core.VarDesc.VarType.READER, persistable=source_var.persistable)
        else:
            if should_rename:
                new_var_name = var + ".tmp"
                dst_block._rename_var(var, new_var_name)
            dst_var = dst_block.create_var(
                name=var,
                shape=source_var.shape,
                dtype=source_var.dtype,
                type=source_var.type,
                lod_level=source_var.lod_level,
                persistable=source_var.persistable,
                is_data=source_var.is_data,
                need_check_feed=source_var.desc.need_check_feed())
        if should_rename:
            for op in dst_block.ops:
                input_names = op.desc.input_arg_names()
                output_names = op.desc.output_arg_names()
                inout_names = input_names + output_names
                if var in inout_names:
                    op._rename_input(var, new_var_name)
                    op._rename_output(var, new_var_name)
        dst_var.stop_gradient = source_var.stop_gradient



def replace(src_program,
            dst_program,
            src_block_id=0,
            dst_block_id=1,
            src_block_start_op_idx=12,
            src_block_end_op_idx=100,
            dst_block_start_op_idx=None,
            dst_block_end_op_idx=None):
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
    
    dst_op_idx = dst_block_start_op_idx
    copied_op_num = 0
    for i in range(src_block_start_op_idx, src_block_end_op_idx):
        src_op = src_block.ops[i]
        if src_op.type in ["c_sync_calc_stream",
                           "c_sync_comm_stream",
                           "c_broadcast"] or (
                src_op.type == "fill_constant" and
                "BroadCast" in src_op.desc.output_arg_names()[0]):
            copied_op_num += 1
            dst_block._insert_op(
                index=dst_op_idx,
                type=src_op.type,
                attrs=src_op.all_attrs(),
                inputs=op_inputs(src_op, src_block, dst_block),
                outputs=op_outputs(src_op, src_block, dst_block))
            dst_op_idx += 1
            continue 
        dst_op = dst_block.ops[dst_op_idx]
        #FIXME:
        if src_op.type != dst_op.type:           
            break
        assert src_op.type == dst_op.type, ("src_op {} does not match dst_op:"
                " {}".format(src_op.type, dst_op.type))
        dst_op_idx += 1

    
    return
