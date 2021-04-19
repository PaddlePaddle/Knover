import paddle.distributed.fleet as fleet
import paddle.fluid.layers as layers
import paddle.fluid as fluid

fluid.default_main_program().block(0).re
# 用于将一个src_program的op赋值给另一个dst_program
def replace(src_program, dst_program, \
    src_block_id=0, dst_block_id=0, \
    src_block_start_op_idx=None, \
    src_block_end_op_idx=None, \
    dst_block_start_op_idx=None, \
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
        dst_blockl_end_op_idx = len(dst_ops)
    # for i, dst_op in enumerate(dst_ops):
    #     dst_block._remove_op(i)
    for i in range(src_block_start_op_idx, src_block_end_op_idx-1):
        dst_block._remove_op(i)
    
    for i, src_op in enumerate(src_ops):
        src_op_desc = src_op.desc
        dst_ap_op = dst_block.desc.append_op()
        dst_ap_op.copy_from(src_op_desc)
    dst_block._sync_with_cpp()

    dst_op_idx = 0
    dst_op_size = dst_block.desc.op_size()
    # while dst_op_idx < dst_op_size:

    with open("src_program.txt", "w") as f:
        f.write(str(src_program))
    with open("dst_program.txt", "w") as f:
        f.write(str(dst_program))
    return
    

        

