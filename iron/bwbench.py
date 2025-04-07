import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_


def my_matmul(arch: str = "npu2"):

    if arch == "npu1":
        dev = AIEDevice.npu1_4col
        total_cols = 4
        total_rows = 4
        use_cols = 4
        use_rows = 1
    elif arch == "npu2":
        dev = AIEDevice.npu2
        total_cols = 8
        total_rows = 4
        use_cols = 4
        use_rows = 1
    else:
        raise ValueError(f"Invalid device: {arch}")


    dtype = np.dtype[np.uint32]

    with mlir_mod_ctx() as ctx:
        @device(dev)
        def device_body():
            # Tile declarations
            ShimTiles = []
            MemTiles = []
            cores = []
            for col in range(use_cols):
                ShimTiles.append(tile(col, 0))
                MemTiles.append(tile(col, 1))
                ComputeTilesRow = []
                for row in range(use_rows):
                    ComputeTilesRow.append(tile(col, row + 2))

                cores.append(ComputeTilesRow)

            token_rate = 32
            burst_size = 1024
            data_slice_ty = np.ndarray[(burst_size // use_rows, 1), dtype] # 1KB
            data_ty = np.ndarray[(burst_size, ), dtype] # 1KB


            # Input A
            channel_0_it_fifos = []
            channel_1_it_fifos = []
            for col in range(use_cols):
                channel_0_it_fifos.append(object_fifo(f"channel_0_it_fifos{col}", ShimTiles[col], MemTiles[col], 2, data_ty))
                channel_1_it_fifos.append(object_fifo(f"channel_1_it_fifos{col}", ShimTiles[col], MemTiles[col], 2, data_ty))

            broadcast_0_fifos = []
            broadcast_1_fifos = []
            for col in range(use_cols):
                core_list = []
                for row in range(use_rows):
                    core_list.append(cores[col][row])
                broadcast_0_fifos.append(object_fifo(f"broadcast_0_fifos{col}", MemTiles[col], core_list, 2, data_ty))
                broadcast_1_fifos.append(object_fifo(f"broadcast_1_fifos{col}", MemTiles[col], core_list, 2, data_ty))

            # link fifos
            for col in range(use_cols):
                object_fifo_link(channel_0_it_fifos[col], broadcast_0_fifos[col])
                object_fifo_link(channel_1_it_fifos[col], broadcast_1_fifos[col])

            # output token fifo
            # Output C
            token_fifos = []
            for col in range(use_cols):
                fifos = []
                for row in range(use_rows):
                    fifos.append(object_fifo(f"token_out_fifos_{col}{row}", cores[col][row], MemTiles[col], 2, data_slice_ty))
                token_fifos.append(fifos)
 
            token_out_fifos = []
            for col in range(use_cols):
                token_out_fifos.append(object_fifo(f"token_out_fifos_{col}", MemTiles[col], ShimTiles[col], 2, data_ty))
 
            for col in range(use_cols):
                offsets = []
                for row in range(use_rows):
                    offsets.append(row * burst_size // use_rows)
                object_fifo_link(token_fifos[col], token_out_fifos[col], offsets, [])

            # Set up compute tiles
            for col in range(use_cols):
                for row in range(use_rows):
                    @core(cores[col][row])
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            # passthrough
                            token = token_fifos[col][row].acquire(ObjectFifoPort.Produce, 1)
                            for _ in range_(token_rate):
                                elem_in_a = broadcast_0_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_b = broadcast_1_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                                broadcast_0_fifos[col].release(ObjectFifoPort.Consume, 1)
                                broadcast_1_fifos[col].release(ObjectFifoPort.Consume, 1)
                            token_fifos[col][row].release(ObjectFifoPort.Produce, 1)


            # To/from AIE-array data movement
            @runtime_sequence(
                np.ndarray[(use_cols * burst_size, token_rate * 2), dtype],
                np.ndarray[(use_cols * burst_size,), dtype],
                np.ndarray[(16384 // 4,), dtype]
            )
            def sequence(A, C, T):
                rounds = 4
                for r in range(rounds // 2):
                    for pp in range(2):
                        bd_offset = pp * 8
                        abs_rr = r * 2 + pp
                        for i in range(use_cols):
                            col_offset =  0# i * burst_size * token_rate // use_cols
                            npu_dma_memcpy_nd(
                                metadata=channel_0_it_fifos[i],
                                bd_id=1 + bd_offset,
                                mem=A,
                                offsets=[0, 0, 0, col_offset],
                                sizes=[1, 1, 1, token_rate * burst_size],
                                strides=[0, 0, 0, 1],
                            )
                            npu_dma_memcpy_nd(
                                metadata=channel_1_it_fifos[i],
                                bd_id=2 + bd_offset,
                                mem=A,
                                offsets=[0, 0, 0, col_offset + 0 * token_rate * burst_size * 2],
                                sizes=[1, 1, 1, token_rate * burst_size],
                                strides=[0, 0, 0, 1],
                            )
                            npu_dma_memcpy_nd(
                                metadata=token_out_fifos[i],
                                bd_id=0 + bd_offset,
                                mem=C,
                                offsets=[0, 0, 0, i * burst_size],
                                sizes = [1, 1, 1, burst_size],
                                strides = [0, 0, 0, 1]
                            )
                        if abs_rr > 0:
                            dma_wait(*token_out_fifos)
                dma_wait(*token_out_fifos)

    print(ctx.module)


my_matmul(sys.argv[1])
