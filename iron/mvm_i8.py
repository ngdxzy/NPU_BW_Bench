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
        mvm_rows = 1
        mvm_cols = 1
    elif arch == "npu2":
        dev = AIEDevice.npu2
        total_cols = 8
        total_rows = 4
        mvm_rows = 1
        mvm_cols = 1
    else:
        raise ValueError(f"Invalid device: {arch}")

    M = 128 * mvm_cols * mvm_rows * 4
    K = 128 * 4
    m = 128
    k = 128

    n_cores = mvm_rows * mvm_cols
    K_div_k = K // k
    m_x_k = m * k
    m_x_K = m * K

    dtype_in = np.dtype[np.int8]
    dtype_out = np.dtype[np.int32]

    if (M // (mvm_rows * mvm_cols * m)) * (mvm_rows * mvm_cols * m) != M:
        raise ValueError(f"M is not divisible by mvm_rows * mvm_cols * m")

    with mlir_mod_ctx() as ctx:
        @device(dev)
        def device_body():
            inA_ty = np.ndarray[(mvm_rows * m * k,), dtype_in]
            A_ty = np.ndarray[(m, k), dtype_in]
            inB_ty = np.ndarray[(k,), dtype_in]
            outC_ty = np.ndarray[(mvm_rows * m,), dtype_out]
            C_ty = np.ndarray[(m, ), dtype_out]

            # AIE Core Function declarations
            zero = external_func("zero_m_int8", inputs=[C_ty])
            matvec = external_func(
                "mv_int8",
                inputs=[A_ty, inB_ty, C_ty],
            )

            # Tile declarations
            ShimTiles = []
            MemTiles = []
            cores = []
            for col in range(mvm_cols):
                ShimTiles.append(tile(col, 0))
                MemTiles.append(tile(col, 1))
                ComputeTilesRow = []
                for row in range(mvm_rows):
                    ComputeTilesRow.append(tile(col, row + 2))

                cores.append(ComputeTilesRow)

            B_ShimTile = ShimTiles[0]
            B_MemTile = MemTiles[0]

            # Input A
            memA_fifos = []
            for col in range(mvm_cols):
                memA_fifos.append(object_fifo(f"memA{col}", ShimTiles[col], MemTiles[col], 2, inA_ty))

            inA_fifos = []
            for col in range(mvm_cols):
                fifos = []
                for row in range(mvm_rows):
                    fifos.append(object_fifo(f"inA{col}{row}", MemTiles[col], cores[col][row], 2, A_ty, ([(k // 4, 4), (m, k), (4, 1)])))
                inA_fifos.append(fifos)
                del fifos

            for col in range(mvm_cols):
                offsets = []
                for row in range(mvm_rows):
                    offsets.append(row * m_x_k)
                object_fifo_link(memA_fifos[col], [*inA_fifos[col]], [], [*offsets])

            # Output C
            outC_fifos = []
            for col in range(mvm_cols):
                fifos = []
                for row in range(mvm_rows):
                    fifos.append(object_fifo(f"outC{col}{row}", cores[col][row], MemTiles[col], 2, C_ty))
                outC_fifos.append(fifos)
                del  fifos

            memC_fifos = []
            for col in range(mvm_cols):
                memC_fifos.append(object_fifo(f"memC{col}", MemTiles[col], ShimTiles[col], 2, outC_ty))

            for col in range(mvm_cols):
                offsets = []
                for row in range(mvm_rows):
                    offsets.append(row * m)
                object_fifo_link([*outC_fifos[col]], memC_fifos[col], [*offsets], [])

            # Input B
            memB_fifo = object_fifo("memB", B_ShimTile, B_MemTile, 2, inB_ty)
            core_list = []
            for col in range(mvm_cols):
                for row in range(mvm_rows):
                    core_list.append(cores[col][row])
            inB_fifo = object_fifo(f"inB", B_MemTile, core_list, 2, inB_ty)
            object_fifo_link(memB_fifo, inB_fifo)

            # Set up compute tiles
            for col in range(mvm_cols):
                for row in range(mvm_rows):
                    @core(cores[col][row], f"mvm_i8.o")
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            elem_out = outC_fifos[col][row].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)

                            for _ in range_(K_div_k):
                                elem_in_a = inA_fifos[col][row].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                                matvec(elem_in_a, elem_in_b, elem_out)
                                inA_fifos[col][row].release(ObjectFifoPort.Consume, 1)
                                inB_fifo.release(ObjectFifoPort.Consume, 1)

                            outC_fifos[col][row].release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(
                np.ndarray[(M*K,), dtype_in],
                np.ndarray[(K,), dtype_in],
                np.ndarray[(M,), dtype_out],
            )
            def sequence(A, B, C):
                r = M // m // n_cores
                assert r * m * n_cores == M
                npu_dma_memcpy_nd(
                    metadata=memB_fifo,
                    bd_id=2,
                    mem=B,
                    offsets=[0, 0, 0, 0],
                    sizes=[M // m // n_cores, 1, 1, K],
                    strides=[0, 0, 0, 1],
                )
                for i in range(mvm_cols):
                    # M offset: each column handles M // mvm_cols row in total
                    A_offset = i * M * K // mvm_cols
                    C_offset = i * M // mvm_cols
                    npu_dma_memcpy_nd(
                        metadata=memA_fifos[i],
                        bd_id=1,
                        mem=A,
                        offsets=[0, 0, 0, A_offset],
                        sizes=[M // mvm_cols // (m * mvm_rows), K_div_k, mvm_rows * m, k],
                        strides=[m_x_K * mvm_rows, k, K, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=memC_fifos[i],
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, M // m // mvm_cols // mvm_rows, mvm_rows * m],
                        strides=[0, 0, mvm_rows * m, 1],
                    )
                dma_wait(*memC_fifos)

    print(ctx.module)


my_matmul(sys.argv[1])
