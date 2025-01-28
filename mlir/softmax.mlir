module {
  aie.device(npu1_1col) {
    func.func private @softmax_bf16_vector(memref<1024xbf16>, memref<1024xbf16>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @inA(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @outC(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @memA0(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @memC0(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @memA1(%tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @memC1(%tile_0_3, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@inA] -> [@memA0, @memA1]([] [0, 1024])
    aie.objectfifo.link [@memC0, @memC1] -> [@outC]([0, 1024] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @memA0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @softmax_bf16_vector(%3, %1) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @memA0(Consume, 1)
          aie.objectfifo.release @memC0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "bf16_softmax.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @memA1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @softmax_bf16_vector(%3, %1) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @memA1(Consume, 1)
          aie.objectfifo.release @memC1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "bf16_softmax.o"}
    aiex.runtime_sequence(%arg0: memref<262144xbf16>, %arg1: memref<262144xbf16>) {
      %0 = aiex.dma_configure_task_for @inA {
        aie.dma_bd(%arg0 : memref<262144xbf16>, 0, 262144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 262144, stride = 1>])
        aie.end
      } {issue_token = true}
      %1 = aiex.dma_configure_task_for @outC {
        aie.dma_bd(%arg1 : memref<262144xbf16>, 0, 262144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 262144, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
    }
  }
}

