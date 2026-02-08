/* libavutil/tensor_abi.h
 * STRICT ABI for Tensor Payloads within AVFrame.data[0]
 * Alignment: 128-byte header ensures payload is 64-byte aligned.
 */
#ifndef AVUTIL_TENSOR_ABI_H
#define AVUTIL_TENSOR_ABI_H

#include <stdint.h>
#include "libavutil/attributes.h"

#define TENSOR_MAGIC 0x544E5352u /* "TNSR" */

typedef struct TensorHeader {
    uint32_t magic;           // 4 bytes
    uint32_t dtype;           // 4 bytes
    uint32_t n_dims;          // 4 bytes (Explicit Rank)
    uint32_t flags;           // 4 bytes (Reserved)
    
    uint64_t n_elem;          // 8 bytes
    uint64_t byte_size;       // 8 bytes
    uint64_t generation_id;   // 8 bytes
    
    uint64_t shape[8];        // 64 bytes (8 dims * 8 bytes)
    
    uint8_t  padding[24];     // PADDING: Rounds struct size to exactly 128 bytes
} TensorHeader;

// Static assertion to enforce alignment contract at compile time
_Static_assert(sizeof(TensorHeader) == 128, "TensorHeader ABI violation: Must be 128 bytes");

static inline uint8_t* av_tensor_payload(uint8_t *data) {
    return data + sizeof(TensorHeader);
}

#endif /* AVUTIL_TENSOR_ABI_H */
