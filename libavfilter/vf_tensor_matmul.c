/* libavfilter/vf_tensor_matmul.c */
#include "libavutil/tensor_abi.h"
#include "avfilter.h"
#include "filters.h"
#include "internal.h"

typedef struct TensorMatMulContext {
    const AVClass *class;
    AVFrame *pending_a;
    AVFrame *pending_b;
} TensorMatMulContext;

static int execute_matmul(AVFilterContext *ctx, AVFrame *out, const AVFrame *a, const AVFrame *b, 
                          uint64_t m, uint64_t n, uint64_t p) {
    const float *A = (const float *)av_tensor_payload(a->data[0]);
    const float *B = (const float *)av_tensor_payload(b->data[0]);
    float *C = (float *)av_tensor_payload(out->data[0]);

    // Naive O(n^3) - Replace with CBLAS for production speed
    for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = 0; j < p; j++) {
            double sum = 0.0; // Use double accumulator for precision
            for (uint64_t k = 0; k < n; k++) sum += (double)A[i*n + k] * (double)B[k*p + j];
            C[i*p + j] = (float)sum;
        }
    }

    TensorHeader *ho = (TensorHeader *)out->data[0];
    ho->magic = TENSOR_MAGIC;
    ho->n_dims = 2;
    ho->n_elem = m * p;
    ho->shape[0] = m;
    ho->shape[1] = p;
    ho->byte_size = ho->n_elem * sizeof(float);
    return 0;
}

static int activate(AVFilterContext *ctx) {
    TensorMatMulContext *s = ctx->priv;
    AVFilterLink *alink = ctx->inputs[0];
    AVFilterLink *blink = ctx->inputs[1];
    AVFilterLink *outlink = ctx->outputs[0];
    int ret;

    FF_FILTER_FORWARD_STATUS_BACK_ALL(outlink, ctx);

    if (!s->pending_a) { ret = ff_inlink_consume_frame(alink, &s->pending_a); if(ret < 0) return ret; }
    if (!s->pending_b) { ret = ff_inlink_consume_frame(blink, &s->pending_b); if(ret < 0) return ret; }

    if (s->pending_a && s->pending_b) {
        // SAFETY: Check Magic
        TensorHeader *ha = (TensorHeader *)s->pending_a->data[0];
        TensorHeader *hb = (TensorHeader *)s->pending_b->data[0];
        if (ha->magic != TENSOR_MAGIC || hb->magic != TENSOR_MAGIC) {
            return AVERROR(EINVAL);
        }

        // SAFETY: Check Rank
        if (ha->n_dims != 2 || hb->n_dims != 2) return AVERROR(EINVAL);

        uint64_t m = ha->shape[0];
        uint64_t n = ha->shape[1];
        uint64_t n_check = hb->shape[0];
        uint64_t p = hb->shape[1];

        if (n != n_check) return AVERROR(EINVAL);

        AVFrame *out = av_frame_alloc();
        if (!out) return AVERROR(ENOMEM);
        
        av_frame_copy_props(out, s->pending_a);
        out->pts = FFMAX(s->pending_a->pts, s->pending_b->pts); // Sync PTS

        size_t payload_size = m * p * sizeof(float);
        out->buf[0] = av_buffer_alloc(sizeof(TensorHeader) + payload_size);
        if (!out->buf[0]) { av_frame_free(&out); return AVERROR(ENOMEM); }
        out->data[0] = out->buf[0]->data;

        execute_matmul(ctx, out, s->pending_a, s->pending_b, m, n, p);
        
        av_frame_free(&s->pending_a);
        av_frame_free(&s->pending_b);
        return ff_filter_frame(outlink, out);
    }
    
    if (ff_outlink_frame_wanted(outlink)) {
        if (!s->pending_a) ff_inlink_request_frame(alink);
        if (!s->pending_b) ff_inlink_request_frame(blink);
    }
    return 0;
}
// Pads/Struct same as Update...
static const AVFilterPad tensor_matmul_inputs[] = {
    { .name = "a", .type = AVMEDIA_TYPE_VIDEO },
    { .name = "b", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};
static const AVFilterPad tensor_matmul_outputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};
const AVFilter ff_vf_tensor_matmul = {
    .name = "tensor_matmul",
    .priv_size = sizeof(TensorMatMulContext),
    .activate = activate,
    .inputs = tensor_matmul_inputs,
    .outputs = tensor_matmul_outputs,
};
