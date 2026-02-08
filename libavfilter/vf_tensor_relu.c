/* libavfilter/vf_tensor_relu.c */
#include "libavutil/tensor_abi.h"
#include "avfilter.h"
#include "internal.h"

static int filter_frame(AVFilterLink *inlink, AVFrame *in) {
    int ret = av_frame_make_writable(in);
    if (ret < 0) { av_frame_free(&in); return ret; }

    TensorHeader *h = (TensorHeader *)in->data[0];
    if (h->magic != TENSOR_MAGIC) {
        av_frame_free(&in);
        return AVERROR(EINVAL);
    }

    float *x = (float *)av_tensor_payload(in->data[0]);
    for (uint64_t i = 0; i < h->n_elem; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
    h->generation_id++;
    return ff_filter_frame(inlink->dst->outputs[0], in);
}

static const AVFilterPad tensor_relu_inputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO, .filter_frame = filter_frame },
    { NULL }
};
static const AVFilterPad tensor_relu_outputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};
const AVFilter ff_vf_tensor_relu = {
    .name = "tensor_relu",
    .inputs = tensor_relu_inputs,
    .outputs = tensor_relu_outputs,
};
