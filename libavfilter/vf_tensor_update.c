/* libavfilter/vf_tensor_update.c */
#include "libavutil/opt.h"
#include "libavutil/tensor_abi.h"
#include "avfilter.h"
#include "filters.h"
#include "internal.h"
#include <math.h>

enum GateMode {
    GATE_ALWAYS = 0,
    GATE_THRESHOLD,
    GATE_DIRECTION,
};

typedef struct TensorUpdateContext {
    const AVClass *class;
    float learning_rate;
    int gate_mode;
    float gate_val; // Renamed to avoid 'threshold' collision
    uint64_t update_count;
    uint64_t skip_count;
    AVFrame *pending_w;
    AVFrame *pending_g;
} TensorUpdateContext;

#define OFFSET(x) offsetof(TensorUpdateContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption tensor_update_options[] = {
    { "lr", "Learning rate", OFFSET(learning_rate), AV_OPT_TYPE_FLOAT, { .dbl = 0.001 }, 0, 100, FLAGS },
    { "mode", "Logic gate mode", OFFSET(gate_mode), AV_OPT_TYPE_INT, { .i64 = GATE_THRESHOLD }, 0, 2, FLAGS, "mode" },
        { "always", "Always update", 0, AV_OPT_TYPE_CONST, { .i64 = GATE_ALWAYS }, 0, 0, FLAGS, "mode" },
        { "thresh", "Magnitude threshold", 0, AV_OPT_TYPE_CONST, { .i64 = GATE_THRESHOLD }, 0, 0, FLAGS, "mode" },
        { "dir",   "Directional activity", 0, AV_OPT_TYPE_CONST, { .i64 = GATE_DIRECTION }, 0, 0, FLAGS, "mode" },
    { "gate", "Gate value", OFFSET(gate_val), AV_OPT_TYPE_FLOAT, { .dbl = 0.01 }, 0, 100, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(tensor_update);

static int process_update(AVFilterContext *ctx, AVFrame *out, const AVFrame *w, const AVFrame *g) {
    TensorUpdateContext *s = ctx->priv;
    TensorHeader hw;
    memcpy(&hw, w->data[0], sizeof(hw)); 
    
    float *p_w = (float *)av_tensor_payload(w->data[0]);
    float *p_g = (float *)av_tensor_payload(g->data[0]);
    float *p_out = (float *)av_tensor_payload(out->data[0]);
    uint64_t n = hw.n_elem;

    int apply_update = 1;
    if (s->gate_mode == GATE_THRESHOLD) {
        double sum_sq = 0.0;
        for (uint64_t i = 0; i < n; i++) sum_sq += (double)p_g[i] * (double)p_g[i];
        if (sqrt(sum_sq) < (double)s->gate_val) apply_update = 0;
    }

    TensorHeader *h_out = (TensorHeader *)out->data[0];
    if (apply_update) {
        for (uint64_t i = 0; i < n; i++) p_out[i] = p_w[i] - (s->learning_rate * p_g[i]);
        h_out->generation_id = hw.generation_id + 1;
        s->update_count++;
    } else {
        memcpy(p_out, p_w, (size_t)hw.byte_size);
        h_out->generation_id = hw.generation_id;
        s->skip_count++;
    }
    return 0;
}

static int activate(AVFilterContext *ctx) {
    TensorUpdateContext *s = ctx->priv;
    AVFilterLink *wlink = ctx->inputs[0];
    AVFilterLink *glink = ctx->inputs[1];
    AVFilterLink *outlink = ctx->outputs[0];
    int ret;

    FF_FILTER_FORWARD_STATUS_BACK_ALL(outlink, ctx);

    // 1. Safe Consumption
    if (!s->pending_w) { ret = ff_inlink_consume_frame(wlink, &s->pending_w); if (ret < 0) return ret; }
    if (!s->pending_g) { ret = ff_inlink_consume_frame(glink, &s->pending_g); if (ret < 0) return ret; }

    // 2. Sync Barrier
    if (s->pending_w && s->pending_g) {
        TensorHeader *hw = (TensorHeader *)s->pending_w->data[0];
        
        AVFrame *out = av_frame_alloc();
        if (!out) return AVERROR(ENOMEM);
        
        // Critical: Propagate Timestamp
        av_frame_copy_props(out, s->pending_w);
        out->pts = s->pending_w->pts; 

        // Aligned Allocation
        out->buf[0] = av_buffer_alloc(sizeof(TensorHeader) + hw->byte_size);
        if (!out->buf[0]) { av_frame_free(&out); return AVERROR(ENOMEM); }
        out->data[0] = out->buf[0]->data;

        memcpy(out->data[0], s->pending_w->data[0], sizeof(TensorHeader));
        process_update(ctx, out, s->pending_w, s->pending_g);
        
        av_frame_free(&s->pending_w);
        av_frame_free(&s->pending_g);
        return ff_filter_frame(outlink, out);
    }

    // 3. Signal Logic
    if (ff_inlink_acknowledge_status(wlink, &ret, &s->pending_w)) {
        ff_outlink_set_status(outlink, ret, s->pending_w ? s->pending_w->pts : 0);
        return 0;
    }
    if (ff_inlink_acknowledge_status(glink, &ret, &s->pending_g)) {
        ff_outlink_set_status(outlink, ret, s->pending_g ? s->pending_g->pts : 0);
        return 0;
    }

    if (ff_outlink_frame_wanted(outlink)) {
        if (!s->pending_w) ff_inlink_request_frame(wlink);
        if (!s->pending_g) ff_inlink_request_frame(glink);
    }
    return 0;
}

static const AVFilterPad tensor_update_inputs[] = {
    { .name = "weights", .type = AVMEDIA_TYPE_VIDEO },
    { .name = "gradients", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};
static const AVFilterPad tensor_update_outputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};
const AVFilter ff_vf_tensor_update = {
    .name = "tensor_update",
    .priv_size = sizeof(TensorUpdateContext),
    .priv_class = &tensor_update_class,
    .activate = activate,
    .inputs = tensor_update_inputs,
    .outputs = tensor_update_outputs,
};
