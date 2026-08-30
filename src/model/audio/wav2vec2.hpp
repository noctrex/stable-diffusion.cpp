#ifndef __SD_MODEL_AUDIO_WAV2VEC2_HPP__
#define __SD_MODEL_AUDIO_WAV2VEC2_HPP__

#include <cinttypes>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "model/common/ggml_block.hpp"
#include "model.h"

/*================================================ Wav2Vec2 audio encoder ================================================*/

// Port of ComfyUI comfy/audio_encoders/wav2vec2.py (HF wav2vec2 inference subset).
// Checkpoint: wav2vec2-large-english (embed_dim 1024, 24 layers, 16 heads,
// conv_norm/conv_bias true, stable layer norm). Keys carry a "wav2vec2." prefix;
// load the storage map with that prefix so names line up with the block tree.

struct Wav2Vec2Config {
    int64_t embed_dim         = 1024;
    int64_t conv_dim          = 512;
    int num_heads             = 16;
    int num_layers            = 24;
    bool conv_norm            = true;
    bool conv_bias            = true;
    bool do_normalize         = true;
    bool do_stable_layer_norm = true;

    // Mirrors comfy/audio_encoders/audio_encoders.py:49-76: embed_dim inferred from
    // the shape of encoder.layer_norm.bias.
    static Wav2Vec2Config detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
        Wav2Vec2Config config;
        auto it = tensor_storage_map.find(prefix + "encoder.layer_norm.bias");
        if (it == tensor_storage_map.end()) {
            LOG_WARN("wav2vec2: %sencoder.layer_norm.bias not found, using large defaults", prefix.c_str());
            return config;
        }
        config.embed_dim = it->second.ne[0];
        if (config.embed_dim == 1024) {  // large
            config.embed_dim            = 1024;
            config.num_heads            = 16;
            config.num_layers           = 24;
            config.conv_norm            = true;
            config.conv_bias            = true;
            config.do_normalize         = true;
            config.do_stable_layer_norm = true;
        } else if (config.embed_dim == 768) {  // base
            config.embed_dim            = 768;
            config.num_heads            = 12;
            config.num_layers           = 12;
            config.conv_norm            = false;
            config.conv_bias            = false;
            config.do_normalize         = false;
            config.do_stable_layer_norm = false;
        } else {
            LOG_WARN("wav2vec2: unsupported embed_dim %" PRId64 ", using large defaults", config.embed_dim);
            config.embed_dim = 1024;
        }
        return config;
    }
};

// Conv1d with optional groups (weight [kernel, in_channels, out_channels]). For
// groups == channels == out_channels this stays depthwise-capable via ggml; general
// groups run per-group ggml_conv_1d on channel views and concat.
struct Wav2Vec2Conv1d : public UnaryBlock {
    int64_t in_channels;
    int64_t out_channels;
    int64_t groups;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool bias;
    std::string prefix;

    Wav2Vec2Conv1d(int64_t in_channels,
                   int64_t out_channels,
                   int kernel_size,
                   int stride     = 1,
                   int padding    = 0,
                   int dilation   = 1,
                   int64_t groups = 1,
                   bool bias      = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          groups(groups),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix     = prefix;
        ggml_type wtype  = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16);
        params["weight"] = ggml_new_tensor_3d(ctx, wtype, kernel_size, in_channels / groups, out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [L, in_channels, N]
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = bias ? params["bias"] : nullptr;

        GGML_ASSERT(x->ne[1] == in_channels);

        if (groups == 1) {
            // F32 im2col + F32 weight keeps the conv core in full precision (see
            // conv_1d_f32_im2col); mul_mat's mixed-type path requires an F32
            // right-hand side, so cast F16 checkpoint weights once.
            ggml_tensor* w32 = w->type == GGML_TYPE_F32 ? w : ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F32);
            ggml_tensor* out = conv_1d_f32_im2col(ctx->ggml_ctx, w32, x, stride, padding, dilation);
            if (b != nullptr) {
                b   = ggml_reshape_3d(ctx->ggml_ctx, b, 1, out_channels, 1);
                out = ggml_add_inplace(ctx->ggml_ctx, out, b);
            }
            return out;
        }
        return grouped_conv_1d(ctx, x, w, groups, stride, padding, b);
    }

    // Grouped conv via per-group channel views; output channels concatenated in order.
    // ggml_conv_1d hardcodes an F16 im2col destination, which rounds the input
    // patches even for F32 weights; wav2vec2 carries large activation outliers
    // that this rounding amplifies through 24 layers, so the grouped path uses
    // ggml_conv_1d's exact composition with an F32 im2col instead (the pos_conv
    // weight is always F32 here - it is recomputed in-graph from weight_norm).
    static ggml_tensor* conv_1d_f32_im2col(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int s0, int p0, int d0) {
        ggml_tensor* im2col = ggml_im2col(ctx, w, x, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F32);
        ggml_tensor* result = ggml_mul_mat(ctx,
                                           ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                                           ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
        return ggml_reshape_3d(ctx, result, im2col->ne[1], w->ne[2], im2col->ne[2]);
    }

    static ggml_tensor* grouped_conv_1d(GGMLRunnerContext* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* w,
                                        int64_t groups,
                                        int stride,
                                        int padding,
                                        ggml_tensor* b = nullptr) {
        const int64_t ic_g = x->ne[1] / groups;
        const int64_t oc_g = w->ne[2] / groups;
        // ggml's im2col kernels (the backend of ggml_conv_1d) read input planes
        // with flat indexing and only honor nb[1] for the channel offset. A
        // non-contiguous x (e.g. a permuted view with time stride 4*C) therefore
        // reads wrong elements through the per-group views below, because
        // ggml_view_3d forces nb[0] to the type size and hides the true stride.
        // Materialize once; contiguous inputs pass through untouched.
        if (x->nb[0] != ggml_type_size(x->type) || x->nb[1] != (size_t)x->ne[0] * x->nb[0]) {
            x = ggml_cont(ctx->ggml_ctx, x);
        }
        if (getenv("WAV2VEC2_DEBUG_CONV") != nullptr) {
            printf("[dbg] x: ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu]\n",
                   (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2],
                   x->nb[0], x->nb[1], x->nb[2]);
            printf("[dbg] w: ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu] type=%d\n",
                   (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2],
                   w->nb[0], w->nb[1], w->nb[2], (int)w->type);
        }
        ggml_tensor* acc = nullptr;
        for (int64_t i = 0; i < groups; ++i) {
            ggml_tensor* x_i   = ggml_view_3d(ctx->ggml_ctx, x,
                                              x->ne[0], ic_g, x->ne[2],
                                              x->nb[1], x->nb[2],
                                              i * ic_g * x->nb[1]);
            ggml_tensor* w_i   = ggml_view_3d(ctx->ggml_ctx, w,
                                              w->ne[0], ic_g, oc_g,
                                              w->nb[1], w->nb[2],
                                              i * oc_g * w->nb[2]);
            ggml_tensor* out_i = conv_1d_f32_im2col(ctx->ggml_ctx, w_i, x_i, stride, padding, 1);
            if (b != nullptr) {
                ggml_tensor* b_i = ggml_view_1d(ctx->ggml_ctx, b, oc_g, i * oc_g * b->nb[0]);
                b_i              = ggml_reshape_3d(ctx->ggml_ctx, b_i, 1, oc_g, 1);
                out_i            = ggml_add_inplace(ctx->ggml_ctx, out_i, b_i);
            }
            acc = (acc == nullptr) ? out_i : ggml_concat(ctx->ggml_ctx, acc, out_i, 1);
        }
        return acc;
    }
};

// Conv + (optional) per-channel norm + GELU. ComfyUI LayerNormConv applies the
// LayerNorm over channels for every frame; equivalent to normalizing over the
// channel axis after transposing. GroupNorm variant mirrors LayerGroupNormConv.
struct Wav2Vec2ConvLayer : public UnaryBlock {
    Wav2Vec2ConvLayer(int64_t in_channels,
                      int64_t out_channels,
                      int kernel_size,
                      int stride,
                      bool bias,
                      bool use_layer_norm) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Wav2Vec2Conv1d(in_channels, out_channels, kernel_size, stride, 0, 1, 1, bias));
        if (use_layer_norm) {
            blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(out_channels));
        }
        use_layer_norm_ = use_layer_norm;
        channels_       = out_channels;
    }

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
        if (!use_layer_norm_) {
            // LayerGroupNormConv: GroupNorm(num_groups=out_channels, affine=true)
            params["layer_norm.weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels_);
            params["layer_norm.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels_);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [L, C, N]
        x = std::dynamic_pointer_cast<Wav2Vec2Conv1d>(blocks["conv"])->forward(ctx, x);
        if (use_layer_norm_) {
            auto ln = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
            // [L, C, N] -> [C, L, N] -> LayerNorm over C -> back
            x = ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
            x = ln->forward(ctx, x);
            x = ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);
        } else {
            x = ggml_cont(ctx->ggml_ctx, x);
            x = ggml_group_norm(ctx->ggml_ctx, x, (int)channels_, 1e-05f);
            x = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, params["layer_norm.bias"], 1, channels_, 1));
            x = ggml_mul(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, params["layer_norm.weight"], 1, channels_, 1));
        }
        return ggml_ext_gelu(ctx->ggml_ctx, x, true);
    }

private:
    bool use_layer_norm_;
    int64_t channels_;
};

struct Wav2Vec2ConvFeatureEncoder : public GGMLBlock {
    Wav2Vec2ConvFeatureEncoder(const Wav2Vec2Config& config) {
        // kernel sizes (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2); conv0 maps the
        // single waveform channel to conv_dim and always has a bias.
        const int kernels[7] = {10, 3, 3, 3, 3, 2, 2};
        const int strides[7] = {5, 2, 2, 2, 2, 2, 2};
        int64_t in_channels  = 1;
        for (int i = 0; i < 7; ++i) {
            bool bias                                  = (i == 0) ? true : config.conv_bias;
            blocks["conv_layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                new Wav2Vec2ConvLayer(in_channels, config.conv_dim, kernels[i], strides[i], bias, config.conv_norm));
            in_channels = config.conv_dim;
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [L, 1, N] waveform; returns [conv_dim, L', N]
        for (int i = 0; i < 7; ++i) {
            auto conv = std::dynamic_pointer_cast<Wav2Vec2ConvLayer>(blocks["conv_layers." + std::to_string(i)]);
            x         = conv->forward(ctx, x);
        }
        return ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3);  // [conv_dim, L', N]
    }
};

struct Wav2Vec2FeatureProjection : public UnaryBlock {
    Wav2Vec2FeatureProjection(const Wav2Vec2Config& config) {
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(config.conv_dim));
        blocks["projection"] = std::shared_ptr<GGMLBlock>(new Linear(config.conv_dim, config.embed_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [conv_dim, L', N] -> [embed_dim, L', N]
        auto ln         = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
        auto projection = std::dynamic_pointer_cast<Linear>(blocks["projection"]);
        x               = ln->forward(ctx, x);
        x               = projection->forward(ctx, x);
        return x;
    }
};

struct Wav2Vec2PositionalConvEmbedding : public UnaryBlock {
    int64_t embed_dim;
    int64_t groups;
    int kernel_size;
    enum NormConvention {
        NORM_PER_KERNEL_TAP,    // g [1, 1, k]: norm over all channels per kernel tap
        NORM_PER_OUT_KERNEL,    // g [out, 1, k]: norm over in/g per (out, kernel tap)
        NORM_PER_CHANNEL_PAIR,  // g [out, ic_g, 1]: norm over kernel per channel pair
    };
    NormConvention norm_convention;
    bool legacy_key_names;  // weight_g/weight_v vs parametrizations.weight.original0/1

    Wav2Vec2PositionalConvEmbedding(const Wav2Vec2Config& config)
        : embed_dim(config.embed_dim), groups(16), kernel_size(128), norm_convention(NORM_PER_KERNEL_TAP), legacy_key_names(true) {}

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        // prefix arrives with a trailing dot, e.g. "...encoder.pos_conv_embed.";
        // the weight-norm params live under the inner "conv" module.
        const std::string base = prefix + "conv.";
        const int64_t ic_g     = embed_dim / groups;
        ggml_type wtype        = GGML_TYPE_F16;
        auto key_g             = base + "weight_g";
        auto key_v             = base + "weight_v";
        legacy_key_names       = tensor_storage_map.find(key_g) != tensor_storage_map.end();
        if (!legacy_key_names) {
            key_g = base + "parametrizations.weight.original0";
            key_v = base + "parametrizations.weight.original1";
        }
        // The stored g shape decides the norm convention (files exist with
        // parametrizations-style [1, 1, k] g under legacy key names).
        auto it_g = tensor_storage_map.find(key_g);
        GGML_ASSERT(it_g != tensor_storage_map.end() && it_g->second.ne[1] == 1);
        if (it_g->second.ne[2] == 1) {
            norm_convention = NORM_PER_KERNEL_TAP;
        } else if (it_g->second.ne[1] == 1 && it_g->second.ne[2] == embed_dim) {
            norm_convention = NORM_PER_OUT_KERNEL;
        } else if (it_g->second.ne[0] == 1 && it_g->second.ne[1] == ic_g && it_g->second.ne[2] == embed_dim) {
            norm_convention = NORM_PER_CHANNEL_PAIR;
        } else {
            GGML_ABORT("wav2vec2: unsupported weight-norm g shape");
        }
        // Param keys must include the inner "conv." segment: get_param_tensors
        // joins the block prefix with the param key verbatim.
        const std::string rel_g = key_g.substr(prefix.size());  // "conv.weight_g" / "conv.parametrizations..."
        const std::string rel_v = key_v.substr(prefix.size());
        params[rel_g]           = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                     it_g->second.ne[0], it_g->second.ne[1], it_g->second.ne[2]);
        params[rel_v]           = ggml_new_tensor_3d(ctx, get_type(key_v, tensor_storage_map, GGML_TYPE_F16),
                                                     kernel_size, ic_g, embed_dim);
        if (tensor_storage_map.find(base + "bias") != tensor_storage_map.end()) {
            params["conv.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
        }
    }

    // Per-tap (or per-convention) norm of v, in-graph.
    ggml_tensor* norm(GGMLRunnerContext* ctx) {
        const int64_t ic_g = embed_dim / groups;
        const char* key_v  = legacy_key_names ? "conv.weight_v" : "conv.parametrizations.weight.original1";
        ggml_tensor* v     = params[key_v];
        v                  = ggml_cast(ctx->ggml_ctx, v, GGML_TYPE_F32);
        ggml_tensor* sq    = ggml_mul(ctx->ggml_ctx, v, v);
        ggml_tensor* norm;
        if (norm_convention == NORM_PER_KERNEL_TAP) {
            // sq [k, ic_g, out]: reduce channels per kernel tap -> [k, 1, 1]
            // ggml permute axes are destinations of the source dims (inverse of
            // torch's permute), so (2, 0, 1) yields [ic_g, out, k].
            ggml_tensor* pt = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, sq, 2, 0, 1, 3));  // [ic_g, out, k]
            ggml_tensor* s  = ggml_sum_rows(ctx->ggml_ctx, ggml_reshape_2d(ctx->ggml_ctx, pt, ic_g * embed_dim, kernel_size));
            norm            = ggml_sqrt(ctx->ggml_ctx, ggml_reshape_3d(ctx->ggml_ctx, s, kernel_size, 1, 1));
        } else if (norm_convention == NORM_PER_OUT_KERNEL) {
            // sq [k, ic_g, out] -> [ic_g, k, out]: reduce ic_g per (k, out) -> [k, 1, out]
            ggml_tensor* pt = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, sq, 1, 0, 2, 3));
            norm            = ggml_sqrt(ctx->ggml_ctx,
                                        ggml_reshape_3d(ctx->ggml_ctx,
                                                        ggml_sum_rows(ctx->ggml_ctx, ggml_reshape_2d(ctx->ggml_ctx, pt, ic_g, kernel_size * embed_dim)),
                                                        kernel_size, 1, embed_dim));
        } else {
            // sq [k, ic_g, out]: reduce kernel per channel pair -> [1, ic_g, out]
            norm = ggml_sqrt(ctx->ggml_ctx, ggml_sum_rows(ctx->ggml_ctx, sq));
        }
        return norm;
    }

    // Recombines the weight-normalized conv weight: weight = v * g / norm(v).
    ggml_tensor* weight(GGMLRunnerContext* ctx) {
        const char* key_g = legacy_key_names ? "conv.weight_g" : "conv.parametrizations.weight.original0";
        const char* key_v = legacy_key_names ? "conv.weight_v" : "conv.parametrizations.weight.original1";
        ggml_tensor* g    = params[key_g];
        ggml_tensor* v    = params[key_v];
        v                 = ggml_cast(ctx->ggml_ctx, v, GGML_TYPE_F32);
        return ggml_mul(ctx->ggml_ctx, v, ggml_div(ctx->ggml_ctx, g, norm(ctx)));
    }

    // Debug: raw v^2 (pre-reduction) and the transposed copy feeding the reduction.
    ggml_tensor* sq_dump(GGMLRunnerContext* ctx) {
        const char* key_v = legacy_key_names ? "conv.weight_v" : "conv.parametrizations.weight.original1";
        ggml_tensor* v    = params[key_v];
        v                 = ggml_cast(ctx->ggml_ctx, v, GGML_TYPE_F32);
        return ggml_mul(ctx->ggml_ctx, v, v);
    }

    ggml_tensor* pt_dump(GGMLRunnerContext* ctx) {
        return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, sq_dump(ctx), 2, 0, 1, 3));
    }

    // Debug: grouped conv output after drop-last, before GELU (with_bias=false
    // also skips the bias add).
    ggml_tensor* conv_raw(GGMLRunnerContext* ctx, ggml_tensor* x, bool with_bias) {
        ggml_tensor* b = with_bias && params.count("conv.bias") > 0 ? params["conv.bias"] : nullptr;
        ggml_tensor* t = conv_input(ctx, x);
        t              = Wav2Vec2Conv1d::grouped_conv_1d(ctx, t, weight(ctx), groups, 1, kernel_size / 2, b);
        t              = ggml_view_3d(ctx->ggml_ctx, t, t->ne[0] - 1, t->ne[1], t->ne[2], t->nb[1], t->nb[2], 0);
        return t;  // [L, embed_dim, N]
    }

    // The materialized [L, embed_dim, N] plane-contiguous tensor that
    // grouped_conv_1d actually consumes (cont of the permuted projection
    // output; im2col requires this layout).
    ggml_tensor* conv_input(GGMLRunnerContext* ctx, ggml_tensor* x) {
        return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
    }

    // Debug: group-0-only conv (no bias), post drop-last.
    ggml_tensor* conv_group0(GGMLRunnerContext* ctx, ggml_tensor* x) {
        const int64_t ic_g = embed_dim / groups;
        ggml_tensor* t     = conv_input(ctx, x);
        ggml_tensor* w     = weight(ctx);
        ggml_tensor* x_0   = ggml_view_3d(ctx->ggml_ctx, t, t->ne[0], ic_g, t->ne[2], t->nb[1], t->nb[2], 0);
        ggml_tensor* w_0   = ggml_view_3d(ctx->ggml_ctx, w, w->ne[0], ic_g, w->ne[2] / groups, w->nb[1], w->nb[2], 0);
        ggml_tensor* out   = Wav2Vec2Conv1d::conv_1d_f32_im2col(ctx->ggml_ctx, w_0, x_0, 1, kernel_size / 2, 1);
        out                = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0] - 1, out->ne[1], out->ne[2], out->nb[1], out->nb[2], 0);
        return out;  // [L, ic_g, N]
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [embed_dim, L, N]; returns [embed_dim, L, N] (last sample dropped)
        ggml_tensor* w = weight(ctx);
        ggml_tensor* b = params.count("conv.bias") > 0 ? params["conv.bias"] : nullptr;
        ggml_tensor* t = conv_input(ctx, x);
        t              = Wav2Vec2Conv1d::grouped_conv_1d(ctx, t, w, groups, 1, kernel_size / 2, b);
        // GELU must be out-of-place and precede the drop-last view: an in-place
        // op on a view writes through a buffer the graph allocator does not
        // reserve for it, so reuse of the conv output's storage silently
        // corrupts the result depending on the surrounding graph shape.
        t = ggml_ext_gelu(ctx->ggml_ctx, t, false);
        t = ggml_view_3d(ctx->ggml_ctx, t, t->ne[0] - 1, t->ne[1], t->ne[2], t->nb[1], t->nb[2], 0);
        return ggml_permute(ctx->ggml_ctx, t, 1, 0, 2, 3);  // [embed_dim, L, N]
    }
};

struct Wav2Vec2FeedForward : public UnaryBlock {
    Wav2Vec2FeedForward(const Wav2Vec2Config& config) {
        blocks["intermediate_dense"] = std::shared_ptr<GGMLBlock>(new Linear(config.embed_dim, config.embed_dim * 4));
        blocks["output_dense"]       = std::shared_ptr<GGMLBlock>(new Linear(config.embed_dim * 4, config.embed_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto intermediate_dense = std::dynamic_pointer_cast<Linear>(blocks["intermediate_dense"]);
        auto output_dense       = std::dynamic_pointer_cast<Linear>(blocks["output_dense"]);
        x                       = intermediate_dense->forward(ctx, x);
        x                       = ggml_ext_gelu(ctx->ggml_ctx, x, true);
        x                       = output_dense->forward(ctx, x);
        return x;
    }
};

struct Wav2Vec2TransformerEncoderLayer : public UnaryBlock {
    bool do_stable_layer_norm;

    Wav2Vec2TransformerEncoderLayer(const Wav2Vec2Config& config)
        : do_stable_layer_norm(config.do_stable_layer_norm) {
        blocks["attention"]        = std::shared_ptr<GGMLBlock>(new MultiheadAttention(config.embed_dim, config.num_heads, true, true));
        blocks["layer_norm"]       = std::shared_ptr<GGMLBlock>(new LayerNorm(config.embed_dim));
        blocks["feed_forward"]     = std::shared_ptr<GGMLBlock>(new Wav2Vec2FeedForward(config));
        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(config.embed_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [embed_dim, L, N]
        auto attention        = std::dynamic_pointer_cast<MultiheadAttention>(blocks["attention"]);
        auto layer_norm       = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
        auto feed_forward     = std::dynamic_pointer_cast<Wav2Vec2FeedForward>(blocks["feed_forward"]);
        auto final_layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);

        ggml_tensor* residual = x;
        if (do_stable_layer_norm) {
            x = layer_norm->forward(ctx, x);
            x = attention->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, residual, x);
            x = ggml_add(ctx->ggml_ctx, x, feed_forward->forward(ctx, final_layer_norm->forward(ctx, x)));
        } else {
            x = attention->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, residual, x);
            x = layer_norm->forward(ctx, x);
            x = final_layer_norm->forward(ctx, ggml_add(ctx->ggml_ctx, x, feed_forward->forward(ctx, x)));
        }
        return x;
    }
};

struct Wav2Vec2TransformerEncoder : public GGMLBlock {
    int num_layers;
    bool do_stable_layer_norm;

    Wav2Vec2TransformerEncoder(const Wav2Vec2Config& config)
        : num_layers(config.num_layers), do_stable_layer_norm(config.do_stable_layer_norm) {
        blocks["pos_conv_embed"] = std::shared_ptr<GGMLBlock>(new Wav2Vec2PositionalConvEmbedding(config));
        for (int i = 0; i < config.num_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Wav2Vec2TransformerEncoderLayer(config));
        }
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(config.embed_dim));
    }

    ggml_tensor* pos_conv(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->forward(ctx, x);
    }

    ggml_tensor* pos_conv_weight(GGMLRunnerContext* ctx) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->weight(ctx);
    }

    ggml_tensor* pos_conv_norm(GGMLRunnerContext* ctx) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->norm(ctx);
    }

    ggml_tensor* pos_conv_sq(GGMLRunnerContext* ctx) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->sq_dump(ctx);
    }

    ggml_tensor* pos_conv_pt(GGMLRunnerContext* ctx) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->pt_dump(ctx);
    }

    ggml_tensor* pos_conv_raw(GGMLRunnerContext* ctx, ggml_tensor* x, bool with_bias) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->conv_raw(ctx, x, with_bias);
    }

    ggml_tensor* pos_conv_input(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->conv_input(ctx, x);
    }

    ggml_tensor* pos_conv_group0(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto pc = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        return pc->conv_group0(ctx, x);
    }

    // Returns the final hidden state [embed_dim, L, N]. When all_layers != nullptr,
    // it is filled with the 24 pre-layer states plus the final state, concatenated
    // along a new trailing axis: [embed_dim, L, num_layers + 1].
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor** all_layers = nullptr) {
        auto pos_conv_embed = std::dynamic_pointer_cast<Wav2Vec2PositionalConvEmbedding>(blocks["pos_conv_embed"]);
        auto layer_norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);

        std::vector<ggml_tensor*> collected;
        if (all_layers != nullptr) {
            collected.reserve(num_layers + 1);
        }

        x = ggml_add(ctx->ggml_ctx, x, pos_conv_embed->forward(ctx, x));
        if (!do_stable_layer_norm) {
            x = layer_norm->forward(ctx, x);
        }
        for (int i = 0; i < num_layers; ++i) {
            if (all_layers != nullptr) {
                collected.push_back(x);
            }
            auto layer = std::dynamic_pointer_cast<Wav2Vec2TransformerEncoderLayer>(blocks["layers." + std::to_string(i)]);
            x          = layer->forward(ctx, x);
        }
        if (do_stable_layer_norm) {
            x = layer_norm->forward(ctx, x);
        }
        if (all_layers != nullptr) {
            collected.push_back(x);
            ggml_tensor* stack = collected[0];
            for (size_t i = 1; i < collected.size(); ++i) {
                stack = ggml_concat(ctx->ggml_ctx, stack, collected[i], 2);
            }
            *all_layers = stack;
        }
        return x;
    }
};

struct Wav2Vec2Model : public GGMLBlock {
    Wav2Vec2Config config;

    Wav2Vec2Model() = default;
    Wav2Vec2Model(const Wav2Vec2Config& config_) : config(config_) {
        blocks["feature_extractor"]  = std::shared_ptr<GGMLBlock>(new Wav2Vec2ConvFeatureEncoder(config));
        blocks["feature_projection"] = std::shared_ptr<GGMLBlock>(new Wav2Vec2FeatureProjection(config));
        blocks["encoder"]            = std::shared_ptr<GGMLBlock>(new Wav2Vec2TransformerEncoder(config));
    }

    // waveform: [L, 1, N] (already channel-mixed). Returns the final hidden state;
    // all_layers receives the per-layer stack when non-null.
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor** all_layers = nullptr) {
        auto feature_extractor  = std::dynamic_pointer_cast<Wav2Vec2ConvFeatureEncoder>(blocks["feature_extractor"]);
        auto feature_projection = std::dynamic_pointer_cast<Wav2Vec2FeatureProjection>(blocks["feature_projection"]);
        auto encoder            = std::dynamic_pointer_cast<Wav2Vec2TransformerEncoder>(blocks["encoder"]);

        x = feature_extractor->forward(ctx, x);
        x = feature_projection->forward(ctx, x);
        x = encoder->forward(ctx, x, all_layers);
        return x;
    }

    // Debug front-end stages (see Wav2Vec2ModelRunner::build_graph):
    // 1 = feature extractor, 2 = + projection, 3 = + pos_conv and residual add,
    // 4 = the recomputed pos_conv weight tensor, 5 = the pos_conv weight norm.
    ggml_tensor* forward_front(GGMLRunnerContext* ctx, ggml_tensor* x, int stage) {
        auto feature_extractor  = std::dynamic_pointer_cast<Wav2Vec2ConvFeatureEncoder>(blocks["feature_extractor"]);
        auto feature_projection = std::dynamic_pointer_cast<Wav2Vec2FeatureProjection>(blocks["feature_projection"]);
        auto encoder            = std::dynamic_pointer_cast<Wav2Vec2TransformerEncoder>(blocks["encoder"]);

        if (stage == 11) {
            x = feature_extractor->forward(ctx, x);
            x = feature_projection->forward(ctx, x);
            return encoder->pos_conv_group0(ctx, x);
        }
        if (stage == 10) {
            x = feature_extractor->forward(ctx, x);
            x = feature_projection->forward(ctx, x);
            return encoder->pos_conv_input(ctx, x);
        }
        if (stage == 9 || stage == 8) {
            x = feature_extractor->forward(ctx, x);
            x = feature_projection->forward(ctx, x);
            return encoder->pos_conv_raw(ctx, x, stage == 8);
        }
        if (stage == 7) {
            return encoder->pos_conv_pt(ctx);
        }
        if (stage == 6) {
            return encoder->pos_conv_sq(ctx);
        }
        if (stage == 5) {
            return encoder->pos_conv_norm(ctx);
        }
        if (stage == 4) {
            return encoder->pos_conv_weight(ctx);
        }
        x = feature_extractor->forward(ctx, x);
        if (stage == 1) {
            return x;
        }
        x = feature_projection->forward(ctx, x);
        if (stage == 2) {
            return x;
        }
        return ggml_add(ctx->ggml_ctx, x, encoder->pos_conv(ctx, x));
    }
};

class Wav2Vec2ModelRunner : public GGMLRunner {
public:
    Wav2Vec2Model model;
    std::string weight_prefix;

    Wav2Vec2ModelRunner(ggml_backend_t backend,
                        const String2TensorStorage& tensor_storage_map      = {},
                        const std::string prefix                            = "wav2vec2.",
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager), weight_prefix(prefix) {
        config = Wav2Vec2Config::detect_from_weights(tensor_storage_map, prefix);
        model  = Wav2Vec2Model(config);
        // GGMLBlock::init/get_param_tensors append their own '.'; the loader-facing
        // prefix convention carries one.
        std::string block_prefix = weight_prefix;
        if (!block_prefix.empty() && block_prefix.back() == '.') {
            block_prefix.pop_back();
        }
        model.init(params_ctx, tensor_storage_map, block_prefix);
        LOG_INFO("%s", get_desc().c_str());
    }

    std::string get_desc() override {
        char buf[256];
        snprintf(buf, sizeof(buf), "wav2vec2: embed_dim %" PRId64 ", %d layers, %d heads%s",
                 config.embed_dim, config.num_layers, config.num_heads,
                 config.do_stable_layer_norm ? ", stable-ln" : "");
        return std::string(buf);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        std::string block_prefix = weight_prefix;
        if (!block_prefix.empty() && block_prefix.back() == '.') {
            block_prefix.pop_back();
        }
        model.get_param_tensors(tensors, block_prefix);
    }

    // Allocates all param tensors on the backend's default buffer. Standalone
    // harnesses need this before loading weights; the full pipeline routes
    // allocation through ModelManager instead.
    bool alloc_params_on_backend(ggml_backend_t backend) {
        params_buffer = ggml_backend_alloc_ctx_tensors_from_buft(params_ctx, ggml_backend_get_default_buffer_type(backend));
        return params_buffer != nullptr;
    }

    // waveform: [L, 1, 1] mono; already host-normalized. stage 0 (default) returns
    // [embed_dim, L', num_layers + 1] with the per-layer stack (last slice = final
    // hidden state). Stages 1..3 expose front-end intermediates for parity debugging:
    // 1 = feature extractor [conv_dim, L', 1], 2 = + projection [embed_dim, L', 1],
    // 3 = + pos_conv and residual add [embed_dim, L', 1].
    ggml_cgraph* build_graph(const sd::Tensor<float>& waveform_tensor, int stage = 0) {
        ggml_cgraph* gf       = ggml_new_graph(compute_ctx);
        ggml_tensor* waveform = make_input(waveform_tensor);

        auto runner_ctx = get_context();

        if (stage == 0) {
            ggml_tensor* all_layers = nullptr;
            model.forward(&runner_ctx, waveform, &all_layers);
            GGML_ASSERT(all_layers != nullptr);
            ggml_build_forward_expand(gf, all_layers);
        } else {
            // stages produce views (permutes); materialize them so the flat dump
            // reflects logical [C, T, N] order
            ggml_tensor* out = ggml_cont(runner_ctx.ggml_ctx, model.forward_front(&runner_ctx, waveform, stage));
            ggml_build_forward_expand(gf, out);
        }
        return gf;
    }

    sd::Tensor<float> compute(const int n_threads, const std::vector<float>& mono_waveform, int stage = 0) {
        GGML_ASSERT(!mono_waveform.empty());
        const int64_t num_samples = (int64_t)mono_waveform.size();
        sd::Tensor<float> waveform({num_samples, 1, 1});
        std::copy(mono_waveform.begin(), mono_waveform.end(), waveform.data());
        normalize(waveform.data(), num_samples);

        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(waveform, stage);
        };
        return take_or_empty(GGMLRunner::compute(get_graph, n_threads, true));
    }

private:
    Wav2Vec2Config config;
    ggml_backend_buffer_t params_buffer = nullptr;

    // torch: (x - x.mean()) / torch.sqrt(x.var() + 1e-7); var is population variance.
    static void normalize(float* x, int64_t n) {
        double mean = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            mean += x[i];
        }
        mean /= n;
        double var = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double d = x[i] - mean;
            var += d * d;
        }
        var /= n;
        const float scale = (float)(1.0 / std::sqrt(var + 1e-7));
        for (int64_t i = 0; i < n; ++i) {
            x[i] = (float)((x[i] - mean) * scale);
        }
    }
};

#endif  // __SD_MODEL_AUDIO_WAV2VEC2_HPP__
