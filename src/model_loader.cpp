#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/util.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/torch_legacy_io.h"
#include "model_io/torch_zip_io.h"
#include "model_loader.h"
#include "runtime/imatrix.h"
#include "stable-diffusion.h"

#include "core/ggml_extend_backend.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "json.hpp"
#include "zip.h"

#include "name_conversion.h"

/*================================================= Preprocess ==================================================*/

const char* unused_tensors[] = {
    "betas",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "cond_stage_model.1.model.text_model.embeddings.position_ids",
    "cond_stage_model.transformer.vision_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
    "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
    "conditioner.embedders.0.model.logit_scale",
    "conditioner.embedders.1.model.logit_scale",
    "model.diffusion_model.time_embedding.cond_proj.weight",
    "unet.time_embedding.cond_proj.weight",
    "model_ema.decay",
    "model_ema.num_updates",
    "model_ema.diffusion_model",
    "embedding_manager",
    "denoiser.sigmas",
    "text_encoders.t5xxl.transformer.encoder.embed_tokens.weight",  // only used during training
    "ztsnr",                                                        // Found in some SDXL vpred models
    "edm_vpred.sigma_min",                                          // Found in CosXL
    // TODO: find another way to avoid the "unknown tensor" for these two
    // "edm_vpred.sigma_max", // Used to detect CosXL
    // "v_pred", // Used to detect SDXL vpred models
    "text_encoders.llm.output.weight",
    "text_encoders.llm.lm_head.",
    "language_model.lm_head.",
    "vision_model.",
};

bool is_unused_tensor(const std::string& name) {
    for (size_t i = 0; i < sizeof(unused_tensors) / sizeof(const char*); i++) {
        if (starts_with(name, unused_tensors[i])) {
            return true;
        }
    }
    return false;
}

void f64_to_f32_vec(double* src, float* dst, int64_t n) {
    // support inplace op
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

void i64_to_i32_vec(int64_t* src, int32_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (int32_t)src[i];
    }
}

// Dequantize one row of a packed ComfyUI convrot_w4a4 weight to F32.
// packed_k bytes hold K = 2*packed_k values: low nibble = even k, high nibble =
// odd k, two's-complement int4, scaled by the per-row F32 weight_scale.
static void dequant_w4a4_row(const uint8_t* packed, int64_t packed_k, float row_scale, float* dst) {
    for (int64_t j = 0; j < packed_k; j++) {
        const int lo_nib = packed[j] & 0xF;
        const int hi_nib = packed[j] >> 4;
        const int lo     = lo_nib < 8 ? lo_nib : lo_nib - 16;
        const int hi     = hi_nib < 8 ? hi_nib : hi_nib - 16;
        dst[2 * j]       = lo * row_scale;
        dst[2 * j + 1]   = hi * row_scale;
    }
}

// Dequantize one row of a packed ComfyUI asym_w4a8_int8 weight to F32:
// w[k] = codebook[code] * s_channel * s_rel[k / group_size], codes unsigned 0-15.
static void dequant_w4a8_row(const uint8_t* packed,
                             int64_t packed_k,
                             const float* codebook,
                             float s_channel,
                             const float* s_rel_row,
                             int s_rel_group_size,
                             float* dst) {
    for (int64_t j = 0; j < packed_k; j++) {
        const int lo   = packed[j] & 0xF;
        const int hi   = packed[j] >> 4;
        dst[2 * j]     = codebook[lo] * s_channel * s_rel_row[(2 * j) / s_rel_group_size];
        dst[2 * j + 1] = codebook[hi] * s_channel * s_rel_row[(2 * j + 1) / s_rel_group_size];
    }
}

static bool read_stream_at(std::ifstream& file, size_t offset, char* dst, size_t n) {
    file.clear();
    file.seekg((std::streamoff)offset, std::ios::beg);
    file.read(dst, (std::streamsize)n);
    return static_cast<bool>(file);
}

// Disk sidecar cache for reblocked w4 tensors: "<model>.w4reblock".
// Validated against the source file's size + mtime; any mismatch forces a reconvert.
static const char W4_CACHE_MAGIC[8] = {'S', 'D', 'W', '4', 'R', 'B', 'C', '1'};
static constexpr uint32_t W4_CACHE_VERSION = 1;

static bool write_u32(std::ofstream& out, uint32_t v) { return (bool)out.write((const char*)&v, 4); }
static bool write_u64(std::ofstream& out, uint64_t v) { return (bool)out.write((const char*)&v, 8); }
static bool read_u32(std::ifstream& in, uint32_t* v) { return (bool)in.read((char*)v, 4); }
static bool read_u64(std::ifstream& in, uint64_t* v) { return (bool)in.read((char*)v, 8); }

static bool try_load_w4_cache(const std::string& cache_path,
                              const std::string& model_path,
                              const std::vector<std::pair<std::string, TensorStorage>>& items,
                              std::map<std::string, std::vector<char>>& out_map) {
    std::ifstream in(cache_path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    char magic[8];
    uint32_t version = 0, count = 0, reserved = 0;
    uint64_t src_size = 0, src_mtime = 0;
    if (!in.read(magic, 8) || memcmp(magic, W4_CACHE_MAGIC, 8) != 0 || !read_u32(in, &version) ||
        !read_u32(in, &reserved) || !read_u64(in, &src_size) || !read_u64(in, &src_mtime) || !read_u32(in, &count)) {
        return false;
    }
    if (version != W4_CACHE_VERSION || count != items.size()) {
        return false;
    }
    std::error_code ec;
    if (std::filesystem::file_size(model_path, ec) != src_size || ec ||
        std::filesystem::last_write_time(model_path, ec).time_since_epoch().count() != (int64_t)src_mtime || ec) {
        return false;
    }

    std::map<std::string, std::vector<char>> loaded;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t name_len = 0;
        if (!read_u32(in, &name_len) || name_len == 0 || name_len > 4096) {
            return false;
        }
        std::string name(name_len, '\0');
        if (!in.read(name.data(), name_len)) {
            return false;
        }
        uint32_t kind = 0;
        uint64_t nbytes = 0;
        if (!read_u32(in, &kind) || !read_u64(in, &nbytes)) {
            return false;
        }
        const TensorStorage* ts = nullptr;
        for (const auto& [n, s] : items) {
            if (n == name) {
                ts = &s;
                break;
            }
        }
        if (ts == nullptr || (int)kind != ts->w4_convrot_kind ||
            nbytes != (uint64_t)ggml_row_size(ts->type == GGML_TYPE_Q4_0 ? GGML_TYPE_Q4_0 : GGML_TYPE_IQ4_NL,
                                              ts->ne[0]) *
                ts->ne[1]) {
            return false;
        }
        std::vector<char>& buf = loaded[name];
        buf.resize(nbytes);
        if (!in.read(buf.data(), (std::streamsize)nbytes)) {
            return false;
        }
    }
    out_map = std::move(loaded);
    return true;
}

static bool write_w4_cache(const std::string& cache_path,
                           const std::string& model_path,
                           const std::vector<std::pair<std::string, TensorStorage>>& items,
                           const std::map<std::string, std::vector<char>>& data) {
    std::error_code ec;
    const uint64_t src_size    = std::filesystem::file_size(model_path, ec);
    const uint64_t src_mtime   = (uint64_t)std::filesystem::last_write_time(model_path, ec).time_since_epoch().count();
    const std::string tmp_path = cache_path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out.is_open()) {
            return false;
        }
        out.write(W4_CACHE_MAGIC, 8);
        write_u32(out, W4_CACHE_VERSION);
        write_u32(out, 0);
        write_u64(out, src_size);
        write_u64(out, src_mtime);
        write_u32(out, (uint32_t)items.size());
        for (const auto& [name, ts] : items) {
            const auto& buf = data.at(name);
            write_u32(out, (uint32_t)name.size());
            out.write(name.data(), (std::streamsize)name.size());
            write_u32(out, (uint32_t)ts.w4_convrot_kind);
            write_u64(out, (uint64_t)buf.size());
            out.write(buf.data(), (std::streamsize)buf.size());
        }
        if (!out) {
            return false;
        }
    }
    std::filesystem::rename(tmp_path, cache_path, ec);
    return !ec;
}

const TensorStorage* find_w4_companion(const String2TensorStorage& map,
                                       const std::string& module_name,
                                       const char* suffix) {
    auto it = map.find(module_name + suffix);
    return it == map.end() ? nullptr : &it->second;
}

// Converts one packed w4 convrot tensor storage into a ggml-native Q4_0/IQ4_NL buffer.
// Opens the source file once and services all reads through it; keeps no shared state,
// so it is safe to run concurrently for different tensors.
static bool convert_one_w4_tensor(const String2TensorStorage& tensor_storage_map,
                                  const std::vector<std::string>& file_paths,
                                  const std::string& name,
                                  const TensorStorage& ts,
                                  std::vector<char>& out,
                                  float* max_err_out,
                                  float* rel_l2_out,
                                  std::string& error) {
    if (ts.index_in_zip >= 0) {
        error = sd_format("packed w4 convrot weights inside zip archives are not supported: '%s'", name.c_str());
        return false;
    }
    if (ts.file_index >= file_paths.size()) {
        error = sd_format("missing file for packed w4 convrot weight '%s'", name.c_str());
        return false;
    }
    const bool is_w4a4            = ts.w4_convrot_kind == W4_CONVROT_W4A4;
    const ggml_type target_type   = is_w4a4 ? GGML_TYPE_Q4_0 : GGML_TYPE_IQ4_NL;
    const int64_t k               = ts.ne[0];  // de-packed
    const int64_t n               = ts.ne[1];
    const int64_t packed_k        = k / 2;
    const std::string module_name = name.substr(0, name.size() - std::string(".weight").size());
    const std::string& file_path  = file_paths[ts.file_index];
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        error = sd_format("failed to open '%s' for packed w4 convrot weight '%s'", file_path.c_str(), name.c_str());
        return false;
    }

        // companion storages must live in the same file as the weight
        const TensorStorage* scale_ts     = nullptr;
        const TensorStorage* codebook_ts  = nullptr;
        const TensorStorage* s_channel_ts = nullptr;
        const TensorStorage* s_rel_ts     = nullptr;
        if (is_w4a4) {
            scale_ts = find_w4_companion(tensor_storage_map, module_name, ".weight_scale");
            if (scale_ts == nullptr || scale_ts->type != GGML_TYPE_F32 || scale_ts->nelements() != n ||
                scale_ts->file_index != ts.file_index) {
                error = sd_format("missing or invalid weight_scale for packed w4a4 weight '%s'", name.c_str());
                return false;
            }
        } else {
            codebook_ts  = find_w4_companion(tensor_storage_map, module_name, ".weight_codebook");
            s_channel_ts = find_w4_companion(tensor_storage_map, module_name, ".weight_s_channel");
            s_rel_ts     = find_w4_companion(tensor_storage_map, module_name, ".weight_s_rel");
            if (codebook_ts == nullptr || codebook_ts->type != GGML_TYPE_F32 || codebook_ts->nelements() != 16 ||
                codebook_ts->file_index != ts.file_index ||
                s_channel_ts == nullptr || s_channel_ts->type != GGML_TYPE_F32 || s_channel_ts->nelements() != n ||
                s_channel_ts->file_index != ts.file_index ||
                s_rel_ts == nullptr || !s_rel_ts->is_f8_e4m3 || s_rel_ts->n_dims != 2 ||
                s_rel_ts->ne[0] != k / ts.w4a8_group_size || s_rel_ts->ne[1] != n ||
                s_rel_ts->file_index != ts.file_index) {
                error = sd_format("missing or invalid codebook/s_channel/s_rel for packed w4a8 weight '%s'", name.c_str());
                return false;
            }
        }

        // read small companions fully
        std::vector<float> row_scales(n, 1.0f);
        std::vector<float> codebook(16, 0.0f);
        std::vector<float> s_rel_f32;
        if (is_w4a4) {
            if (!read_stream_at(file, scale_ts->offset, (char*)row_scales.data(), n * sizeof(float))) {
                error = sd_format("failed to read weight_scale for packed w4a4 weight '%s'", name.c_str());
                return false;
            }
        } else {
            const int64_t s_rel_elements = s_rel_ts->nelements();
            std::vector<uint8_t> s_rel_raw(s_rel_elements);
            if (!read_stream_at(file, codebook_ts->offset, (char*)codebook.data(), 16 * sizeof(float)) ||
                !read_stream_at(file, s_channel_ts->offset, (char*)row_scales.data(), n * sizeof(float)) ||
                !read_stream_at(file, s_rel_ts->offset, (char*)s_rel_raw.data(), s_rel_elements)) {
                error = sd_format("failed to read companions for packed w4a8 weight '%s'", name.c_str());
                return false;
            }
            s_rel_f32.resize(s_rel_elements);
            auto f8_traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
            f8_traits->to_float(s_rel_raw.data(), s_rel_f32.data(), s_rel_elements);
        }

        // dequant + requant in row chunks to bound transient memory
        const int64_t out_row_bytes = ggml_row_size(target_type, k);
        out.resize(out_row_bytes * n);
        const int64_t row_chunk = std::max<int64_t>(1, (4 * 1024 * 1024) / (k * (int64_t)sizeof(float)));
        std::vector<float> f32((size_t)k * row_chunk);
        std::vector<uint8_t> packed((size_t)packed_k * row_chunk);
        std::vector<float> imatrix(k, 1.0f);
        // reblock fidelity stats: requantized rows dequantized back vs source
        float max_err = 0.f, sum_sq_err = 0.f, sum_sq_ref = 0.f;
        for (int64_t row0 = 0; row0 < n; row0 += row_chunk) {
            const int64_t rows      = std::min(row_chunk, n - row0);
            const size_t read_start = ts.offset + (size_t)(row0 * packed_k);
            if (!read_stream_at(file, read_start, (char*)packed.data(), (size_t)(rows * packed_k))) {
                error = sd_format("failed to read packed weight data for '%s'", name.c_str());
                return false;
            }
            for (int64_t r = 0; r < rows; r++) {
                float* dst_row         = f32.data() + (size_t)r * k;
                const uint8_t* src_row = packed.data() + (size_t)r * packed_k;
                if (is_w4a4) {
                    dequant_w4a4_row(src_row, packed_k, row_scales[row0 + r], dst_row);
                } else {
                    dequant_w4a8_row(src_row,
                                     packed_k,
                                     codebook.data(),
                                     row_scales[row0 + r],
                                     s_rel_f32.data() + (size_t)(row0 + r) * (k / ts.w4a8_group_size),
                                     ts.w4a8_group_size,
                                     dst_row);
                }
            }
            ggml_quantize_chunk(target_type,
                                f32.data(),
                                out.data() + (size_t)(row0 * out_row_bytes),
                                0,
                                (int)rows,
                                (int)k,
                                imatrix.data());
            {
                std::vector<float> back((size_t)rows * k);
                auto traits = ggml_get_type_traits(target_type);
                traits->to_float(out.data() + (size_t)(row0 * out_row_bytes), back.data(), rows * k);
                for (int64_t i = 0; i < rows * k; i++) {
                    const float diff = back[i] - f32[i];
                    max_err          = std::max(max_err, std::fabs(diff));
                    sum_sq_err += diff * diff;
                    sum_sq_ref += f32[i] * f32[i];
                }
            }
        }
    *max_err_out = max_err;
    *rel_l2_out  = std::sqrt(sum_sq_err) / (std::sqrt(sum_sq_ref) + 1e-9f);
    return true;
}

bool ModelLoader::convert_w4_tensor_storages() {
    converted_w4_data_.clear();
    const int64_t t0 = ggml_time_ms();
    // snapshot the tensors to convert; remember whether they all live in one file,
    // which is the precondition for the per-model disk sidecar cache
    std::vector<std::pair<std::string, TensorStorage>> items;
    std::string single_source;
    bool single_file = true;
    for (const auto& [name, ts] : tensor_storage_map) {
        if (ts.w4_convrot_kind == W4_CONVROT_NONE || is_unused_tensor(name)) {
            continue;
        }
        if (ts.file_index >= file_paths_.size()) {
            LOG_ERROR("missing file for packed w4 convrot weight '%s'", name.c_str());
            return false;
        }
        const std::string& path = file_paths_[ts.file_index];
        if (single_source.empty()) {
            single_source = path;
        } else if (single_source != path) {
            single_file = false;
        }
        items.emplace_back(name, ts);
    }
    if (items.empty()) {
        return true;
    }
    std::string cache_path;
    if (single_file) {
        cache_path = single_source + ".w4reblock";
        if (try_load_w4_cache(cache_path, single_source, items, converted_w4_data_)) {
            uint64_t cached_bytes = 0;
            for (const auto& [n, buf] : converted_w4_data_) {
                cached_bytes += buf.size();
            }
            LOG_INFO("loaded %zu reblocked w4 tensors (%.2f MB) from cache '%s', taking %.2fs",
                     items.size(),
                     cached_bytes / (1024.0 * 1024.0),
                     cache_path.c_str(),
                     (ggml_time_ms() - t0) / 1000.f);
            return true;
        }
    }
    std::atomic<size_t> next_idx(0);
    std::atomic<bool> any_failed(false);
    std::mutex result_mutex;
    std::map<std::string, std::vector<char>> results;
    std::vector<std::string> stat_lines;
    std::string first_error;
    auto worker = [&]() {
        while (!any_failed.load(std::memory_order_relaxed)) {
            const size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= items.size()) {
                break;
            }
            const auto& [name, ts] = items[i];
            std::vector<char> out;
            float max_err = 0.f, rel_l2 = 0.f;
            std::string error;
            if (!convert_one_w4_tensor(tensor_storage_map, file_paths_, name, ts, out, &max_err, &rel_l2, error)) {
                std::lock_guard<std::mutex> lock(result_mutex);
                if (first_error.empty()) {
                    first_error = error;
                }
                any_failed.store(true);
                return;
            }
            std::lock_guard<std::mutex> lock(result_mutex);
            stat_lines.push_back(sd_format("reblocked '%s': max |diff| = %.5f, rel L2 = %.5f",
                                           name.c_str(),
                                           max_err,
                                           rel_l2));
            results.emplace(name, std::move(out));
        }
    };
    const int n_threads = std::max(1, std::min<int>(n_threads_, (int)items.size()));
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
    if (any_failed.load()) {
        LOG_ERROR("%s", first_error.c_str());
        return false;
    }
    uint64_t converted_bytes = 0;
    for (auto& [name, buf] : results) {
        converted_bytes += buf.size();
        converted_w4_data_.emplace(name, std::move(buf));
    }
    for (const auto& line : stat_lines) {
        LOG_VERBOSE("%s", line.c_str());
    }
    const bool cached = !cache_path.empty() && write_w4_cache(cache_path, single_source, items, results);
    if (!cache_path.empty()) {
        if (cached) {
            LOG_VERBOSE("wrote reblock cache '%s'", cache_path.c_str());
        } else {
            LOG_WARN("failed to write reblock cache '%s' (will reconvert on next load)", cache_path.c_str());
        }
    }
    LOG_INFO("reblocked %zu packed w4 convrot tensors (%.2f MB) on %d threads, taking %.2fs",
             items.size(),
             converted_bytes / (1024.0 * 1024.0),
             n_threads,
             (ggml_time_ms() - t0) / 1000.f);
    return true;
}

void convert_tensor(void* src,
                    ggml_type src_type,
                    void* dst,
                    ggml_type dst_type,
                    int nrows,
                    int n_per_row,
                    std::vector<float> imatrix = {}) {
    int n = nrows * n_per_row;
    if (src_type == dst_type) {
        size_t nbytes = n * ggml_type_size(src_type) / ggml_blck_size(src_type);
        memcpy(((char*)dst), ((char*)src), nbytes);
    } else if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src, (ggml_fp16_t*)dst, n);
        } else {
            imatrix.resize(n_per_row, 1.0f);
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src, dst, 0, nrows, n_per_row, im);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)src, (float*)dst, n);
        } else {
            auto qtype = ggml_get_type_traits(src_type);
            if (qtype->to_float == nullptr) {
                throw std::runtime_error(sd_format("type %s unsupported for integer quantization: no dequantization available",
                                                   ggml_type_name(src_type)));
            }
            qtype->to_float(src, (float*)dst, n);
        }
    } else {
        // src_type == GGML_TYPE_F16 => dst_type is quantized
        // src_type is quantized => dst_type == GGML_TYPE_F16 or dst_type is quantized
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == nullptr) {
            throw std::runtime_error(sd_format("type %s unsupported for integer quantization: no dequantization available",
                                               ggml_type_name(src_type)));
        }
        std::vector<char> buf;
        buf.resize(sizeof(float) * n);
        char* src_data_f32 = buf.data();
        qtype->to_float(src, (float*)src_data_f32, n);
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src_data_f32, (ggml_fp16_t*)dst, n);
        } else {
            imatrix.resize(n_per_row, 1.0f);
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src_data_f32, dst, 0, nrows, n_per_row, im);
        }
    }
}

/*================================================= ModelLoader ==================================================*/

ModelLoader::ModelLoader()
    : n_threads_(sd_get_num_physical_cores()) {
}

size_t ModelLoader::add_file_path(const std::string& file_path) {
    auto it = std::find(file_paths_.begin(), file_paths_.end(), file_path);
    if (it != file_paths_.end()) {
        return static_cast<size_t>(it - file_paths_.begin());
    }
    invalidate_file_data();
    file_paths_.push_back(file_path);
    return file_paths_.size() - 1;
}

void ModelLoader::add_tensor_storage(const TensorStorage& tensor_storage) {
    if (tensor_storage_map.count(tensor_storage.name) != 0) {
        throw std::runtime_error("duplicate tensor in model source: " + tensor_storage.name);
    }
    tensor_storage_map[tensor_storage.name] = tensor_storage;
}

void ModelLoader::set_n_threads(int n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    LOG_VERBOSE("using %d threads for model loading", n_threads_);
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    return add_file(file_path, prefix);
}

bool ModelLoader::parse_file(const std::string& file_path, const std::string& prefix) {
    FileStamp stamp;
    if (!read_file_stamp(file_path, stamp)) {
        return false;
    }
    parsed_dependencies_.push_back(stamp);
    if (is_directory(file_path)) {
        const std::string diffusers_index_path = path_join(file_path, "model_index.json");
        const std::string diffusers_unet_path  = path_join(file_path, "unet/diffusion_pytorch_model.safetensors");
        const bool has_diffusers_layout        = file_exists(diffusers_index_path) || file_exists(diffusers_unet_path);

        const std::string safetensors_index_path = path_join(file_path, "model.safetensors.index.json");
        if (!has_diffusers_layout && file_exists(safetensors_index_path)) {
            LOG_INFO("load %s using root safetensors index", file_path.c_str());
            return parse_file(safetensors_index_path, prefix);
        }
        LOG_INFO("load %s using diffusers format", file_path.c_str());
        return init_from_diffusers_file(file_path, prefix);
    } else if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (ends_with(file_path, ".json")) {
        LOG_INFO("load %s using safetensors index format", file_path.c_str());
        return init_from_safetensors_index_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
        LOG_INFO("load %s using safetensors format", file_path.c_str());
        return init_from_safetensors_file(file_path, prefix);
    } else if (is_torch_zip_file(file_path)) {
        LOG_INFO("load %s using torch zip format", file_path.c_str());
        return init_from_torch_zip_file(file_path, prefix);
    } else if (init_from_torch_legacy_file(file_path, prefix)) {
        LOG_INFO("load %s using torch legacy format", file_path.c_str());
        return true;
    } else {
        if (file_exists(file_path)) {
            LOG_WARN("unknown format %s", file_path.c_str());
        } else {
            LOG_WARN("file %s not found", file_path.c_str());
        }
        return false;
    }
}

void ModelLoader::convert_tensors_name() {
    if (names_converted_) {
        return;
    }
    names_converted_ = true;
    rebuild_catalog();
}

bool ModelLoader::init_from_file_and_convert_name(const std::string& file_path, const std::string& prefix, SDVersion version) {
    if (version_ == VERSION_COUNT && version != VERSION_COUNT) {
        version_ = version;
    }
    if (!init_from_file(file_path, prefix)) {
        return false;
    }
    convert_tensors_name();
    return true;
}

/*================================================= GGUFModelLoader ==================================================*/

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    LOG_VERBOSE("init from '%s'", file_path.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_gguf_file(file_path, tensor_storages, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    size_t file_index = add_file_path(file_path);

    for (auto& tensor_storage : tensor_storages) {
        // LOG_VERBOSE("%s", tensor_storage.name.c_str());

        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);
    }

    return true;
}

/*================================================= SafeTensorsModelLoader ==================================================*/

bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    LOG_VERBOSE("init from '%s', prefix = '%s'", file_path.c_str(), prefix.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_safetensors_file(file_path, tensor_storages, &error, &metadata_, &parsed_tensor_names_[file_path])) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    size_t file_index = add_file_path(file_path);

    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);

        // LOG_VERBOSE("%s", tensor_storage.to_string().c_str());
    }

    return true;
}

bool ModelLoader::init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix) {
    LOG_VERBOSE("init from safetensors index '%s', prefix = '%s'", file_path.c_str(), prefix.c_str());

    std::vector<std::string> shard_paths;
    std::string error;
    if (!read_safetensors_index_file(file_path, shard_paths, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    for (const std::string& shard_path : shard_paths) {
        if (!parse_file(shard_path, prefix)) {
            return false;
        }
    }

    std::ifstream index_file(file_path);
    const auto index = nlohmann::json::parse(index_file);
    for (const auto& entry : index.at("weight_map").items()) {
        const auto expected = (std::filesystem::u8path(file_path).parent_path() /
                               std::filesystem::u8path(entry.value().get<std::string>()))
                                  .lexically_normal();
        bool found = false;
        for (const auto& shard : parsed_tensor_names_) {
            if (std::filesystem::u8path(shard.first).lexically_normal() == expected) {
                found = shard.second.count(entry.key()) != 0;
                break;
            }
        }
        if (!found) {
            LOG_ERROR("safetensors index tensor '%s' is missing from its declared shard", entry.key().c_str());
            return false;
        }
    }

    return true;
}

/*================================================= TorchLegacyModelLoader ==================================================*/

bool ModelLoader::init_from_torch_legacy_file(const std::string& file_path, const std::string& prefix) {
    LOG_VERBOSE("init from torch legacy '%s'", file_path.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_torch_legacy_file(file_path, tensor_storages, &error)) {
        if ((!error.empty()) && (ends_with(file_path, ".pt") || ends_with(file_path, ".pth"))) {
            LOG_WARN("%s", error.c_str());
        }
        return false;
    }

    size_t file_index = add_file_path(file_path);

    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);
    }

    return true;
}

/*================================================= TorchZipModelLoader ==================================================*/

bool ModelLoader::init_from_torch_zip_file(const std::string& file_path, const std::string& prefix) {
    LOG_VERBOSE("init from '%s'", file_path.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_torch_zip_file(file_path, tensor_storages, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    size_t file_index = add_file_path(file_path);

    for (auto& tensor_storage : tensor_storages) {
        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);

        // LOG_VERBOSE("%s", tensor_storage.to_string().c_str());
    }

    return true;
}

/*================================================= DiffusersModelLoader ==================================================*/

bool ModelLoader::init_from_diffusers_file(const std::string& file_path, const std::string& prefix) {
    std::string unet_path   = path_join(file_path, "unet/diffusion_pytorch_model.safetensors");
    std::string vae_path    = path_join(file_path, "vae/diffusion_pytorch_model.safetensors");
    std::string clip_path   = path_join(file_path, "text_encoder/model.safetensors");
    std::string clip_g_path = path_join(file_path, "text_encoder_2/model.safetensors");

    if (!parse_file(unet_path, prefix + "unet.")) {
        return false;
    }

    if (file_exists(vae_path) && !parse_file(vae_path, prefix + "vae.")) {
        return false;
    }
    if (file_exists(clip_path) && !parse_file(clip_path, prefix + "te.")) {
        return false;
    }
    if (file_exists(clip_g_path) && !parse_file(clip_g_path, prefix + "te.1.")) {
        return false;
    }
    return true;
}

SDVersion ModelLoader::get_sd_version() const {
    TensorStorage token_embedding_weight, input_block_weight, context_ebedding_weight;

    bool has_multiple_encoders = false;
    bool is_unet               = false;

    bool is_xl                       = false;
    bool is_flux                     = false;
    bool is_flux2                    = false;
    bool has_single_block_47         = false;
    bool is_wan                      = false;
    bool is_s2v                      = false;
    int64_t patch_embedding_channels = 0;
    bool has_img_emb                 = false;
    bool has_middle_block_1          = false;
    bool has_output_block_311        = false;
    bool has_output_block_71         = false;
    bool has_attn_1024               = false;

    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (tensor_storage.name.find("model.diffusion_model.double_blocks.") != std::string::npos ||
            tensor_storage.name.find("model.diffusion_model.single_transformer_blocks.") != std::string::npos) {
            is_flux = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.net.lq_proj.latent_proj.0.weight") != std::string::npos) {
            return VERSION_PID;
        }
        if (tensor_storage.name.find("embed_image_indicator.weight") != std::string::npos) {
            return VERSION_IDEOGRAM4;
        }
        if (tensor_storage.name.find("model.diffusion_model.txtfusion.projector.weight") != std::string::npos ||
            tensor_storage.name.find("model.diffusion_model.text_fusion.projector.weight") != std::string::npos) {
            return VERSION_KREA2;
        }
        if (tensor_storage.name.find("model.diffusion_model.nerf_final_layer_conv.") != std::string::npos) {
            return VERSION_CHROMA_RADIANCE;
        }
        if (tensor_storage.name.find("model.diffusion_model.joint_blocks.") != std::string::npos) {
            return VERSION_SD3;
        }
        if (tensor_storage.name.find("model.x_embedder.proj1.weight") != std::string::npos &&
            tensor_storage_map.find("model.language_model.layers.0.self_attn.q_proj.weight") != tensor_storage_map.end()) {
            return VERSION_HIDREAM_O1;
        }
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight") != std::string::npos &&
            tensor_storage_map.find("model.diffusion_model.transformer_blocks.0.img_mlp.w1.weight") != tensor_storage_map.end()) {
            return VERSION_LENS;
        }
        if (tensor_storage.name.find("net.img_embedder.proj1.weight") != std::string::npos) {
            return VERSION_MINIT2I;
        }
        if (tensor_storage.name.find("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight") != std::string::npos) {
            return VERSION_SENSENOVA_U1_5;
        }
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.img_mod.1.weight") != std::string::npos) {
            auto img_in = tensor_storage_map.find("model.diffusion_model.img_in.weight");
            if (img_in != tensor_storage_map.end() && img_in->second.ne[0] == 128) {
                return VERSION_MAGE_FLOW;
            }
            if (tensor_storage_map.find("model.diffusion_model.time_text_embed.addition_t_embedding.weight") != tensor_storage_map.end()) {
                return VERSION_QWEN_IMAGE_LAYERED;
            }
            return VERSION_QWEN_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight") != std::string::npos) {
            return VERSION_HUNYUAN_VIDEO;
        }
        if (tensor_storage.name.find("llm_adapter.blocks.0.cross_attn.q_proj.weight") != std::string::npos) {
            return VERSION_ANIMA;
        }
        if (tensor_storage.name.find("model.diffusion_model.double_stream_modulation_img.lin.weight") != std::string::npos) {
            is_flux2 = true;
        }
        if (tensor_storage.name.find("dual_time_embed.semantic_embedder.linear_1.weight") != std::string::npos) {
            return VERSION_SEFI_IMAGE;
        }
        if (tensor_storage.name.find("single_blocks.47.linear1.weight") != std::string::npos) {
            has_single_block_47 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.double_blocks.0.img_mlp.gate_proj.weight") != std::string::npos) {
            return VERSION_OVIS_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.cap_embedder.0.weight") != std::string::npos) {
            return VERSION_Z_IMAGE;
        }
        if (tensor_storage.name.find("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight") != std::string::npos) {
            return VERSION_BOOGU_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.layers.0.adaLN_sa_ln.weight") != std::string::npos) {
            return VERSION_ERNIE_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias") != std::string::npos) {
            return VERSION_LTXAV;
        }
        if (tensor_storage.name.find("model.diffusion_model.video_patch_proj.weight") != std::string::npos &&
            tensor_storage_map.find("model.diffusion_model.audio_patch_proj.weight") != tensor_storage_map.end()) {
            return VERSION_MINIMAX_H3;
        }
        if (tensor_storage.name.find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight") != std::string::npos) {
            is_wan = true;
        }
        if (tensor_storage.name.find("casual_audio_encoder.weights") != std::string::npos ||
            tensor_storage.name.find("audio_injector.injector.0.q.weight") != std::string::npos) {
            // S2V and T2V-14B share patch_embedding shapes.
            is_s2v = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.patch_embedder.weight") != std::string::npos) {
            return VERSION_LINGBOT_VIDEO;
        }
        if (tensor_storage.name.find("model.diffusion_model.patch_embedding.weight") != std::string::npos) {
            patch_embedding_channels = tensor_storage.ne[3];
        }
        if (tensor_storage.name.find("model.diffusion_model.img_emb") != std::string::npos) {
            has_img_emb = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.input_blocks.") != std::string::npos ||
            tensor_storage.name.find("unet.down_blocks.") != std::string::npos) {
            is_unet = true;
            if (has_multiple_encoders) {
                is_xl = true;
            }
        }
        if (tensor_storage.name.find("conditioner.embedders.1") != std::string::npos ||
            tensor_storage.name.find("cond_stage_model.1") != std::string::npos ||
            tensor_storage.name.find("te.1") != std::string::npos) {
            has_multiple_encoders = true;
            if (is_unet) {
                is_xl = true;
            }
        }
        if (tensor_storage.name.find("model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor") != std::string::npos) {
            return VERSION_SVD;
        }
        if (tensor_storage.name.find("model.diffusion_model.middle_block.1.") != std::string::npos ||
            tensor_storage.name.find("unet.mid_block.resnets.1.") != std::string::npos) {
            has_middle_block_1 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.output_blocks.3.1.transformer_blocks.1") != std::string::npos ||
            tensor_storage.name.find("unet.up_blocks.1.attentions.0.transformer_blocks.1") != std::string::npos) {
            has_output_block_311 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.output_blocks.7.1") != std::string::npos ||
            tensor_storage.name.find("unet.up_blocks.2.attentions.1") != std::string::npos) {
            has_output_block_71 = true;
            if (tensor_storage.name.find("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight") != std::string::npos) {
                if (tensor_storage.ne[0] == 1024)
                    has_attn_1024 = true;
            }
        }
        if (tensor_storage.name == "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "cond_stage_model.model.token_embedding.weight" ||
            tensor_storage.name == "text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "te.text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "conditioner.embedders.0.model.token_embedding.weight" ||
            tensor_storage.name == "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") {
            token_embedding_weight = tensor_storage;
            // break;
        }
        if (tensor_storage.name == "model.diffusion_model.input_blocks.0.0.weight" ||
            tensor_storage.name == "model.diffusion_model.img_in.weight" ||
            tensor_storage.name == "unet.conv_in.weight") {
            input_block_weight = tensor_storage;
        }
        if (tensor_storage.name == "model.diffusion_model.txt_in.weight" || tensor_storage.name == "model.diffusion_model.context_embedder.weight") {
            context_ebedding_weight = tensor_storage;
        }
    }
    if (is_wan) {
        LOG_VERBOSE("patch_embedding_channels %d", patch_embedding_channels);
        if (is_s2v) {
            return VERSION_WAN2_2_S2V;
        }
        if (patch_embedding_channels == 184320 && !has_img_emb) {
            return VERSION_WAN2_2_I2V;
        }
        if (patch_embedding_channels == 147456 && !has_img_emb) {
            return VERSION_WAN2_2_TI2V;
        }
        return VERSION_WAN2;
    }
    bool is_inpaint = input_block_weight.ne[2] == 9;
    bool is_ip2p    = input_block_weight.ne[2] == 8;
    if (is_xl) {
        if (is_inpaint) {
            return VERSION_SDXL_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SDXL_PIX2PIX;
        }
        if (!has_middle_block_1) {
            if (!has_output_block_311) {
                return VERSION_SDXL_VEGA;
            }
            return VERSION_SDXL_SSD1B;
        }
        return VERSION_SDXL;
    }

    if (is_flux && !is_flux2) {
        if (context_ebedding_weight.ne[0] == 3584) {
            return VERSION_LONGCAT;
        } else {
            if (input_block_weight.ne[0] == 384) {
                return VERSION_FLUX_FILL;
            }
            if (input_block_weight.ne[0] == 128) {
                return VERSION_FLUX_CONTROLS;
            }
            if (input_block_weight.ne[0] == 196) {
                return VERSION_FLEX_2;
            }
            return VERSION_FLUX;
        }
    }

    if (is_flux2) {
        if (has_single_block_47) {
            return VERSION_FLUX2;
        }
        return VERSION_FLUX2_KLEIN;
    }

    if (token_embedding_weight.ne[0] == 768) {
        if (is_inpaint) {
            return VERSION_SD1_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SD1_PIX2PIX;
        }
        if (!has_middle_block_1) {
            if (!has_output_block_71) {
                return VERSION_SDXS_512_DS;
            }
            return VERSION_SD1_TINY_UNET;
        }
        return VERSION_SD1;
    } else if (token_embedding_weight.ne[0] == 1024) {
        if (is_inpaint) {
            return VERSION_SD2_INPAINT;
        }
        if (!has_middle_block_1) {
            return has_attn_1024 ? VERSION_SDXS_09 : VERSION_SD2_TINY_UNET;
        }
        return VERSION_SD2;
    }
    return VERSION_COUNT;
}

std::map<ggml_type, uint32_t> ModelLoader::get_wtype_stat() const {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_conditioner_wtype_stat() const {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if ((tensor_storage.name.find("text_encoders") == std::string::npos &&
             tensor_storage.name.find("cond_stage_model") == std::string::npos &&
             tensor_storage.name.find("te.text_model.") == std::string::npos &&
             tensor_storage.name.find("conditioner") == std::string::npos)) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_diffusion_model_wtype_stat() const {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("model.diffusion_model.") == std::string::npos && tensor_storage.name.find("unet.") == std::string::npos) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_vae_wtype_stat() const {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("vae.") == std::string::npos &&
            tensor_storage.name.find("first_stage_model") == std::string::npos) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules) {
    TensorTypeRules result;
    for (const auto& item : split_string(tensor_type_rules, ',')) {
        if (item.size() == 0)
            continue;
        std::string::size_type pos = item.find('=');
        if (pos == std::string::npos) {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
            continue;
        }
        std::string tensor_pattern = item.substr(0, pos);
        std::string type_name      = item.substr(pos + 1);

        ggml_type tensor_type = GGML_TYPE_COUNT;

        if (type_name == "f32") {
            tensor_type = GGML_TYPE_F32;
        } else {
            for (size_t i = 0; i < GGML_TYPE_COUNT; i++) {
                auto trait = ggml_get_type_traits((ggml_type)i);
                if (trait->to_float && trait->type_size && type_name == trait->type_name) {
                    tensor_type = (ggml_type)i;
                }
            }
        }

        if (tensor_type != GGML_TYPE_COUNT) {
            result.emplace_back(tensor_pattern, tensor_type);
        } else {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
        }
    }
    return result;
}

void ModelLoader::set_wtype_override(ggml_type wtype, std::string tensor_type_rules) {
    wtype_override_    = wtype;
    tensor_type_rules_ = tensor_type_rules;
    auto map_rules     = parse_tensor_type_rules(tensor_type_rules);
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (tensor_storage.w4_convrot_kind != W4_CONVROT_NONE) {
            // target type is intrinsic to the packed format, not subject to overrides
            continue;
        }
        tensor_storage.expected_type = GGML_TYPE_COUNT;
        ggml_type dst_type           = wtype;
        for (const auto& tensor_type_rule : map_rules) {
            std::regex pattern(tensor_type_rule.first);
            if (std::regex_search(name, pattern)) {
                dst_type = tensor_type_rule.second;
                break;
            }
        }
        if (dst_type == GGML_TYPE_COUNT) {
            continue;
        }
        if (!tensor_should_be_converted(tensor_storage, dst_type)) {
            continue;
        }
        tensor_storage.expected_type = dst_type;
    }
    invalidate_file_data();
    ++revision_;
}

void ModelLoader::process_model_files(bool enable_mmap, bool writable_mmap) {
    if (model_files_processed) {
        return;
    }

    if (!convert_w4_tensor_storages()) {
        converted_w4_data_.clear();
        throw std::runtime_error("failed to reblock packed w4 convrot tensors");
    }

    std::vector<TensorStorage> processed_tensor_storages;
    for (const auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        processed_tensor_storages.push_back(tensor_storage);
    }

    for (size_t file_index = 0; file_index < file_paths_.size(); file_index++) {
        std::string file_path = file_paths_[file_index];

        std::vector<TensorStorage> file_tensors;
        for (const auto& ts : processed_tensor_storages) {
            if (ts.file_index == file_index) {
                file_tensors.push_back(ts);
            }
        }
        if (file_tensors.empty()) {
            continue;
        }

        bool is_zip = false;
        for (auto const& ts : file_tensors) {
            if (ts.index_in_zip >= 0) {
                is_zip = true;
                break;
            }
        }

        ModelFileData fdata = {};
        fdata.path          = file_path;
        fdata.is_zip        = is_zip;
        fdata.tensors       = std::move(file_tensors);

        if (enable_mmap && !is_zip) {
            LOG_VERBOSE("using mmap for I/O");
            std::unique_ptr<MmapWrapper> mmapped = MmapWrapper::create(file_path, writable_mmap);
            if (mmapped) {
                uint8_t* mmap_data             = static_cast<uint8_t*>(mmapped->writable_data());
                ggml_backend_buffer_t buf_mmap = ggml_backend_cpu_buffer_from_ptr(mmap_data, mmapped->size());
                if (buf_mmap) {
                    LOG_INFO("using mmap for '%s'", file_path.c_str());
                    fdata.mmbuffer = std::shared_ptr<struct ggml_backend_buffer>(buf_mmap, ggml_backend_buffer_free);
                } else {
                    LOG_WARN("mmap: failed to create backend buffer for file %s", fdata.path.c_str());
                }
                fdata.mmapped = std::shared_ptr<MmapWrapper>(std::move(mmapped));
            } else {
                LOG_WARN("failed to memory-map '%s' (falling back to read())", file_path.c_str());
            }
        }

        file_data.push_back(std::move(fdata));
    }

    model_files_processed = true;
}

std::vector<MmapTensorStore> ModelLoader::mmap_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                                       std::set<std::string> ignore_tensors,
                                                       bool writable_mmap) {
    std::set<std::string> names;
    for (const auto& entry : tensors) {
        names.insert(entry.first);
    }
    if (!validate_sources(&names)) {
        return {};
    }
    process_model_files(true, writable_mmap);

    std::vector<MmapTensorStore> result;
    uint64_t mapped_bytes = 0;
    size_t mapped_tensors = 0;

    LOG_VERBOSE("memory-mapping tensors...");

    int64_t t_start = ggml_time_ms();

    for (auto& fdata : file_data) {
        if (!fdata.mmbuffer)
            continue;

        const std::vector<TensorStorage>& file_tensors = fdata.tensors;

        size_t file_mapped_bytes   = 0;
        size_t file_mapped_tensors = 0;

        for (const auto& tensor_storage : file_tensors) {
            const std::string& name = tensor_storage.name;

            bool is_ignored = false;
            for (const auto& ignore_prefix : ignore_tensors) {
                if (starts_with(name, ignore_prefix)) {
                    is_ignored = true;
                    break;
                }
            }
            if (is_ignored)
                continue;

            auto it = tensors.find(name);
            if (it == tensors.end())
                continue;

            ggml_tensor* dst_tensor = it->second;
            if (dst_tensor == nullptr)
                continue;

            if (tensor_storage.is_f64 ||
                tensor_storage.is_i64 ||
                tensor_storage.type != dst_tensor->type) {
                continue;
            }

            size_t tensor_size   = tensor_storage.nbytes();
            size_t tensor_offset = tensor_storage.offset;

            if (tensor_storage.w4_convrot_kind != W4_CONVROT_NONE) {
                // raw file bytes are packed; reblocked bytes are injected by load_tensors
                continue;
            }

            if (tensor_storage.ne[0] != dst_tensor->ne[0] ||
                tensor_storage.ne[1] != dst_tensor->ne[1] ||
                tensor_storage.ne[2] != dst_tensor->ne[2] ||
                tensor_storage.ne[3] != dst_tensor->ne[3] ||
                tensor_size != ggml_nbytes(dst_tensor)) {
                // let load_tensors worry about this
                continue;
            }

            ggml_backend_buffer_t buf_mmap = fdata.mmbuffer.get();
            uint8_t* mmap_data             = static_cast<uint8_t*>(ggml_backend_buffer_get_base(buf_mmap));
            dst_tensor->buffer             = buf_mmap;
            dst_tensor->data               = mmap_data + tensor_offset;

            file_mapped_bytes += tensor_size;
            file_mapped_tensors++;
        }

        if (file_mapped_bytes > 0) {
            mapped_tensors += file_mapped_tensors;
            mapped_bytes += file_mapped_bytes;
            result.push_back({fdata.mmapped, fdata.mmbuffer});
        }
    }

    int64_t t_end       = ggml_time_ms();
    int64_t duration_ms = t_end - t_start;

    LOG_INFO("memory-mapped %zu tensors in %zu files (%.2f MB), taking %.2fs",
             mapped_tensors,
             result.size(),
             mapped_bytes / (1024.0 * 1024.0),
             duration_ms / 1000.0);

    return result;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb,
                               bool enable_mmap,
                               const std::set<std::string>* target_tensor_names,
                               bool log_progress) {
    if (!validate_sources(target_tensor_names)) {
        return false;
    }
    process_model_files(enable_mmap, false);

    std::atomic<int64_t> read_time_ms(0);
    std::atomic<int64_t> memcpy_time_ms(0);
    std::atomic<int64_t> copy_to_backend_time_ms(0);
    std::atomic<int64_t> convert_time_ms(0);
    std::atomic<uint64_t> bytes_processed(0);

    int num_threads_to_use = n_threads_;

    int64_t start_time = ggml_time_ms();

    size_t total_tensors_to_process = 0;
    std::vector<size_t> file_tensors_to_process_counts;
    file_tensors_to_process_counts.reserve(file_data.size());
    for (const auto& fdata : file_data) {
        size_t file_tensors_to_process = 0;
        if (target_tensor_names == nullptr) {
            file_tensors_to_process = fdata.tensors.size();
        } else {
            for (const TensorStorage& tensor_storage : fdata.tensors) {
                if (target_tensor_names->find(tensor_storage.name) != target_tensor_names->end()) {
                    file_tensors_to_process++;
                }
            }
        }
        file_tensors_to_process_counts.push_back(file_tensors_to_process);
        total_tensors_to_process += file_tensors_to_process;
    }

    bool success                   = true;
    size_t total_tensors_processed = 0;
    const int64_t t_start          = start_time;
    int last_n_threads             = 1;
    SDVersion imatrix_version      = (version_ == VERSION_COUNT) ? get_sd_version() : version_;

    for (size_t file_index = 0; file_index < file_data.size(); ++file_index) {
        auto& fdata                  = file_data[file_index];
        const std::string& file_path = fdata.path;

        const std::vector<TensorStorage>& file_tensors = fdata.tensors;
        std::vector<const TensorStorage*> tensors_to_process;
        size_t file_tensors_to_process = file_tensors_to_process_counts[file_index];
        tensors_to_process.reserve(file_tensors_to_process);
        if (target_tensor_names == nullptr) {
            for (const TensorStorage& tensor_storage : file_tensors) {
                tensors_to_process.push_back(&tensor_storage);
            }
        } else {
            for (const TensorStorage& tensor_storage : file_tensors) {
                if (target_tensor_names->find(tensor_storage.name) != target_tensor_names->end()) {
                    tensors_to_process.push_back(&tensor_storage);
                }
            }
        }
        if (tensors_to_process.empty()) {
            continue;
        }
        LOG_VERBOSE("loading %zu/%zu tensors from %s",
                    tensors_to_process.size(),
                    file_tensors.size(),
                    file_path.c_str());

        bool is_zip = fdata.is_zip;

        std::shared_ptr<MmapWrapper> mmapped = fdata.mmapped;

        int n_threads = is_zip ? 1 : std::min(num_threads_to_use, (int)tensors_to_process.size());
        if (n_threads < 1) {
            n_threads = 1;
        }
        last_n_threads = n_threads;

        std::atomic<size_t> tensor_idx(0);
        std::atomic<bool> failed(false);
        std::vector<std::thread> workers;
        std::mutex backend_tensor_set_mutex;

        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([&, file_path, is_zip]() {
                std::ifstream file;
                zip_t* zip = nullptr;
                if (is_zip) {
                    zip = zip_open(file_path.c_str(), 0, 'r');
                    if (zip == nullptr) {
                        LOG_ERROR("failed to open zip '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                } else if (!mmapped) {
                    file.open(file_path, std::ios::binary);
                    if (!file.is_open()) {
                        LOG_ERROR("failed to open '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                }

                std::vector<uint8_t> read_buffer;
                std::vector<uint8_t> convert_buffer;
                std::vector<uint8_t> zip_entry_buffer;

                while (true) {
                    int64_t t0, t1;
                    size_t idx = tensor_idx.fetch_add(1);
                    if (idx >= tensors_to_process.size() || failed) {
                        break;
                    }

                    const TensorStorage& tensor_storage = *tensors_to_process[idx];
                    ggml_tensor* dst_tensor             = nullptr;

                    t0 = ggml_time_ms();

                    if (!on_new_tensor_cb(tensor_storage, &dst_tensor)) {
                        LOG_WARN("process tensor failed: '%s'", tensor_storage.name.c_str());
                        failed = true;
                        break;
                    }

                    if (dst_tensor == nullptr) {
                        t1 = ggml_time_ms();
                        read_time_ms.fetch_add(t1 - t0);
                        continue;
                    }

                    if (dst_tensor->data == nullptr) {
                        LOG_ERROR("process tensor data failed: '%s'", tensor_storage.name.c_str());
                        failed = true;
                        break;
                    }

                    // skip mmapped tensors
                    if (dst_tensor->buffer != nullptr && dst_tensor->buffer == fdata.mmbuffer.get()) {
                        continue;
                    }

                    size_t nbytes_to_read = tensor_storage.nbytes_to_read();

                    auto read_data = [&](char* buf, size_t n) -> bool {
                        if (zip != nullptr) {
                            if (zip_entry_openbyindex(zip, tensor_storage.index_in_zip) != 0) {
                                LOG_ERROR("failed to open zip entry for tensor '%s'", tensor_storage.name.c_str());
                                return false;
                            }
                            size_t entry_size = zip_entry_size(zip);
                            if (tensor_storage.offset > entry_size) {
                                LOG_ERROR("tensor '%s' exceeds its zip storage entry", tensor_storage.name.c_str());
                                zip_entry_close(zip);
                                return false;
                            }
                            size_t tensor_offset = static_cast<size_t>(tensor_storage.offset);
                            if (n > entry_size - tensor_offset) {
                                LOG_ERROR("tensor '%s' exceeds its zip storage entry", tensor_storage.name.c_str());
                                zip_entry_close(zip);
                                return false;
                            }

                            if (entry_size != n) {
                                int64_t t_memcpy_start;
                                zip_entry_buffer.resize(entry_size);
                                auto bytes_read = zip_entry_noallocread(zip, (void*)zip_entry_buffer.data(), entry_size);
                                if (bytes_read < 0 || static_cast<size_t>(bytes_read) != entry_size) {
                                    LOG_ERROR("failed to read zip entry for tensor '%s'", tensor_storage.name.c_str());
                                    zip_entry_close(zip);
                                    return false;
                                }
                                t_memcpy_start = ggml_time_ms();
                                memcpy((void*)buf, (void*)(zip_entry_buffer.data() + tensor_offset), n);
                                memcpy_time_ms.fetch_add(ggml_time_ms() - t_memcpy_start);
                            } else {
                                auto bytes_read = zip_entry_noallocread(zip, (void*)buf, n);
                                if (bytes_read < 0 || static_cast<size_t>(bytes_read) != n) {
                                    LOG_ERROR("failed to read zip entry for tensor '%s'", tensor_storage.name.c_str());
                                    zip_entry_close(zip);
                                    return false;
                                }
                            }
                            zip_entry_close(zip);
                        } else if (mmapped) {
                            if (!mmapped->copy_data(buf, n, tensor_storage.offset)) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                return false;
                            }
                        } else {
                            file.seekg(tensor_storage.offset);
                            file.read(buf, n);
                            if (!file) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                return false;
                            }
                        }
                        return true;
                    };

                    char* read_buf    = nullptr;
                    char* target_buf  = nullptr;
                    char* convert_buf = nullptr;
                    if (dst_tensor->buffer == nullptr || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        if (tensor_storage.type == dst_tensor->type) {
                            GGML_ASSERT(ggml_nbytes(dst_tensor) == tensor_storage.nbytes());
                            if (tensor_storage.is_f64 || tensor_storage.is_i64) {
                                read_buffer.resize(tensor_storage.nbytes_to_read());
                                read_buf = (char*)read_buffer.data();
                            } else {
                                read_buf = (char*)dst_tensor->data;
                            }
                            target_buf = (char*)dst_tensor->data;
                        } else {
                            read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                            read_buf    = (char*)read_buffer.data();
                            target_buf  = read_buf;
                            convert_buf = (char*)dst_tensor->data;
                        }
                    } else {
                        read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                        read_buf   = (char*)read_buffer.data();
                        target_buf = read_buf;

                        if (tensor_storage.type != dst_tensor->type) {
                            convert_buffer.resize(ggml_nbytes(dst_tensor));
                            convert_buf = (char*)convert_buffer.data();
                        }
                    }

                    t0                      = ggml_time_ms();
                    const auto converted_w4 = converted_w4_data_.find(tensor_storage.name);
                    if (converted_w4 != converted_w4_data_.end()) {
                        // reblocked bytes were produced in memory; skip file I/O
                        GGML_ASSERT((size_t)nbytes_to_read == converted_w4->second.size());
                        memcpy(read_buf, converted_w4->second.data(), nbytes_to_read);
                    } else if (!read_data(read_buf, nbytes_to_read)) {
                        failed = true;
                        break;
                    }
                    t1 = ggml_time_ms();
                    read_time_ms.fetch_add(t1 - t0);

                    t0 = ggml_time_ms();
                    if (tensor_storage.is_f64) {
                        f64_to_f32_vec((double*)read_buf, (float*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_i64) {
                        i64_to_i32_vec((int64_t*)read_buf, (int32_t*)target_buf, tensor_storage.nelements());
                    }
                    if (tensor_storage.type != dst_tensor->type) {
                        if (convert_buf == nullptr) {
                            LOG_ERROR("read tensor data failed: too less memory for conversion");
                            failed = true;
                            return;
                        }
                        std::string processed_name = convert_tensor_name(tensor_storage.name, imatrix_version);
                        std::vector<float> imatrix = get_imatrix_collector().get_values(processed_name);
                        convert_tensor((void*)target_buf,
                                       tensor_storage.type,
                                       convert_buf,
                                       dst_tensor->type,
                                       (int)tensor_storage.nelements() / (int)tensor_storage.ne[0],
                                       (int)tensor_storage.ne[0],
                                       std::move(imatrix));
                    } else {
                        convert_buf = read_buf;
                    }
                    t1 = ggml_time_ms();
                    convert_time_ms.fetch_add(t1 - t0);

                    if (dst_tensor->buffer != nullptr && !ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        t0 = ggml_time_ms();

                        std::lock_guard<std::mutex> lock(backend_tensor_set_mutex);
                        ggml_backend_tensor_set(dst_tensor, convert_buf, 0, ggml_nbytes(dst_tensor));

                        t1 = ggml_time_ms();
                        copy_to_backend_time_ms.fetch_add(t1 - t0);
                    }

                    bytes_processed.fetch_add((uint64_t)nbytes_to_read);
                }
                if (zip != nullptr) {
                    zip_close(zip);
                }
            });
        }

        while (true) {
            size_t current_idx = tensor_idx.load();
            if (current_idx >= tensors_to_process.size() || failed) {
                break;
            }
            size_t curr_num       = total_tensors_processed + current_idx;
            float elapsed_seconds = (ggml_time_ms() - t_start) / 1000.0f;
            if (log_progress && total_tensors_to_process > 0) {
                pretty_bytes_progress(static_cast<int>(curr_num),
                                      static_cast<int>(total_tensors_to_process),
                                      bytes_processed.load(),
                                      elapsed_seconds);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(total_tensors_to_process <= 4 ? 10 : 200));
        }

        for (auto& w : workers) {
            w.join();
        }

        if (failed) {
            success = false;
            break;
        }
        total_tensors_processed += tensors_to_process.size();
        if (log_progress && total_tensors_to_process > 0) {
            pretty_bytes_progress(static_cast<int>(total_tensors_processed),
                                  static_cast<int>(total_tensors_to_process),
                                  bytes_processed.load(),
                                  (ggml_time_ms() - t_start) / 1000.0f);
        }
        if (log_progress && total_tensors_processed < total_tensors_to_process && total_tensors_to_process > 0) {
            printf("\n");
        }
    }

    int64_t end_time = ggml_time_ms();
    if (log_progress) {
        LOG_INFO("loading tensors completed, taking %.2fs (read: %.2fs, memcpy: %.2fs, convert: %.2fs, copy_to_backend: %.2fs)",
                 (end_time - start_time) / 1000.f,
                 (read_time_ms.load() / (float)last_n_threads) / 1000.f,
                 (memcpy_time_ms.load() / (float)last_n_threads) / 1000.f,
                 (convert_time_ms.load() / (float)last_n_threads) / 1000.f,
                 (copy_to_backend_time_ms.load() / (float)last_n_threads) / 1000.f);
    }
    return success && validate_sources(target_tensor_names);
}

bool ModelLoader::load_tensor(const TensorStorage& tensor_storage, ggml_tensor* dst_tensor) {
    if (dst_tensor == nullptr || dst_tensor->data == nullptr) {
        LOG_ERROR("load tensor failed: null destination for '%s'", tensor_storage.name.c_str());
        return false;
    }

    bool loaded = false;
    std::set<std::string> target_tensor_names{tensor_storage.name};
    auto on_new_tensor_cb = [&](const TensorStorage& current_tensor_storage, ggml_tensor** out_tensor) -> bool {
        *out_tensor = nullptr;
        if (current_tensor_storage.name != tensor_storage.name) {
            return true;
        }

        if (current_tensor_storage.file_id != tensor_storage.file_id ||
            current_tensor_storage.file_revision != tensor_storage.file_revision ||
            current_tensor_storage.file_index != tensor_storage.file_index ||
            current_tensor_storage.offset != tensor_storage.offset ||
            current_tensor_storage.index_in_zip != tensor_storage.index_in_zip) {
            LOG_ERROR("load tensor failed: storage mismatch for '%s'", tensor_storage.name.c_str());
            return false;
        }

        if (current_tensor_storage.n_dims != tensor_storage.n_dims ||
            current_tensor_storage.nelements() != tensor_storage.nelements()) {
            LOG_ERROR("load tensor failed: metadata changed for '%s'", tensor_storage.name.c_str());
            return false;
        }

        for (int i = 0; i < current_tensor_storage.n_dims; i++) {
            if (current_tensor_storage.ne[i] != dst_tensor->ne[i]) {
                LOG_ERROR("load tensor failed: shape mismatch for '%s'", tensor_storage.name.c_str());
                return false;
            }
        }

        *out_tensor = dst_tensor;
        loaded      = true;
        return true;
    };

    if (!load_tensors(on_new_tensor_cb, false, &target_tensor_names, false)) {
        LOG_ERROR("load tensor failed: '%s'", tensor_storage.name.c_str());
        return false;
    }

    if (!loaded) {
        LOG_ERROR("load tensor failed: tensor '%s' not found", tensor_storage.name.c_str());
        return false;
    }

    return true;
}

bool ModelLoader::load_float_tensor(const std::string& name,
                                    std::vector<float>& data,
                                    int n_threads,
                                    bool use_mmap) {
    data.clear();

    auto tensor_storage_it = tensor_storage_map.find(name);
    if (tensor_storage_it == tensor_storage_map.end()) {
        return false;
    }

    const TensorStorage& tensor_storage = tensor_storage_it->second;
    int64_t n_elements                  = tensor_storage.nelements();
    if (n_elements <= 0) {
        LOG_ERROR("tensor '%s' has invalid element count: %" PRId64, name.c_str(), n_elements);
        return false;
    }
    if (tensor_storage.n_dims <= 0 || tensor_storage.n_dims > GGML_MAX_DIMS) {
        LOG_ERROR("tensor '%s' has unsupported dims: %d", name.c_str(), tensor_storage.n_dims);
        return false;
    }

    std::vector<float> loaded_data(static_cast<size_t>(n_elements));
    ggml_init_params params;
    params.mem_size   = ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        LOG_ERROR("failed to create context for tensor '%s'", name.c_str());
        return false;
    }

    ggml_tensor* tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, tensor_storage.n_dims, tensor_storage.ne);
    ggml_set_name(tensor, name.c_str());
    tensor->data = loaded_data.data();

    bool loaded           = false;
    auto on_new_tensor_cb = [&](const TensorStorage& current_tensor_storage, ggml_tensor** dst_tensor) -> bool {
        *dst_tensor = nullptr;
        if (current_tensor_storage.name != name) {
            return true;
        }
        if (current_tensor_storage.nelements() != n_elements) {
            LOG_ERROR("tensor '%s' element count changed during load", name.c_str());
            return false;
        }
        *dst_tensor = tensor;
        loaded      = true;
        return true;
    };

    std::set<std::string> target_tensor_names{name};
    if (n_threads > 0) {
        set_n_threads(n_threads);
    }
    bool success = load_tensors(on_new_tensor_cb, use_mmap, &target_tensor_names);
    ggml_free(ctx);

    if (!success || !loaded) {
        data.clear();
        return false;
    }

    data = std::move(loaded_data);
    return true;
}

bool ModelLoader::load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               std::set<std::string> ignore_tensors,
                               bool enable_mmap) {
    std::set<std::string> tensor_names_in_file;
    std::mutex tensor_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        // LOG_VERBOSE("%s", tensor_storage.to_string().c_str());
        {
            std::lock_guard<std::mutex> lock(tensor_names_mutex);
            tensor_names_in_file.insert(name);
        }

        ggml_tensor* real;
        if (tensors.find(name) != tensors.end()) {
            real = tensors[name];
        } else {
            for (auto& ignore_tensor : ignore_tensors) {
                if (starts_with(name, ignore_tensor)) {
                    return true;
                }
            }
            LOG_INFO("unknown tensor '%s' in model file", tensor_storage.to_string().c_str());
            return true;
        }

        if (
            real->ne[0] != tensor_storage.ne[0] ||
            real->ne[1] != tensor_storage.ne[1] ||
            real->ne[2] != tensor_storage.ne[2] ||
            real->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "tensor '%s' has wrong shape in model file: "
                "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
            return false;
        }

        *dst_tensor = real;

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, enable_mmap);
    if (!success) {
        LOG_ERROR("load tensors from file failed");
        return false;
    }

    bool some_tensor_not_init = false;

    for (auto pair : tensors) {
        if (pair.first.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos) {
            continue;
        }

        if (pair.first.find("alphas_cumprod") != std::string::npos) {
            continue;
        }

        if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
            LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
            some_tensor_not_init = true;
        }
    }

    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) const {
    const std::string& name = tensor_storage.name;
    if (tensor_storage.is_int8_tensorwise) {
        return false;
    }
    // Packed ComfyUI w4 convrot weights are reblocked to Q4_0/IQ4_NL by the loader
    // itself; never route them through the generic conversion machinery.
    if (tensor_storage.w4_convrot_kind != W4_CONVROT_NONE) {
        return false;
    }
    if (ends_with(name, ".weight_codebook") || ends_with(name, ".weight_s_rel") ||
        ends_with(name, ".weight_s_channel")) {
        // Pass, do not convert (w4 convrot companions)
    } else if (type != GGML_TYPE_COUNT) {
        if (ggml_is_quantized(type) && tensor_storage.ne[0] % ggml_blck_size(type) != 0) {
            // Pass, do not convert
        } else if (ends_with(name, ".bias")) {
            // Pass, do not convert
        } else if (ends_with(name, ".scale")) {
            // Pass, do not convert
        } else if (ends_with(name, ".weight_scale")) {
            // Pass, do not convert
        } else if (contains(name, "img_in.") ||
                   contains(name, "txt_in.") ||
                   contains(name, "time_in.") ||
                   contains(name, "vector_in.") ||
                   contains(name, "guidance_in.") ||
                   contains(name, "final_layer.")) {
            // Pass, do not convert. For FLUX
        } else if (contains(name, "x_embedder.") ||
                   contains(name, "t_embedder.") ||
                   contains(name, "y_embedder.") ||
                   contains(name, "pos_embed") ||
                   contains(name, "context_embedder.")) {
            // Pass, do not convert. For MMDiT
        } else if (contains(name, "time_embed.") || contains(name, "label_emb.")) {
            // Pass, do not convert. For Unet
        } else if (contains(name, "embedding")) {
            // Pass, do not convert embedding
        } else {
            return true;
        }
    }
    return false;
}

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend, ggml_type type) const {
    size_t alignment = 128;
    if (backend != nullptr) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;
    std::vector<TensorStorage> processed_tensor_storages;
    for (auto [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_storage.type = type;
        }
        mem_size += tensor_storage.nbytes() + alignment;
    }

    return mem_size;
}
