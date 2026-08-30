#ifndef __MEDIA_IO_H__
#define __MEDIA_IO_H__

#include <cstdint>
#include <string>
#include <vector>

#include "stable-diffusion.h"

enum class EncodedImageFormat {
    JPEG,
    PNG,
    WEBP,
    UNKNOWN,
};

EncodedImageFormat encoded_image_format_from_path(const std::string& path);

std::vector<uint8_t> encode_image_to_vector(EncodedImageFormat format,
                                            const uint8_t* image,
                                            int width,
                                            int height,
                                            int channels,
                                            const std::string& parameters = "",
                                            int quality                   = 90);

bool write_image_to_file(const std::string& path,
                         const uint8_t* image,
                         int width,
                         int height,
                         int channels,
                         const std::string& parameters = "",
                         int quality                   = 90);

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width   = 0,
                              int expected_height  = 0,
                              int expected_channel = 3);

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width   = 0,
                             int expected_height  = 0,
                             int expected_channel = 3);

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width   = 0,
                                int expected_height  = 0,
                                int expected_channel = 3);

int create_mjpg_avi_from_sd_images(const char* filename,
                                   sd_image_t* images,
                                   int num_images,
                                   int fps,
                                   int quality             = 90,
                                   const sd_audio_t* audio = nullptr);
std::vector<uint8_t> create_mjpg_avi_from_sd_images_to_vector(sd_image_t* images,
                                                              int num_images,
                                                              int fps,
                                                              int quality             = 90,
                                                              const sd_audio_t* audio = nullptr);

#ifdef SD_USE_WEBP
int create_animated_webp_from_sd_images(const char* filename,
                                        sd_image_t* images,
                                        int num_images,
                                        int fps,
                                        int quality = 90);
std::vector<uint8_t> create_animated_webp_from_sd_images_to_vector(sd_image_t* images,
                                                                   int num_images,
                                                                   int fps,
                                                                   int quality = 90);
#endif

#ifdef SD_USE_WEBM
int create_webm_from_sd_images(const char* filename,
                               sd_image_t* images,
                               int num_images,
                               int fps,
                               int quality             = 90,
                               const sd_audio_t* audio = nullptr);
std::vector<uint8_t> create_webm_from_sd_images_to_vector(sd_image_t* images,
                                                          int num_images,
                                                          int fps,
                                                          int quality             = 90,
                                                          const sd_audio_t* audio = nullptr);
#endif

int create_video_from_sd_images(const char* filename,
                                sd_image_t* images,
                                int num_images,
                                int fps,
                                int quality             = 90,
                                const sd_audio_t* audio = nullptr);
std::vector<uint8_t> create_video_from_sd_images_to_vector(const std::string& output_format,
                                                           sd_image_t* images,
                                                           int num_images,
                                                           int fps,
                                                           int quality             = 90,
                                                           const sd_audio_t* audio = nullptr);

bool write_wav_to_file(const std::string& path,
                       const float* interleaved_samples,
                       uint64_t sample_count,
                       uint32_t channels,
                       uint32_t sample_rate);

bool load_wav_from_file(const std::string& path,
                        std::vector<float>& interleaved_samples,
                        uint32_t& sample_rate,
                        uint32_t& channels);

// Downmix interleaved samples to mono by averaging channels. Returns empty vector on invalid input.
std::vector<float> downmix_to_mono(const float* interleaved_samples,
                                   uint64_t sample_count,
                                   uint32_t channels);

// Band-limited sinc resampler matching torchaudio.functional.resample
// (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99). Returns the input unchanged
// when sample rates are equal, and an empty vector on invalid input.
std::vector<float> resample_audio(const float* samples,
                                  uint64_t sample_count,
                                  uint32_t orig_sample_rate,
                                  uint32_t target_sample_rate);

// Convenience: load WAV, downmix to mono and resample to the target sample rate.
bool load_wav_from_file_mono(const std::string& path,
                             std::vector<float>& mono_samples,
                             uint32_t target_sample_rate = 16000);

#endif  // __MEDIA_IO_H__
