#include "sense-voice.h"
#include "silero-vad.h"
#include "common.h"
#include <vector>
#include <string>

// LSTM State Constants for VAD
#define VAD_LSTM_STATE_MEMORY_SIZE 2048
#define VAD_LSTM_STATE_DIM 128

extern "C" {

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT struct sense_voice_context* sv_init(const char* model_path) {
    sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu = true;
    cparams.use_itn = false;
    cparams.flash_attn = false;

    struct sense_voice_context* ctx = sense_voice_small_init_from_file_with_params(model_path, cparams);
    if (ctx) {
        ctx->language_id = sense_voice_lang_id("auto");
    }
    return ctx;
}

EXPORT void sv_free(struct sense_voice_context* ctx) {
    if (ctx) {
        sense_voice_free(ctx);
    }
}

EXPORT void sv_vad_reset(struct sense_voice_context* ctx) {
    if (!ctx || !ctx->state) return;
    
    if (!ctx->state->vad_ctx) {
        ctx->state->vad_ctx = ggml_init({VAD_LSTM_STATE_MEMORY_SIZE, nullptr, true});
        ctx->state->vad_lstm_context = ggml_new_tensor_1d(ctx->state->vad_ctx, GGML_TYPE_F32, VAD_LSTM_STATE_DIM);
        ctx->state->vad_lstm_hidden_state = ggml_new_tensor_1d(ctx->state->vad_ctx, GGML_TYPE_F32, VAD_LSTM_STATE_DIM);

        ctx->state->vad_lstm_context_buffer = ggml_backend_alloc_buffer(ctx->state->backends[0],
                                                                        ggml_nbytes(ctx->state->vad_lstm_context)
                                                                                + ggml_backend_get_alignment(ctx->state->backends[0]));
        ctx->state->vad_lstm_hidden_state_buffer = ggml_backend_alloc_buffer(ctx->state->backends[0],
                                                                             ggml_nbytes(ctx->state->vad_lstm_hidden_state)
                                                                                     + ggml_backend_get_alignment(ctx->state->backends[0]));
        auto context_alloc = ggml_tallocr_new(ctx->state->vad_lstm_context_buffer);
        ggml_tallocr_alloc(&context_alloc, ctx->state->vad_lstm_context);

        auto state_alloc = ggml_tallocr_new(ctx->state->vad_lstm_hidden_state_buffer);
        ggml_tallocr_alloc(&state_alloc, ctx->state->vad_lstm_hidden_state);
    }
    
    ggml_set_zero(ctx->state->vad_lstm_context);
    ggml_set_zero(ctx->state->vad_lstm_hidden_state);
}

EXPORT float sv_vad_process(struct sense_voice_context* ctx, const float* chunk, int length, int num_threads) {
    if (!ctx || !ctx->state || length != 640) return 0.0f;
    std::vector<float> chunk_vec(chunk, chunk + length);
    float speech_prob = 0.0f;
    silero_vad_encode_internal(*ctx, *ctx->state, chunk_vec, num_threads, speech_prob);
    return speech_prob;
}

EXPORT const char* sv_recognize(struct sense_voice_context* ctx, const float* samples, int length, int num_threads) {
    if (!ctx || !ctx->state || length == 0) return "";
    
    std::vector<double> pcmf32(length);
    for(int i = 0; i < length; i++) {
        pcmf32[i] = (double)samples[i];
    }

    sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    wparams.n_threads = num_threads;
    wparams.language = "auto";
    
    ctx->state->ids.clear();
    
    if (sense_voice_full_parallel(ctx, wparams, pcmf32, length, 1) != 0) {
        return "";
    }

    static std::string result_str;
    result_str.clear();
    bool need_prefix = false;
    for (size_t i = (need_prefix ? 0 : 4); i < ctx->state->ids.size(); i++) {
        int id = ctx->state->ids[i];
        if (i > 0 && ctx->state->ids[i - 1] == ctx->state->ids[i])
            continue;
        if (id)
            result_str += ctx->vocab.id_to_token[id];
    }
    
    return result_str.c_str();
}

}
