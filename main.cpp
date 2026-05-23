#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <csignal>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "config/config.h"
#include "include/dataloader.h"
#include "include/gpt.h"
#include "include/backward.h"

// Signal handler (Ctrl-C stops generation gracefully)
static volatile bool g_interrupted = false;
static void sig_handler(int) { g_interrupted = true; }

// Timing helpers
static std::string now_str()
{
      std::time_t t = std::time(nullptr);
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
      return buf;
}

static double wall_secs()
{
      using namespace std::chrono;
      return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static bool file_exists(const std::string &path)
{
      std::ifstream f(path.c_str(), std::ios::binary);
      return f.good();
}

static std::string dir_name(const std::string &path)
{
      std::string::size_type pos = path.find_last_of("/\\");
      if (pos == std::string::npos)
            return ".";
      if (pos == 0)
            return path.substr(0, 1);
      return path.substr(0, pos);
}

static bool is_absolute_path(const std::string &path)
{
      if (path.empty())
            return false;
      if (path.size() > 1 && path[1] == ':')
            return true;
      return path[0] == '/' || path[0] == '\\';
}

static std::string join_path(const std::string &base, const std::string &child)
{
      if (base.empty() || base == ".")
            return child;
      char last = base[base.size() - 1];
      if (last == '/' || last == '\\')
            return base + child;
      return base + "/" + child;
}

static std::string choose_existing_path(const std::string &requested_path,
                                        const std::string &argv0)
{
      if (requested_path.empty())
            return requested_path;
      if (file_exists(requested_path))
            return requested_path;
      if (is_absolute_path(requested_path))
            return requested_path;

      std::vector<std::string> candidates;
      candidates.push_back(join_path(dir_name(argv0), requested_path));
      candidates.push_back(join_path(".", requested_path));

      for (size_t i = 0; i < candidates.size(); ++i)
      {
            if (file_exists(candidates[i]))
                  return candidates[i];
      }
      return requested_path;
}

static std::string choose_output_path(const std::string &requested_path,
                                      const std::string &argv0)
{
      if (requested_path.empty() || is_absolute_path(requested_path))
            return requested_path;

      std::string exe_relative = join_path(dir_name(argv0), requested_path);
      if (file_exists(requested_path) || !file_exists(exe_relative))
            return requested_path;
      return exe_relative;
}

// estimate loss — no gradients, training=false
static float estimate_loss(GPTLanguageModel &model,
                           DataLoader &dl,
                           const std::string &split,
                           std::mt19937 &rng)
{
      float total = 0.0f;
      for (int k = 0; k < EVAL_ITERS; ++k)
      {
            std::pair<std::vector<int>, std::vector<int>> batch =
                dl.get_batch(split, BATCH_SIZE, BLOCK_SIZE, rng);
            std::pair<Tensor, float> result =
                model.forward(batch.first, BATCH_SIZE, BLOCK_SIZE, batch.second, false);
            total += result.second;
      }
      return total / EVAL_ITERS;
}

// Chat window
static void run_chat(GPTLanguageModel &model,
                     DataLoader &dl,
                     int max_new_tokens)
{
      std::cout << "\n"
                << std::string(60, '=') << "\n";
      std::cout << "  Quadtrix CHAT MODE\n";
      std::cout << "  Type your prompt and press Enter. "
                   "Type 'quit' or 'exit' to leave.\n";
      std::cout << std::string(60, '=') << "\n\n";

      while (!g_interrupted)
      {
            std::cout << "\033[1;32mYou>\033[0m ";
            std::cout.flush();

            std::string prompt;
            if (!std::getline(std::cin, prompt))
                  break;

            size_t s = prompt.find_first_not_of(" \t\r\n");
            size_t e = prompt.find_last_not_of(" \t\r\n");
            if (s == std::string::npos)
                  continue;
            prompt = prompt.substr(s, e - s + 1);

            if (prompt == "quit" || prompt == "exit")
            {
                  std::cout << "[Chat] Bye!\n";
                  break;
            }

            std::vector<int> ctx = dl.encode(prompt);
            if (ctx.empty())
            {
                  ctx = {0};
            }

            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());

            std::cout << "\033[1;36mQuadtrix>\033[0m ";
            std::cout.flush();

            for (int tok = 0; tok < max_new_tokens && !g_interrupted; ++tok)
            {
                  ctx = model.generate(ctx, 1);
                  std::cout << dl.decode({ctx.back()}) << std::flush;

                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n\n";
      }
}

// Main
int main(int argc, char *argv[])
{
      std::signal(SIGINT, sig_handler);

      // Banner
      std::cout << std::string(60, '=') << "\n";
      std::cout << " Quadtrix v1.0 (C++)\n";
      std::cout << std::string(60, '=') << "\n";
      std::cout << "\n[INFO] Starting at: " << now_str() << "\n";

      std::string data_path = DEFAULT_CLEANED_PATH;
      const char *env_data_path = std::getenv(DATA_PATH_ENV_VAR.c_str());
      if (env_data_path != nullptr && env_data_path[0] != '\0')
            data_path = env_data_path;

      std::string model_path = BEST_MODEL_PATH;
      const char *env_model_path = std::getenv(MODEL_PATH_ENV_VAR.c_str());
      if (env_model_path != nullptr && env_model_path[0] != '\0')
            model_path = env_model_path;

      bool gen_mode = false;
      bool chat_mode = false;
      int chat_tokens = 200;

      for (int i = 1; i < argc; ++i)
      {
            std::string a = argv[i];
            if (a == "--generate")
                  gen_mode = true;
            else if (a == "--chat")
                  chat_mode = true;
            else if (a == "--chat-tokens" && i + 1 < argc)
                  chat_tokens = std::atoi(argv[++i]);
            else
                  data_path = a;
      }

      data_path = choose_existing_path(data_path, argv[0]);
      model_path = choose_output_path(model_path, argv[0]);

      // Config print
      std::cout << "\n[CONFIG] Hyperparameters:\n";
      std::cout << "         batch_size=" << BATCH_SIZE
                << "  block_size=" << BLOCK_SIZE << "\n";
      std::cout << "         max_iters=" << MAX_ITERS
                << "  learning_rate=" << LEARNING_RATE << "\n";
      std::cout << "         n_embd=" << N_EMBD
                << "  n_head=" << N_HEAD
                << "  n_layer=" << N_LAYER
                << "  dropout=" << DROPOUT << "\n";

      //  Data
      DataLoader dl;
      try
      {
            dl.load(data_path);
      }
      catch (const std::exception &e)
      {
            std::cerr << e.what() << "\n";
            std::cerr << "[HINT]  Put your text at " << DEFAULT_CLEANED_PATH
                      << ", pass a file path as the first argument, or set "
                      << DATA_PATH_ENV_VAR << ".\n";
            return 1;
      }
      GPTLanguageModel model(dl.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, SEED);

      long n_params = model.num_params();
      std::cout << "[MODEL] Parameters  : "
                << std::fixed << std::setprecision(2)
                << n_params / 1.0e6f << " M  (" << n_params << " total)\n";
      std::cout << "[MODEL] Architecture: "
                << N_LAYER << " layers x "
                << N_HEAD << " heads x "
                << N_EMBD << " embedding dim\n";

      // chat mode
      if (chat_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot start chat because model weights were not found at "
                            << model_path << "\n";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.\n";
                  return 1;
            }

            model.load(model_path);
            std::cout << "[CHAT]  Weights loaded from " << model_path << "\n";
            std::cout << "[CHAT]  Max tokens per reply: " << chat_tokens
                      << "  (override with --chat-tokens N)\n";

            run_chat(model, dl, chat_tokens);
            return 0;
      }

      // Generate-only mode
      if (gen_mode)
      {
            if (!file_exists(model_path))
            {
                  std::cerr << "[ERROR] Cannot generate because model weights were not found at "
                            << model_path << "\n";
                  std::cerr << "[HINT]  Train first, or set " << MODEL_PATH_ENV_VAR
                            << " to an existing weights file.\n";
                  return 1;
            }

            model.load(model_path);
            std::cout << "\n"
                      << std::string(60, '-') << "\n";
            std::cout << "  Quadtrix OUTPUT  (Ctrl+C to stop)\n";
            std::cout << std::string(60, '-') << "\n\n";
            std::vector<int> ctx = {0};
            while (!g_interrupted)
            {
                  ctx = model.generate(ctx, 1);
                  std::cout << dl.decode({ctx.back()}) << std::flush;
                  if ((int)ctx.size() > BLOCK_SIZE)
                        ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
            }
            std::cout << "\n\n[Stopped by user]\n";
            return 0;
      }

      // optimizer
      AdamWState opt = build_optimizer(model, LEARNING_RATE);

      // Separate RNG for batch sampling
      std::mt19937 rng(SEED);

      // training loop
      std::cout << "\n"
                << std::string(60, '-') << "\n";
      std::cout << "  TRAINING  ("
                << MAX_ITERS << " iters, eval every "
                << EVAL_INTERVAL << ")\n";
      std::cout << std::string(60, '-') << "\n";

      float best_val_loss = 1e30f;
      double train_start = wall_secs();
      double last_eval_time = train_start; // ← tracks time of previous eval

      for (int iter = 0; iter <= MAX_ITERS && !g_interrupted; ++iter)
      {

            // Periodic eval checkpoint
            if (iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS)
            {
                  double now = wall_secs();
                  double elapsed = now - train_start;

                  // ms per training step since the last eval window
                  double window_secs = now - last_eval_time;
                  int steps_in_win = (iter == 0) ? 1 : EVAL_INTERVAL;
                  double ms_per_step = window_secs * 1000.0 / steps_in_win;

                  // tokens processed per second
                  long toks_in_win = (long)BATCH_SIZE * BLOCK_SIZE * steps_in_win;
                  int tok_per_sec = (window_secs > 0.0)
                                        ? (int)(toks_in_win / window_secs)
                                        : 0;

                  last_eval_time = now; // reset window

                  float tl = estimate_loss(model, dl, "train", rng);
                  float vl = estimate_loss(model, dl, "val", rng);

                  bool better = vl < best_val_loss;
                  if (better)
                  {
                        best_val_loss = vl;
                        model.save(model_path);
                  }

                  // ── new log line ─────────────────────────────────────────────
                  std::cout
                      << "step "
                      << std::setw(5) << iter << "/" << MAX_ITERS
                      << " | loss "
                      << std::fixed << std::setprecision(6) << tl
                      << " | val "
                      << std::fixed << std::setprecision(6) << vl
                      << " | lr "
                      << std::scientific << std::setprecision(2) << (float)LEARNING_RATE
                      << " | "
                      << std::fixed << std::setprecision(2) << ms_per_step << " ms"
                      << " | " << tok_per_sec << " tok/s"
                      << (better ? "  *best*" : "")
                      << "\n";
                  std::cout.flush();

                  if (iter == MAX_ITERS)
                        break;
            }

            // Sample training batch
            std::pair<std::vector<int>, std::vector<int>> batch =
                dl.get_batch("train", BATCH_SIZE, BLOCK_SIZE, rng);

            // Forward — saves all intermediate activations
            SavedForward saved = forward_save(model,
                                              batch.first, BATCH_SIZE, BLOCK_SIZE,
                                              batch.second, /*training=*/true);

            //  Backward — exact analytical gradients
            Grads grads = backward(model, saved);

            // AdamW parameter update
            apply_grads(model, grads, opt);
      }

      double total = wall_secs() - train_start;
      std::cout << "\n[DONE]  Training finished in "
                << std::fixed << std::setprecision(1) << total << "s ("
                << total / 60.0 << " min)  |  Best val loss: "
                << std::setprecision(4) << best_val_loss << "\n";
      std::cout << "[SAVE]  Best weights saved to " << model_path << "\n";

      //  Continuous generation
      std::cout << "\n"
                << std::string(60, '-') << "\n";
      std::cout << "  MODEL OUTPUT  (Ctrl+C to stop)\n";
      std::cout << std::string(60, '-') << "\n\n";

      model.load(model_path);
      model.rng = std::mt19937(SEED + 42);

      std::vector<int> ctx = {0};
      while (!g_interrupted)
      {
            ctx = model.generate(ctx, 1);
            std::cout << dl.decode({ctx.back()}) << std::flush;
            if ((int)ctx.size() > BLOCK_SIZE)
                  ctx = std::vector<int>(ctx.end() - BLOCK_SIZE, ctx.end());
      }

      std::cout << "\n\n[Stopped by user]\n";
      std::cout << "[TOTAL] Wall-clock: "
                << std::fixed << std::setprecision(1)
                << (wall_secs() - train_start) << "s\n";
      return 0;
}