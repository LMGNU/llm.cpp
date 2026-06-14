#pragma once

#include "config/config.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

// DataLoader handles BPE tokenization, corpus loading, and batch generation.
struct DataLoader
{
    std::vector<std::string>          vocab;        // id to token string
    std::map<std::string, int>        token_to_id;  // token string to id
    std::vector<std::pair<int, int>>  merges;       // merge rules in training order
    std::map<std::pair<int,int>, int> merge_rank;   // pair to training step index
    int base_vocab_size{0};                         // number of single-char tokens
    int vocab_size{0};

    std::vector<int> train_data;
    std::vector<int> val_data;

    // Load text file and train BPE tokenizer on it.
    // Splits into train and validation sets.
    void load(const std::string &path,
              int    target_vocab = BPE_VOCAB_SIZE,
              double train_split  = TRAIN_SPLIT)
    {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("[DataLoader] Cannot open file: " + path);

        std::ostringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        if (text.empty())
            throw std::runtime_error("[DataLoader] File is empty: " + path);

        std::cout << "[BPE]   Text length: " << text.size() << " characters\n";
        std::cout << "[BPE]   Target vocab size: " << target_vocab << "\n";
        std::cout.flush();

        std::vector<int> data = train_bpe(text, target_vocab);

        int n = (int)(train_split * (double)data.size());
        train_data = std::vector<int>(data.begin(), data.begin() + n);
        val_data   = std::vector<int>(data.begin() + n, data.end());

        if ((int)train_data.size() <= BLOCK_SIZE ||
            (int)val_data.size()   <= BLOCK_SIZE)
            throw std::runtime_error(
                "[DataLoader] Dataset too small for BLOCK_SIZE=" +
                std::to_string(BLOCK_SIZE));

        std::cout << "[DATA]  Total tokens : " << data.size()       << "\n";
        std::cout << "[DATA]  Train tokens : " << train_data.size() << "\n";
        std::cout << "[DATA]  Val tokens   : " << val_data.size()   << "\n";
    }

    // Encode a string to BPE token ids.
    // Characters not found in base vocab are skipped silently.
    std::vector<int> encode(const std::string &text) const
    {
        return apply_merges(base_encode(text));
    }

    // Decode BPE token ids back to the original string.
    std::string decode(const std::vector<int> &ids) const
    {
        std::string out;
        for (int id : ids)
            if (id >= 0 && id < (int)vocab.size())
                out += vocab[id];
        return out;
    }

    // Get a random batch of (input, target) token pairs.
    // Input is [batch_size * block_size] tokens.
    // Target is the next token for each input position.
    std::pair<std::vector<int>, std::vector<int>>
    get_batch(const std::string &split, int batch_size, int block_size,
              std::mt19937 &rng) const
    {
        const std::vector<int> &d = (split == "train") ? train_data : val_data;
        std::uniform_int_distribution<int> dist(0, (int)d.size() - block_size - 1);

        std::vector<int> x(batch_size * block_size);
        std::vector<int> y(batch_size * block_size);

        for (int b = 0; b < batch_size; ++b) {
            int start = dist(rng);
            for (int t = 0; t < block_size; ++t) {
                x[b * block_size + t] = d[start + t];
                y[b * block_size + t] = d[start + t + 1];
            }
        }
        return {x, y};
    }

private:

    // Convert a string to char-level token ids.
    std::vector<int> base_encode(const std::string &text) const
    {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            auto it = token_to_id.find(std::string(1, c));
            if (it != token_to_id.end())
                ids.push_back(it->second);
        }
        return ids;
    }

    // Apply all BPE merge rules in training order to a token sequence.
    std::vector<int> apply_merges(std::vector<int> ids) const
    {
        for (int rank = 0; rank < (int)merges.size(); ++rank) {
            int left  = merges[rank].first;
            int right = merges[rank].second;
            int new_id = base_vocab_size + rank;

            std::vector<int> out;
            out.reserve(ids.size());
            int i = 0;
            while (i < (int)ids.size()) {
                if (i + 1 < (int)ids.size() &&
                    ids[i] == left && ids[i+1] == right)
                {
                    out.push_back(new_id);
                    i += 2;
                } else {
                    out.push_back(ids[i]);
                    ++i;
                }
            }
            ids = std::move(out);
        }
        return ids;
    }

    // Train BPE on the input text.
    // Returns the final BPE-encoded token sequence for the full corpus.
    std::vector<int> train_bpe(const std::string &text, int target_vocab)
    {
        // Build base vocabulary from all unique characters.
        std::set<char> chars(text.begin(), text.end());
        for (char c : chars) {
            std::string s(1, c);
            token_to_id[s] = (int)vocab.size();
            vocab.push_back(s);
        }
        base_vocab_size = (int)vocab.size();

        if (target_vocab <= base_vocab_size) {
            vocab_size = base_vocab_size;
            return base_encode(text);
        }

        // Encode full text as base character-level tokens.
        std::vector<int> ids = base_encode(text);

        int num_merges = target_vocab - base_vocab_size;
        merges.reserve(num_merges);
        std::cout << "[BPE]   Running " << num_merges << " merges...\n";
        std::cout.flush();

        for (int step = 0; step < num_merges; ++step)
        {
            // Count frequency of all adjacent token pairs.
            std::unordered_map<long long, int> counts;
            counts.reserve(ids.size());
            for (int i = 0; i + 1 < (int)ids.size(); ++i) {
                long long key = ((long long)ids[i] << 32) | (unsigned int)ids[i+1];
                ++counts[key];
            }

            if (counts.empty()) break;

            // Find the most frequent pair.
            long long best_key = 0;
            int       best_cnt = -1;
            for (auto &kv : counts) {
                if (kv.second > best_cnt) {
                    best_cnt = kv.second;
                    best_key = kv.first;
                }
            }

            int left  = (int)(best_key >> 32);
            int right = (int)(best_key & 0xFFFFFFFFLL);

            // Create a new token for this merge.
            int new_id = (int)vocab.size();
            std::string new_tok = vocab[left] + vocab[right];
            vocab.push_back(new_tok);
            token_to_id[new_tok] = new_id;
            merges.push_back({left, right});
            merge_rank[{left, right}] = step;

            // Apply the merge everywhere in the sequence.
            std::vector<int> out;
            out.reserve(ids.size());
            int i = 0;
            while (i < (int)ids.size()) {
                if (i + 1 < (int)ids.size() &&
                    ids[i] == left && ids[i+1] == right)
                {
                    out.push_back(new_id);
                    i += 2;
                } else {
                    out.push_back(ids[i]);
                    ++i;
                }
            }
            ids = std::move(out);

            // Print progress every 500 merges.
            if ((step + 1) % 500 == 0 || step + 1 == num_merges) {
                std::cout << "[BPE]   step " << (step + 1) << "/" << num_merges
                          << "  vocab=" << (int)vocab.size()
                          << "  seq=" << ids.size() << "\n";
                std::cout.flush();
            }
        }

        vocab_size = (int)vocab.size();
        std::cout << "[BPE]   Done. Final vocab size: " << vocab_size << "\n";
        return ids;
    }
};
