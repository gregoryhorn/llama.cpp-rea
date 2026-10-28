// Differential correctness and large-vocabulary CPU penalty performance probe.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

int main(int argc, char ** argv) {
    constexpr int nv = 248320;
    std::mt19937 rng(42);
    for (int mode = 0; mode < 5; ++mode) {
        auto * s = llama_sampler_init_penalties(nv, 64, 1.1f, 0.2f, 1.1f);
        std::vector<int> history;
        for (int i = 0; i < 96; ++i) {
            int t = rng() % nv;
            if (i % 3 == 0) t = 17;
            history.push_back(t);
            llama_sampler_accept(s, t);
        }
        if (mode == 3) { auto * copy = llama_sampler_clone(s); llama_sampler_free(s); s = copy; }
        if (mode == 4) { llama_sampler_reset(s); history.clear(); }
        std::unordered_map<int,int> counts;
        for (size_t i = history.size() > 64 ? history.size() - 64 : 0; i < history.size(); ++i) ++counts[history[i]];
        std::vector<llama_token_data> data;
        for (int i = 0; i < (mode == 2 ? 100 : nv); ++i) data.push_back({i, float(int(rng() % 1000)-500)/100.f, 0.f});
        if (mode == 1) std::shuffle(data.begin(), data.end(), rng);
        auto expected = data;
        for (auto & d : expected) {
            auto it = counts.find(d.id);
            if (it == counts.end()) continue;
            d.logit = d.logit <= 0 ? d.logit * 1.1f : d.logit / 1.1f;
            d.logit -= float(it->second) * 0.2f + 1.1f;
        }
        llama_token_data_array a{data.data(), data.size(), -1, false};
        llama_sampler_apply(s, &a);
        for (size_t i = 0; i < data.size(); ++i) if (data[i].id != expected[i].id || data[i].logit != expected[i].logit) {
            std::fprintf(stderr, "FAIL mode=%d index=%zu got=%g expected=%g\n",mode,i,data[i].logit,expected[i].logit); return 1;
        }
        llama_sampler_free(s);
    }
    // Exercise scratch reuse across eviction, reset, clone, and copy_state.
    // Invalid/duplicate candidate IDs retain the historical map behavior.
    for (int vocab : {0, 32, nv}) {
        auto * s = llama_sampler_init_penalties(vocab, 8, 1.1f, 0.2f, 1.1f);
        std::vector<int> history;
        for (int step = 0; step < 100; ++step) {
            if (step % 23 == 0) { llama_sampler_reset(s); history.clear(); }
            int token = int(rng() % 40) - 2;
            llama_sampler_accept(s, token);
            history.push_back(token);
            if (step % 17 == 0) { auto * copy = llama_sampler_clone(s); llama_sampler_free(s); s = copy; }
            if (step % 19 == 0) {
                auto * copy = llama_sampler_init_penalties(vocab, 8, 1.1f, 0.2f, 1.1f);
                llama_sampler_accept(copy, 37);
                std::vector<llama_token_data> old_data = {{37, 2.f, 0.f}};
                llama_token_data_array old_array{old_data.data(), old_data.size(), -1, false};
                llama_sampler_apply(copy, &old_array); // allocate and use destination scratch before copying
                llama_sampler_copy(s, copy);
                llama_sampler_free(s); s = copy;
            }
            std::unordered_map<int,int> counts;
            for (size_t j = history.size() > 8 ? history.size()-8 : 0; j < history.size(); ++j) ++counts[history[j]];
            std::vector<llama_token_data> data;
            for (int j = 0; j < 100; ++j) data.push_back({j % 40 - 2, float(j-50)/10.f, 0.f});
            std::shuffle(data.begin(), data.end(), rng);
            for (int pass = 0; pass < 2; ++pass) {
                auto expected = data;
                for (auto & d : expected) {
                    auto it = counts.find(d.id);
                    if (it == counts.end()) continue;
                    d.logit = d.logit <= 0 ? d.logit * 1.1f : d.logit / 1.1f;
                    d.logit -= float(it->second) * 0.2f + 1.1f;
                }
                llama_token_data_array a{data.data(), data.size(), -1, true};
                llama_sampler_apply(s, &a);
                if (a.sorted) return 3;
                for (size_t j = 0; j < data.size(); ++j) if (data[j].logit != expected[j].logit) {
                    std::fprintf(stderr, "FAIL reuse vocab=%d step=%d pass=%d index=%zu\n", vocab,step,pass,j); return 1;
                }
            }
        }
        llama_sampler_free(s);
    }
    std::puts("Correctness: exact float match for ordered/shuffled/subset arrays, eviction/reset/clone/copy/reuse, duplicate/out-of-range IDs");
    auto * s = llama_sampler_init_penalties(nv, 64, 1.f, 0.f, 1.1f);
    for (int i = 0; i < 64; ++i) llama_sampler_accept(s, (i*3917)%nv);
    std::vector<llama_token_data> data;
    for (int i = 0; i < nv; ++i) data.push_back({i, float(i % 100), 0.f});
    llama_token_data_array a{data.data(),data.size(),-1,false};
    std::vector<double> samples;
    for (int trial = 0; trial < 7; ++trial) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; ++i) llama_sampler_apply(s,&a);
        double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/200;
        samples.push_back(us);
    }
    std::sort(samples.begin(),samples.end());
    std::printf("penalties median_us=%.3f min_us=%.3f max_us=%.3f\n",samples[3],samples.front(),samples.back());
    llama_sampler_free(s);
    if (argc > 1 && samples[3] >= std::atof(argv[1])) { std::fprintf(stderr,"FAIL: latency exceeds requested ceiling %s us\n",argv[1]); return 2; }
}
