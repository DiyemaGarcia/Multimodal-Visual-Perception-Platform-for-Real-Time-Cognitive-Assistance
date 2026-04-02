#include <torch/extension.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

/*
 * morpheme_segmenter.cpp
 *
 * Fast C++ implementation of greedy morpheme segmentation.
 * Given a known morpheme vocabulary, segments a word into
 * the longest matching morphemes from left to right.
 *
 * This is used as a fast baseline segmentation for evaluation
 * and for building morpheme-level token sequences.
 */

using Vocab = std::unordered_map<std::string, int>;

/*
 * Greedy left-to-right segmentation using a morpheme vocabulary.
 * At each position, takes the longest morpheme that exists in the vocab.
 *
 * Args:
 *   word:       Input word string.
 *   morphemes:  Set of known morpheme strings.
 *   max_morph:  Maximum morpheme length to consider.
 *
 * Returns:
 *   Vector of morpheme strings.
 */
std::vector<std::string> greedy_segment(
    const std::string& word,
    const std::vector<std::string>& morphemes,
    int max_morph = 8
) {
    // Build a hash set for O(1) lookup
    std::unordered_map<std::string, bool> morph_set;
    for (const auto& m : morphemes) morph_set[m] = true;

    std::vector<std::string> result;
    int i = 0;
    int n = static_cast<int>(word.size());

    while (i < n) {
        int best_len = 1;
        // Try longest match first
        for (int l = std::min(max_morph, n - i); l >= 1; --l) {
            std::string candidate = word.substr(i, l);
            if (morph_set.count(candidate)) {
                best_len = l;
                break;
            }
        }
        result.push_back(word.substr(i, best_len));
        i += best_len;
    }

    return result;
}

/*
 * Batch segmentation over a list of words.
 */
std::vector<std::vector<std::string>> batch_segment(
    const std::vector<std::string>& words,
    const std::vector<std::string>& morphemes,
    int max_morph = 8
) {
    std::vector<std::vector<std::string>> results;
    results.reserve(words.size());
    for (const auto& word : words) {
        results.push_back(greedy_segment(word, morphemes, max_morph));
    }
    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "greedy_segment",
        &greedy_segment,
        "Greedy left-to-right morpheme segmentation of a single word.",
        py::arg("word"), py::arg("morphemes"), py::arg("max_morph") = 8
    );
    m.def(
        "batch_segment",
        &batch_segment,
        "Batch greedy segmentation for a list of words.",
        py::arg("words"), py::arg("morphemes"), py::arg("max_morph") = 8
    );
}