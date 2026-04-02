#include <torch/extension.h>
#include <vector>
#include <string>
#include <algorithm>

/*
 * fast_edit_distance.cpp
 *
 * Computes Levenshtein edit distance between pairs of strings.
 * Exposed to Python via PyTorch's C++ extension API.
 * Used for morpheme similarity scoring during evaluation.
 */

int levenshtein(const std::string& s1, const std::string& s2) {
    const int m = static_cast<int>(s1.size());
    const int n = static_cast<int>(s2.size());

    // dp[i][j] = edit distance between s1[:i] and s2[:j]
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({
                    dp[i - 1][j],      // deletion
                    dp[i][j - 1],      // insertion
                    dp[i - 1][j - 1]   // substitution
                });
            }
        }
    }
    return dp[m][n];
}

/*
 * Batch edit distance: computes distances for a list of (s1, s2) pairs.
 * Returns a vector of integer distances.
 */
std::vector<int> batch_edit_distance(
    const std::vector<std::string>& sources,
    const std::vector<std::string>& targets
) {
    if (sources.size() != targets.size()) {
        throw std::invalid_argument("sources and targets must have equal length.");
    }
    std::vector<int> results(sources.size());
    for (size_t i = 0; i < sources.size(); ++i) {
        results[i] = levenshtein(sources[i], targets[i]);
    }
    return results;
}

/*
 * Normalised edit distance: levenshtein / max(len(s1), len(s2)).
 * Returns a float in [0.0, 1.0].
 */
float normalised_edit_distance(const std::string& s1, const std::string& s2) {
    if (s1.empty() && s2.empty()) return 0.0f;
    int dist = levenshtein(s1, s2);
    int max_len = static_cast<int>(std::max(s1.size(), s2.size()));
    return static_cast<float>(dist) / static_cast<float>(max_len);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "levenshtein",
        &levenshtein,
        "Compute Levenshtein edit distance between two strings.",
        py::arg("s1"), py::arg("s2")
    );
    m.def(
        "batch_edit_distance",
        &batch_edit_distance,
        "Compute Levenshtein distances for a batch of string pairs.",
        py::arg("sources"), py::arg("targets")
    );
    m.def(
        "normalised_edit_distance",
        &normalised_edit_distance,
        "Compute normalised edit distance in [0, 1].",
        py::arg("s1"), py::arg("s2")
    );
}