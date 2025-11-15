### **Assignment 1: Problem and Data Understanding Report**

**Team Member:** Muhammad Musaab ul Haq

**Objective:** To conduct an in-depth analysis of the SemEval 2026 Task 4 development dataset to understand its core challenges, define the problem space, and inform the strategy for subsequent assignments.

**Methodology:** A simple TF-IDF + Cosine Similarity baseline was implemented not as a viable solution, but as a diagnostic probe. By analyzing the specific cases where this lexical model fails, we can reverse-engineer the dataset's underlying complexities. The baseline's near-random accuracy of **52.5%** immediately confirmed that the dataset is non-trivial and robust against superficial methods.

---

### **Key Characteristics and Challenges of the Dataset**

The analysis of 95 failure cases reveals that the dataset is deliberately constructed to reward a deep, human-like understanding of narrative while penalizing simple lexical matching. The key challenges embedded within the data are as follows:

#### **Challenge 1: The Presence of High-Signal Lexical Traps**

A primary characteristic of this dataset is the intentional inclusion of narratively dissimilar stories that share a high degree of superficial keyword overlap. A purely lexical model is consistently drawn to these false signals.

*   **Evidence (Proper Nouns):** In **Case 75**, the dataset contains two stories where the protagonists are both named **"Paul."** This single, identical keyword resulted in an overwhelmingly confident (and incorrect) prediction, demonstrating that the data will actively punish systems that over-weight superficial details.
*   **Evidence (Setting & Genre):** In **Case 190**, two stories were incorrectly matched because one began on a **"beach"** and the other involved being lost at **"sea."** The dataset frequently uses shared settings (e.g., "Paris" in **Case 143**), professions (e.g., "writer" in **Case 164**), or genre conventions (e.g., "crime thriller" in **Case 172**) to create these traps.

**Implication:** Any successful system must be able to look past simple word co-occurrence and prioritize deeper structural and thematic connections.

#### **Challenge 2: The Primacy of Abstract Thematic Cores**

The ground truth of the dataset is consistently determined by abstract themes that are not explicitly stated. The data requires a system to infer the core message or concept of a story.

*   **Evidence (Conceptual Gulf):** **Case 106** provides the most stark example. The dataset pairs two allegories about the failure of the Soviet system. Because they use entirely different vocabularies ("Communist worker" vs. "Soviet bureaucracy"), the lexical similarity was zero. This shows the dataset's similarity metric operates on a purely conceptual level, completely detached from surface-level words.
*   **Evidence (Unusual Themes):** In **Case 25**, the true connection was the bizarre, shared theme of characters believing they are animals. The dataset expects systems to identify this highly specific, abstract link over more generic vocabulary related to the story's setting.

**Implication:** A successful model must be capable of semantic reasoning. It needs to understand that "losing a job" and "a failed marriage" can both map to the concept of "loss." This points towards the necessity of high-quality embeddings or large language models.

#### **Challenge 3: Narrative Structure and Outcomes Define Similarity**

The dataset places significant weight on the *entire narrative arc*, especially the outcome. Stories that share an initial premise but have different conclusions are considered dissimilar.

*   **Evidence (Opposite Outcomes):** In **Case 18**, two stories shared a nearly identical setup: a class-based romance facing parental opposition. However, the dataset's ground truth ignores this, as one story ends in a tragic suicide and the other in a happy marriage. The **Outcome** is the deciding factor, rendering the initial lexical similarity irrelevant.
*   **Evidence (Course of Action):** In **Case 46**, the correct choice was based on a shared plot structure (Affair -> Murder -> Protagonist faces consequences). The dataset rewards systems that can identify this sequential, cause-and-effect pattern over stories that merely share a cloud of related keywords.

**Implication:** Our system cannot treat stories as unordered "bags of words." It must have a mechanism to understand sequence, causality, and the profound narrative impact of a story's conclusion.

### **Conclusion and Path Forward**

This analysis concludes that the SemEval 2026 Task 4 dataset is a robust challenge designed to test for true narrative understanding. The problem is not one of simple text classification, but of abstract reasoning about thematic content and plot structure.

This data understanding phase has been critical. We now know not only that the problem is hard, but *why* it is hard, and have a much clearer path forward.

All cases analysis can be found in `reports/A1_Musaab_error_analysis_of_simple_models.md`.