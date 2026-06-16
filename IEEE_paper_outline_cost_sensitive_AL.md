# Cost-Sensitive Active Learning for Reducing Missed Environmental Anomalies in UAV Multispectral Imagery

## Target IEEE-Style Paper Structure

### Abstract
This paper proposes a cost-sensitive active learning framework for UAV multispectral environmental anomaly detection. Unlike conventional active learning methods that primarily optimize global classification accuracy, the proposed framework explicitly penalizes missed anomalies through a configurable risk coefficient. The method combines a cost-sensitive support vector machine, risk-sensitive margin sampling, and kernel k-means diversity filtering to select informative and nonredundant superpixels under limited annotation budgets. Experiments on four UAV multispectral riparian scenes show that the proposed cost-sensitive active learning strategy substantially reduces false negatives and risk-weighted misclassification cost compared with random sampling, standard margin-based active learning, and uncertainty-diversity baselines. At the highest risk level, the aggregate false-negative rate decreases from 0.214 with standard active learning to 0.078 with the proposed method, increasing anomaly recall from 0.786 to 0.922. These results indicate that incorporating operational risk into both model training and query selection is effective for reducing missed environmental anomalies in annotation-constrained UAV monitoring.

### Index Terms
Active learning, cost-sensitive learning, UAV multispectral imagery, environmental anomaly detection, riparian monitoring, false negative reduction, support vector machine, kernel k-means.

## I. Introduction

### Motivation
UAV multispectral imagery enables high-resolution monitoring of riparian and riverine environments, but operational anomaly detection remains annotation-constrained. Expert labeling at the superpixel or pixel level is costly, and missing true environmental anomalies can be more consequential than producing additional false alarms.

### Problem Gap
Most active learning methods for remote sensing select samples to improve global metrics such as accuracy, ROC-AUC, or average precision. These metrics treat false positives and false negatives symmetrically or evaluate ranking quality without directly encoding operational risk. In environmental monitoring, however, false negatives correspond to missed pollution, vegetation stress, water-quality changes, or other high-risk anomalies.

### Proposed Viewpoint
The central argument is that active learning should be optimized not only for informativeness, but also for the asymmetric cost of missed anomalies. The paper introduces a risk coefficient `r+` that increases the penalty of false negatives and uses the same risk-aware decision function for both training and query selection.

### Contributions
1. A cost-sensitive active learning framework for UAV multispectral riparian anomaly detection.
2. A risk-weighted evaluation criterion, `C(r+) = (r+ FN + FP) / N * 100`, tailored to missed-anomaly reduction.
3. A two-stage query strategy combining risk-sensitive margin uncertainty and kernel k-means diversity filtering.
4. A comprehensive comparison with Random AL, Standard AL, and uncertainty-diversity baselines across four UAV scenes and four risk levels.
5. Confusion-matrix evidence showing that the proposed method substantially reduces missed anomalies under limited labeling budgets.

## II. Related Work

### A. Active Learning in Remote Sensing
Discuss pool-based active learning, uncertainty sampling, margin sampling, batch-mode selection, uncertainty-diversity selection, and remote sensing annotation constraints.

### B. Cost-Sensitive Learning
Introduce asymmetric misclassification costs, class-weighted SVMs, and why false-negative-sensitive objectives are important in operational monitoring.

### C. Multispectral Environmental Anomaly Detection
Discuss superpixel-level UAV multispectral anomaly detection and the practical cost of missed anomalies in environmental monitoring.

### D. Positioning of This Work
Emphasize that the novelty is not only a query heuristic, but the integration of risk-sensitive training, risk-sensitive querying, and risk-sensitive evaluation for missed environmental anomaly reduction.

## III. Study Area and Dataset

### A. UAV Multispectral Scenes
Use four scenes: `Z1`, `Z2`, `E1`, and `E2`. Each scene contains five multispectral bands:
- Blue, 475 nm
- Green, 560 nm
- Red, 668 nm
- RedEdge, 717 nm
- NIR, 840 nm

### B. Superpixel Representation
Each scene is segmented into superpixels. Labels are assigned by majority vote from ground-truth anomaly masks:
- `0`: normal
- `1`: anomaly

### C. Feature Extraction
Use 19-D superpixel features:
- 5 band means
- 5 band standard deviations
- NDVI, NDRE, ExG, EVI, BNDVI
- R/B, NIR/R, NIR/RedEdge ratios
- Mahalanobis-like spectral anomaly score

### Suggested Table I
Dataset summary: scene name, image size, number of superpixels, number and percentage of anomalies, train/test split.

## IV. Proposed Method

### A. Risk-Weighted Misclassification Cost
Define the operational cost:

`C(r+) = (r+ * FN + FP) / N * 100`

where `r+ >= 1` controls the penalty of missed anomalies. When `r+=1`, the cost reduces to symmetric misclassification cost; larger values increasingly prioritize anomaly recall.

### B. Cost-Sensitive SVM
Train an RBF SVM with class weights:

`w_anomaly = r+`, `w_normal = 1`

This shifts the decision boundary to reduce false negatives.

### C. Cost-Sensitive Active Learning
At each iteration:
1. Train cSVM on the labeled set.
2. Compute margin uncertainty using the risk-sensitive decision function.
3. Select the top uncertain candidate pool.
4. Apply kernel k-means in the RBF-induced feature space.
5. Query the most uncertain sample from each cluster.

### D. Baselines
Compare against:
- Random AL
- Standard AL
- Unc+KernelKMeans
- cAL with `r+ = 1, 2, 3, 4`

### Suggested Figure 1
Method diagram: UAV multispectral image -> superpixels -> spectral features -> cSVM -> risk-sensitive uncertainty -> kernel k-means diversity -> queried annotations -> updated classifier.

## V. Experimental Design

### A. Active Learning Protocol
- Initial labeled set: 20 superpixels, stratified as 10 anomaly and 10 normal.
- Query batch size: 50.
- Maximum budget: 600 queried samples or 5% of the pool.
- Repeated runs: 3 for cAL experiments.
- Split: 70% pool and 30% test within each scene, class-stratified.

### B. Evaluation Metrics
Primary:
- Risk-weighted cost `C(r+)`
- False negative rate
- Confusion matrix

Secondary:
- ROC-AUC
- Precision
- Recall
- False positive rate

### C. Statistical Recommendation
For the final paper, add AULC-based Wilcoxon signed-rank or bootstrap confidence intervals comparing:
- cAL `r+=4` vs Standard AL
- cAL `r+=4` vs Unc+KernelKMeans
- cAL `r+=3` vs Standard AL

## VI. Results

### A. Risk-Weighted Cost Curves
Use `results/al_cal/cal_cost_aggregated.png`.

**Figure 1. Aggregated risk-weighted cost curves across risk levels.**
The cAL advantage becomes stronger as `r+` increases. At `r+=3` and `r+=4`, cAL consistently lies below Random, Standard AL, and Unc+KernelKMeans over much of the active learning budget.

**Interpretation.**
This confirms the core hypothesis: when false negatives are operationally expensive, risk-sensitive training and querying reduce the relevant cost more effectively than accuracy-oriented active learning.

### B. Per-Scene Risk Cost Comparison
Use `results/al_cal/cal_vs_baselines.png`.

**Figure 2. Scene-wise cAL versus baselines for each risk level.**
This 4x4 figure shows that the benefit is scene-dependent. Easier scenes such as E1 converge near the full-pool reference, whereas challenging scenes such as Z2 and E2 show larger separation between cAL and baseline methods.

**Interpretation.**
The method is most valuable under difficult scene conditions, where conventional active learning continues to miss anomalies.

### C. False Negative Rate Curves
Use `results/al_cal/cal_fnr_curves.png`.

**Figure 3. False negative rate during active learning.**
This is the strongest figure for the paper. Across all four scenes, `cAL r+=4` produces the lowest or near-lowest FNR throughout the later active learning budget. The reduction is particularly clear in Z2 and E2.

**Interpretation.**
The proposed method directly addresses the missed-anomaly problem. It does not merely improve ranking metrics; it changes the error profile by reducing false negatives.

### D. Final-Budget Confusion Matrix Analysis
Use:
- `results/al_cal/confusion_aggregate_standard_al.png`
- `results/al_cal/confusion_aggregate_cal_rp4.png`

### Suggested Table II. Aggregate Confusion Matrix and Error Profile

| Method | TN | FP | FN | TP | FNR | FPR | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 47594.3 | 263.7 | 587.7 | 1720.3 | 0.255 | 0.006 | 0.745 | 0.867 |
| Standard AL | 47645.7 | 212.3 | 493.0 | 1815.0 | 0.214 | 0.004 | 0.786 | 0.895 |
| Unc+KernelKMeans | 47629.7 | 228.3 | 527.3 | 1780.7 | 0.228 | 0.005 | 0.772 | 0.886 |
| cAL r+=4 | 47096.0 | 762.0 | 179.3 | 2128.7 | 0.078 | 0.016 | 0.922 | 0.736 |

**Interpretation.**
The proposed method reduces aggregate false negatives from 493.0 under Standard AL to 179.3 under `cAL r+=4`, corresponding to a reduction of approximately 63.6%. Anomaly recall increases from 0.786 to 0.922. This improvement comes with a controlled increase in false positives, from 212.3 to 762.0. For environmental monitoring, this is a favorable trade-off when missed anomalies are more costly than additional inspection alarms.

### Suggested Table III. Scene-Specific Standard AL vs cAL r+=4

| Scene | Method | FN | TP | FNR | Recall | FP | Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Z1 | Standard AL | 147.7 | 821.3 | 0.152 | 0.848 | 113.7 | 0.878 |
| Z1 | cAL r+=4 | 81.0 | 888.0 | 0.084 | 0.916 | 292.7 | 0.752 |
| Z2 | Standard AL | 166.7 | 334.3 | 0.333 | 0.667 | 49.7 | 0.871 |
| Z2 | cAL r+=4 | 33.7 | 467.3 | 0.067 | 0.933 | 312.7 | 0.599 |
| E1 | Standard AL | 41.3 | 456.7 | 0.083 | 0.917 | 35.0 | 0.929 |
| E1 | cAL r+=4 | 16.3 | 481.7 | 0.033 | 0.967 | 72.3 | 0.869 |
| E2 | Standard AL | 137.3 | 202.7 | 0.404 | 0.596 | 14.0 | 0.935 |
| E2 | cAL r+=4 | 48.3 | 291.7 | 0.142 | 0.858 | 84.3 | 0.776 |

**Interpretation.**
The largest gains occur in the most difficult scenes. In Z2, FNR drops from 0.333 to 0.067. In E2, FNR drops from 0.404 to 0.142. These are the strongest operational results because they show that cAL prevents many missed anomalies precisely where baseline active learning is least reliable.

## VII. Discussion

### A. Why cAL Reduces Missed Anomalies
The cost-sensitive SVM shifts the decision boundary to penalize missed anomalies. The query strategy then samples uncertain points near this risk-adjusted boundary, while kernel k-means avoids redundant selections. This combination explains why cAL improves recall without relying only on random oversampling or threshold tuning.

### B. Risk Trade-Off
The method intentionally increases false positives. This is not a failure; it is the expected trade-off when false negatives are assigned higher cost. In environmental monitoring, additional candidate alarms may be acceptable if they prevent missing true anomalies.

### C. Why Symmetric Evaluation Is Insufficient
Conventional active learning can reduce overall error while still leaving a large number of false negatives. The confusion matrices show that missed-anomaly reduction requires explicit risk-sensitive training and evaluation rather than symmetric metrics alone.

### D. Limitations
1. The dataset contains only four UAV scenes.
2. The current cAL runs use three repetitions; more repetitions would strengthen statistical reliability.
3. Full-pool cost is not always reached under the limited budget, so label-efficiency claims should be phrased carefully.

### E. Practical Implications
The method is suitable for annotation-constrained UAV monitoring workflows where analysts prefer to inspect more suspicious regions rather than miss high-risk anomalies.

## VIII. Conclusion

This paper presents a cost-sensitive active learning framework for reducing missed environmental anomalies in UAV multispectral imagery. The method integrates risk-weighted SVM training, risk-sensitive uncertainty sampling, and kernel k-means diversity filtering. Results show that the proposed approach substantially reduces false negatives and risk-weighted cost under limited labeling budgets, particularly in challenging scenes. The aggregate confusion matrix demonstrates a reduction in false negatives from 493.0 with Standard AL to 179.3 with `cAL r+=4`, increasing anomaly recall from 0.786 to 0.922. These findings support risk-sensitive active learning as a practical strategy for environmental monitoring scenarios in which missed anomalies are more costly than additional false alarms.

## Recommended Figure Order

1. Proposed cAL workflow diagram.
2. Aggregated risk-weighted cost curves: `cal_cost_aggregated.png`.
3. False negative rate curves: `cal_fnr_curves.png`.
4. Scene-wise risk-cost curves: `cal_vs_baselines.png`.
5. Aggregate confusion matrices: `confusion_aggregate_standard_al.png` and `confusion_aggregate_cal_rp4.png`.

## Recommended Main Tables

1. Dataset and scene summary.
2. Active learning configuration.
3. Final risk-weighted cost and FNR by method.
4. Aggregate confusion matrix table.
5. Scene-specific Standard AL vs cAL `r+=4`.

## Paper Figure Set

This section includes only figures directly tied to the manuscript title and main claim.

### Fig. 1. Aggregated Risk-Weighted Cost Curves

This is a core figure. It shows that the proposed cAL strategy becomes more beneficial as the missed-anomaly risk coefficient increases. The `r+=3` and `r+=4` panels are the strongest evidence for the paper's cost-sensitive argument.

![Aggregated cAL cost curves](results/al_cal/cal_cost_aggregated.png)

### Fig. 2. False Negative Rate During Active Learning

This is the strongest figure for the manuscript. It directly supports the title by showing that `cAL r+=4` consistently reduces missed environmental anomalies, especially in the difficult Z2 and E2 scenes.

![cAL false negative rate curves](results/al_cal/cal_fnr_curves.png)

### Fig. 3. Scene-Wise Risk-Cost Curves

Use this as the detailed evidence figure. It shows that the gain is not uniform across scenes: easy scenes converge quickly, while challenging scenes benefit most from risk-sensitive active learning.

![cAL versus baselines](results/al_cal/cal_vs_baselines.png)

### Fig. 4. Aggregate Confusion Matrices: Standard AL vs cAL r+=4

These figures support the confusion-matrix argument. Standard AL has low false positives, but it misses substantially more anomalies than cAL. The key message is the reduction in false negatives from 493.0 to 179.3 and the increase in recall from 0.786 to 0.922.

![Aggregate confusion matrix Standard AL](results/al_cal/confusion_aggregate_standard_al.png)

![Aggregate confusion matrix cAL r+=4](results/al_cal/confusion_aggregate_cal_rp4.png)
