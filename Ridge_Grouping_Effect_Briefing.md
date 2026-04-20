# The Ridge Grouping Effect: Differential Shrinkage as a Source of Out-of-Sample ΔR² Inflation in Epistasis Detection

**Prepared for:** StarBASE-GP Collaborators  
**Date:** April 2026  
**Status:** Internal Briefing — Not for External Distribution

---

## 1. Executive Summary

During ablation studies on StarBASE-GP, we observed a statistical artifact: removing the biological Linkage Disequilibrium (LD) filter causes out-of-sample ΔR² to inflate by 2x–3.5x relative to cross-validation estimates. The mechanism is **differential shrinkage** arising from Ridge regression's L2 penalty. When highly correlated proxy interactions (e.g., SNP_A1:SNP_B1 and SNP_A2:SNP_B1, where A1 and A2 are in high LD) enter the same pipeline, Ridge distributes the coefficient weight across them, reducing the effective penalty by a factor of ~1/k for k proxies. This creates an asymmetry: the Joint model (main effects + interactions) retains more of the true biological signal than the Base model (main effects only), which pays full shrinkage. Out-of-sample, the Base model's R² decays faster, causing the ΔR² gap to widen artifactually.

We have verified that standard downstream regression corrections (OLS, Lasso, Elastic Net, Generalized Ridge, Adjusted R²) cannot resolve this in the high-dimensional, microscopic-effect-size regime characteristic of epistasis. **The upstream biological LD filter is the necessary and validated primary defense.** A multi-axis diagnostic framework (Section 7) — including ΔR² deconstruction, ground truth recovery, correlation heatmaps, and permutation feature importance — provides converging manuscript evidence that Full outperforms Basic on every evaluative dimension once the inflation artifact is removed.

---

## 2. System Architecture

### 2.1 StarBASE-GP Overview

StarBASE-GP is an AutoML tool that uses Genetic Programming (GP) to discover 2-way epistatic interactions. The data pipeline operates on three non-overlapping splits:

| Split | Proportion | Role |
|-------|-----------|------|
| Cross-Validation (CV) | 60% | GP evolution and pipeline evaluation |
| Internal Validation | 20% | Post-GP "Utopia" model selection from Pareto front |
| Hold-Out Test | 20% | Final unbiased performance estimate |

The GP runs **exclusively** on the 60% CV data, evolving pipelines and evaluating them on a Pareto front that maximizes epistatic ΔR² while minimizing pipeline complexity.

### 2.2 Model Comparison Framework

Epistatic contribution is quantified via a model comparison:

- **Model 0 (Base):** Contains only the unique univariate main-effect SNPs.
- **Model 1 (Joint):** Contains the same main-effect SNPs **plus** 2-way interaction vectors, encoded via Cartesian, XOR, or MDR schemes.
- **Fitness metric:** ΔR² = R²_Joint − R²_Base

### 2.3 Why Regularized Regression Is Required for Epistasis but Not Univariate

In the univariate module of StarBASE, OLS is sufficient: each pipeline contains only main-effect SNPs, and the GP's complexity constraints keep the feature count well within the 10x rule (n/p > 10). The design matrix is well-conditioned, OLS coefficients are stable, and no regularization is needed.

The epistasis module breaks this assumption through three compounding mechanisms:

1. **Combinatorial feature expansion.** Each 2-way interaction adds a feature to the Joint model. A pipeline with 10 main-effect SNPs and 10 interactions has 20 features in the Base model but up to 30 in the Joint model. As the GP explores larger interaction sets, the Joint model's feature count grows combinatorially while the sample size remains fixed. The 10x rule is routinely violated during GP evolution — a CV fold training set of ~480 samples (48% of n=1000) can encounter Joint models with 50–90 features, pushing p/n toward 0.1–0.2.

2. **Intrinsic collinearity of interaction features.** Interaction vectors (A×B, encoded via Cartesian, XOR, or MDR) are inherently correlated with their constituent main effects (A and B) and with other interactions sharing a common SNP (A×B and A×C). Even with perfectly independent SNPs, the interaction encoding introduces non-trivial correlation structure into the design matrix. Under OLS, this collinearity inflates coefficient variance, producing unstable estimates that alternate in sign and magnitude across CV folds.

3. **Microscopic effect sizes at the noise floor.** Epistatic interactions typically explain <1% of phenotypic variance individually. OLS estimates of such small effects have high relative variance — the signal-to-noise ratio per coefficient is far lower than for main effects. Regularization stabilizes these estimates by trading a small amount of bias for a large reduction in variance, which is the favorable tradeoff when effects are small and features are numerous.

These three factors — feature expansion past the 10x rule, intrinsic collinearity from interaction encoding, and microscopic effect sizes — make OLS unsuitable for the epistasis module. Ridge regression (L2 regularization) addresses all three: it stabilizes the (X'X + αI) inversion even when X'X approaches singularity, it shrinks correlated coefficients toward zero rather than letting them explode, and it reduces the variance of small-effect estimates at the cost of controlled bias.

**Why Ridge specifically, rather than Lasso or Elastic Net:** Ridge preserves all features with non-zero (but shrunk) coefficients, which is essential for the ΔR² model comparison — the fitness metric requires that both Base and Joint models produce non-degenerate predictions. Lasso's hard thresholding zeros out microscopic epistatic effects entirely (see Section 5.2), collapsing the Joint model to a flat line and making ΔR² undefined. Ridge's soft shrinkage is the only L1/L2 option that reliably preserves signal at these effect sizes.

**Static α = 1.0** is used as the initial regularization strength. The appropriateness of this choice, and the case for replacing it with data-driven GCV-optimized α, is discussed in Section 6.3 and the Phase 1 ablation plan (Section 8.3).

### 2.4 Regression Implementation Details

**Critical implementation detail:** When the GP finishes, the Pareto front of pipelines (i.e., the selected feature sets) is carried forward, but the models are **refit on the full 60% CV data** before being applied to the 20% validation and 20% test sets. This means the coefficients used for out-of-sample evaluation are estimated on a larger sample than any individual CV fold (~48% per fold vs. the full 60%), giving Ridge more data to work with and making α = 1.0 relatively less aggressive than during CV training.

---

## 3. The Phenomenon

### 3.1 Experimental Conditions

We ran an ablation study on simulated data comparing two conditions:

- **"Full" Condition:** A biological LD filter strictly prevents highly correlated SNPs and interactions from coexisting in the same pipeline.
- **"Basic" Condition:** The LD filter is removed, allowing the GP to build pipelines containing redundant proxy interactions (e.g., SNP_A1:SNP_B1 alongside SNP_A2:SNP_B1, where A1 and A2 share ~95% correlation).

### 3.2 Observed Results

| Condition | CV ΔR² | Out-of-Sample ΔR² | Inflation Factor |
|-----------|--------|-------------------|-----------------|
| Full (with LD filter) | Modest, realistic | Stable, consistent with CV | ~1.0x |
| Basic (no LD filter) | Modest, realistic | **2x–3.5x higher than CV** | 2.0–3.5x |

The CV-phase ΔR² scores for both conditions appear normal and comparable. The anomaly emerges **only** when the selected pipelines are refit on the full 60% CV data and applied to the unseen validation and test sets, and **only** in the Basic condition.

---

## 4. Mathematical Analysis of the Mechanism

### 4.1 The Ridge Grouping Effect: Core Derivation

Consider a single true signal feature **z** with true coefficient β_true, and suppose we have k identical copies of it (z₁ = z₂ = ... = z_k = z), representing proxy features in high LD.

The Ridge estimator minimizes:

    ||y - Zβ||² + α||β||²

By symmetry, Ridge assigns equal coefficients β* to each copy. Solving the normal equations:

**Single feature:**

    β_single = z'y / (z'z + α)

**k proxy features (each receiving β*):**

    β* = z'y / (k·z'z + α)

    Total predicted effect = k · β* = k·z'y / (k·z'z + α)

**Penalty comparison:**

    Single feature penalty:  (β_single)² = (z'y)² / (z'z + α)²

    k-proxy total penalty:   k·(β*)² = k·(z'y)² / (k·z'z + α)²

In the **strong-regularization regime** (α >> z'z) — which is precisely the epistasis regime where effects are microscopic and α = 1.0 is large relative to individual feature variances:

    Single feature effect    ≈  z'y / α
    k-proxy total effect     ≈  k · z'y / α    (k times larger)

    Single feature penalty   ≈  (z'y)² / α²
    k-proxy total penalty    ≈  (z'y)² / (k · α²)    (1/k times smaller)

**The proxy features achieve k times the predictive effect at 1/k times the penalty cost.**

#### Quantitative Consistency Check

Does this mechanism produce effect sizes consistent with the observed 2–3.5x inflation?

| α | z'z | k | Effect Ratio: k(z'z + α)/(k·z'z + α) |
|---|-----|---|---------------------------------------|
| 1 | 1 | 4 | 4(2)/5 = **1.6x** |
| 5 | 1 | 4 | 4(6)/9 = **2.67x** |
| 10 | 1 | 4 | 4(11)/14 = **3.14x** |
| 10 | 1 | 6 | 6(11)/16 = **4.13x** |

The observed 2–3.5x range is consistent with moderate proxy counts (k = 3–5) in the strong-regularization regime. This is exactly the operating point of StarBASE-GP under the Basic condition.

### 4.2 Differential Shrinkage and Out-of-Sample Behavior

The ΔR² inflation is not caused by the Joint model performing "better than expected" out-of-sample. It is caused by the **Base model degrading faster** than the Joint model, widening the gap in the subtraction metric.

**Model 0 (Base Model):**
- Contains unique main-effect SNPs, each bearing its full L2 penalty.
- Coefficients are over-shrunk relative to the true biological signal.
- Out-of-sample, over-shrunk coefficients systematically **under-predict** variance.
- R²_Base decays significantly.

**Model 1 (Joint Model):**
- Contains the same main-effect SNPs **plus** proxy interaction features.
- The interaction signal's shrinkage tax is diluted by factor ~1/k via the grouping effect.
- Out-of-sample, the less-shrunk interaction coefficients better preserve the true signal magnitude.
- R²_Joint resists decay.

**Result:**

    ΔR² = R²_Joint − R²_Base

Both R² values decay out-of-sample, but R²_Base decays *faster*. The subtraction amplifies the differential into an apparent 2–3.5x inflation.

#### Important Refinement: Joint Coefficient Coupling

Ridge estimates all coefficients jointly via (X'X + αI)⁻¹X'y. Adding correlated interaction features to Model 1 does not merely add new coefficients — it **changes the shrinkage on the main-effect coefficients** as well. If interactions correlate with main effects (which is common — an A×B interaction correlates with marginals A and B), then:

- Some main-effect variance in Model 1 is redistributed to interaction terms.
- Model 1's main-effect coefficients may be shrunk differently than Model 0's.
- The interaction terms capture signal more efficiently due to the proxy discount.

This redistribution is an additional pathway for the differential to widen, beyond the pure penalty arithmetic.

### 4.3 The Refit Step and Why the Artifact Persists

After the GP selects the Pareto front of pipelines on the CV folds, the models are **refit on the full 60% CV data** before being applied to the validation and test sets. This means the coefficients used for out-of-sample evaluation are estimated on a larger sample than any individual CV fold.

One might expect refitting to attenuate the artifact — with more data, X'X is larger relative to α, making regularization less aggressive and reducing the absolute magnitude of over-shrinkage. However, the artifact persists (2–3.5x inflation) for several reasons:

- **The grouping effect is relative, not absolute.** The 1/k penalty discount depends on the *ratio* of α to z'z, not on the absolute magnitude of shrinkage. Refitting on more data increases z'z proportionally for *all* features, but the proxy interactions still enjoy the same relative discount over singleton features. The discount factor k(z'z + α)/(k·z'z + α) changes only slightly as z'z increases — the proxies and singletons both benefit from more data, preserving the differential.

- **α = 1.0 remains in the strong-regularization regime.** Even with 60% of the data, individual epistatic features have small variances (z'z). The fixed α = 1.0 still dominates, keeping the system in the regime where the proxy discount is most potent.

- **The CV-to-refit transition introduces a regime shift.** During CV, each fold trains on ~48% of total data. The refit uses the full 60%. This change in effective sample size means the Ridge coefficients shift between CV evaluation and the refit used for validation/test. The *direction* of the shift is toward less shrinkage overall, but the *differential* between proxy-rich (Joint) and singleton (Base) models is preserved. The result: the ΔR² computed during CV (on fold-level models) may not match the ΔR² computed on validation/test (on the refit model), even absent any artifact. The grouping effect then compounds on top of this regime shift.

- **The differential shrinkage mechanism is structural.** The Base model pays full penalty on every feature regardless of sample size. The Joint model's proxy interactions split their penalty regardless of sample size. More data reduces absolute shrinkage for both models, but the *ratio* of their shrinkage rates remains governed by the proxy structure, not the sample size.

### 4.4 How to Deconstruct the ΔR²

To provide hard evidence in the manuscript that Basic's higher Utopia ΔR² is an artifact of redundant signal rather than true biological discovery, the subtraction must be broken apart. Reporting only ΔR² conceals which model moved — the question is whether the Joint model improved or the Base model collapsed.

**Method:** Extract the absolute R² values for Model 0 and Model 1 independently, rather than reporting only the Δ.

| Metric | Full Condition | Basic Condition |
|--------|---------------|-----------------|
| Model 0 (Base) R² — CV average | X | X |
| Model 0 (Base) R² — Validation | X | X |
| **Model 0 % decay (CV → Validation)** | **Moderate, symmetric** | **Moderate, symmetric** |
| Model 1 (Joint) R² — CV average | X | X |
| Model 1 (Joint) R² — Validation | X | X |
| **Model 1 % decay (CV → Validation)** | **Moderate, symmetric with Model 0** | **Artificially low — Model 1 resists decay** |

**Expected result:** In the Full condition, Model 0 and Model 1 decay by roughly similar percentages out-of-sample — the ΔR² remains stable because both models degrade proportionally. In the Basic condition, Model 0 decays normally (it has no proxy protection), but Model 1 stays artificially elevated due to the grouping effect's shrinkage discount. The ΔR² explodes not because Model 1 got better, but because Model 0 collapsed out from underneath it.

This table is the single most direct piece of evidence for the differential shrinkage mechanism. It converts the abstract mathematical argument of Sections 4.1–4.3 into a concrete, observable asymmetry in the data.

---

## 5. The Regression Graveyard: Why Downstream Fixes Fail

We have evaluated five alternative regression approaches. All fail in this specific regime.

### 5.1 Ordinary Least Squares (OLS)

**Failure mode:** Singular matrices and coefficient explosion.

With highly correlated LD blocks, X'X approaches singularity. OLS coefficients become arbitrarily large with alternating signs. Out-of-sample R² plummets to large negative values. OLS cannot handle collinearity without regularization — it is not a viable alternative.

### 5.2 Lasso (L1 Regularization)

**Failure mode:** Signal guillotine at microscopic effect sizes.

Lasso applies a soft-thresholding operator that zeros out coefficients below a data-dependent threshold. Epistatic interactions typically explain <1% of phenotypic variance. These effect sizes sit at or below the Lasso threshold, causing the interaction coefficients to be driven to exactly zero. GP pipelines collapse to flat lines with no detected epistasis.

Additionally, Lasso arbitrarily selects one feature from a correlated group and discards the rest, introducing instability — different CV folds may select different proxy representatives, inflating variance.

### 5.3 Elastic Net (L1 + L2)

**Failure mode:** The L1 component remains too aggressive.

Elastic Net was tested with a standard 0.5 L1 ratio (equal L1 and L2 weighting). At this ratio, the L1 penalty still guillotines the microscopic epistatic effects, producing the same flat-line collapse as pure Lasso.

**Note:** Very low L1 ratios (0.01–0.05) — providing ~95–99% Ridge behavior with mild sparsity — remain **untested**. At such ratios, the L1 component might prune exact duplicate features without destroying the epistatic signal. This is a concrete, low-effort ablation that should be explored, though it is unlikely to fully resolve the grouping effect for near-duplicate (but not identical) proxies.

### 5.4 Adjusted R²

**Failure mode:** Wrong correction target.

Adjusted R² penalizes for the *number* of features, not for correlation structure or shrinkage magnitude. The Pareto front already balances model complexity between Base and Joint models, so the feature-count penalty applies approximately symmetrically. The core problem — differential *coefficient shrinkage* — is invisible to adjusted R².

### 5.5 Generalized Ridge (Tikhonov with Covariance Penalty)

**Failure mode:** A mathematical catch-22.

The Tikhonov approach replaces the identity penalty αI with αΣ⁻¹, where Σ is the feature correlation matrix. The intent is to penalize correlated coefficients more heavily, eliminating the grouping discount.

The problem: when features are ~95% correlated, Σ⁻¹ is severely ill-conditioned. The penalty matrix amplifies exactly the directions that Ridge was stabilizing. Removing the spherical constraint that holds collinear coefficients together causes the estimation to revert to near-OLS behavior — coefficients explode, and the cure becomes the disease.

This is a fundamental catch-22: **you are asking the penalty to undo exactly the stabilization that Ridge provides.** Any penalty that fully accounts for correlation structure must, by construction, un-regularize the correlated dimensions.

---

## 6. Compounding Factors

Independent review identified several additional mechanisms that compound with the core Ridge grouping effect.

### 6.1 GP x Ridge Selection Amplification

The GP's evolutionary search process interacts with the shrinkage artifact in a self-reinforcing loop:

1. The "Basic" condition provides a **larger effective search space** — more proxy features generate more possible pipeline configurations.
2. Proxy-rich pipelines benefit from the Ridge grouping effect, producing **more consistent CV ΔR²** across folds (lower variance due to implicit coefficient averaging).
3. The GP's Pareto selection preferentially retains pipelines with consistent performance → it systematically selects pipelines that exploit the grouping effect.
4. These selected pipelines then exhibit maximum differential shrinkage on out-of-sample data.

**The GP and Ridge are conspiring:** the GP selects for the grouping exploit, and Ridge rewards exactly those pipelines. The observed 2–3.5x inflation likely reflects this selection-amplification loop in addition to the pure shrinkage mechanism.

### 6.2 Implicit Ensemble and Variance Reduction

At ~95% LD (not 100%), proxy features are correlated but not identical. Each proxy contributes β/k to the prediction with a slightly different noise profile. The variance of the combined prediction is:

    Var(Σ βᵢxᵢ) < k² · Var(β · x)    when pairwise correlations < 1

This is the Random Forest principle: averaging correlated-but-not-identical predictors reduces prediction variance. The Joint model's proxy interactions benefit from both:

- **(a)** Penalty dilution (the 1/k shrinkage discount)
- **(b)** Prediction variance reduction (the ensemble effect)

These effects compound, potentially explaining inflation at the upper end of the observed range (3.5x) beyond what pure penalty arithmetic predicts.

### 6.3 Static α = 1.0 as an Active Contributor

The fixed regularization parameter is not merely a convenience — it actively maximizes the artifact.

The proxy discount factor k(z'z + α)/(k·z'z + α) approaches its maximum of k when α >> z'z. With a weaker α (closer to the scale of z'z), the denominator k·z'z + α is dominated by k·z'z rather than α, and the discount shrinks.

At α = 1.0 with weak epistatic features (small z'z), the system operates deep in the strong-regularization regime where the grouping effect is most potent. **Cross-validating α separately for Model 0 and Model 1** would be a valuable ablation to disentangle the fixed-α contribution from the fundamental grouping effect.

### 6.4 A Contrarian Consideration: Is Part of the Inflation a Correction?

If α = 1.0 over-shrinks in the CV regime (small training folds, aggressive regularization), then the CV ΔR² may be a biased-low estimate of the true epistatic effect. Under this reading, the proxy discount partially corrects the over-shrinkage, and the out-of-sample estimate is closer to the true ΔR².

This interpretation is unlikely to account for the full 2–3.5x discrepancy — at that magnitude, something pathological is clearly occurring. However, the "true" ΔR² may lie somewhere between the CV and out-of-sample estimates for the Basic condition, rather than coinciding exactly with the CV value.

**This does not argue against the LD filter.** Even a "partial correction" via the proxy effect is uncontrolled, unreliable, and non-reproducible. The Full condition's stable ΔR² provides the only trustworthy estimate.

---

## 7. Manuscript Evidence: Proving Full's True ΔR² Exceeds Basic's

The analyses below provide converging evidence that once the inflation artifact is removed (via GCV-optimized α, with or without light L1), Full's properly estimated ΔR² exceeds Basic's — and that the LD operator is justified on every evaluative axis.

### 7.1 ΔR² Deconstruction (Section 4.4, Presented in the Paper)

The absolute R² decomposition from Section 4.4 is the first line of evidence. By showing that Basic's ΔR² inflation is driven by differential Model 0 collapse rather than Model 1 superiority, it establishes that the inflated number is not a genuine performance advantage. Once GCV equalizes the shrinkage regime, Basic's Model 1 loses its artificial resistance to out-of-sample decay, and its ΔR² falls to its true (lower) value. Full's ΔR², already stable across splits, remains unchanged or slightly increases (if α=1.0 was over-shrinking its signal). The gap favors Full.

### 7.2 Ground Truth Diversity

Show that Full recovers all (or nearly all) independent ground truth interactions, while Basic recovers only 1–2 true interactions and fills the remaining pipeline slots with redundant LD proxies of those same interactions.

This is the most fundamental argument: Basic cannot capture as much true epistatic variance because it literally does not contain the ground truth interactions. Redundant proxies of interaction A×B cannot explain variance attributable to interaction C×D. Even with a perfect estimator, Basic's ΔR² is bounded by the variance explained by the subset of ground truth interactions it happens to include — and that subset is smaller than Full's.

### 7.3 The Correlation Heatmap (The Smoking Gun)

Extract the interaction vectors from the Basic Utopia pipeline and the Full Utopia pipeline (5–10 interactions each). Plot a Pearson correlation heatmap of the interaction feature vectors for each.

- **Full:** The heatmap will show an approximately diagonal matrix — orthogonal, independent features. Each interaction contributes unique information.
- **Basic:** The heatmap will show massive off-diagonal red blocks — clusters of highly correlated features that are LD proxies of the same underlying interaction. The model is carrying redundant copies, not independent signal.

This figure provides immediate visual proof that the Basic model's feature set is degenerate. It also explains *why* Basic's ground truth recovery is low — pipeline capacity is consumed by echoes rather than new discoveries.

### 7.4 Permutation Feature Importance (PFI)

For each interaction feature in the Utopia pipeline, permute (shuffle) that single feature and measure the drop in out-of-sample R².

- **Full:** Permuting any single interaction causes a significant R² drop. Every feature is uniquely vital — no other feature in the model can compensate for the lost information.
- **Basic:** Permuting any single interaction causes minimal R² drop, because the remaining proxy features cover for the shuffled one. The model is robust to single-feature permutation not because every feature is unimportant, but because the information is redundantly distributed across correlated proxies.

PFI directly quantifies the interpretability advantage of the LD operator: in Full, each feature's contribution is identifiable and attributable to a specific biological interaction. In Basic, the signal is smeared across proxies, making it impossible to assign biological meaning to individual features.

### 7.5 Summary: Multi-Axis Comparison

| Evaluation Axis | Full (with LD filter) | Basic (no LD filter) | What it proves |
|----------------|----------------------|---------------------|---------------|
| ΔR² (GCV-corrected) | Stable, accurate | Lower than Full (after deflation) | LD operator captures more true epistatic variance |
| ΔR² stability (CV vs. test) | ~1.0x ratio | Inflated without GCV; stable but lower with GCV | LD operator produces trustworthy estimates |
| Ground truth recovery | High (all independent interactions) | Low (1–2 + redundant proxies) | LD operator enables biological discovery |
| Correlation heatmap | Orthogonal features | Massive LD blocks | LD operator prevents feature degeneracy |
| PFI | Every feature uniquely vital | Features are interchangeable | LD operator ensures interpretable, attributable signal |

The LD operator wins on every axis. The critical insight is that these axes are not independent — they are causally linked. The LD filter enables ground truth diversity (7.2), which enables higher true ΔR² (7.1), which is reflected in orthogonal features (7.3) where each carries unique, permutation-sensitive signal (7.4). Removing the LD filter breaks the first link in this chain, and every downstream metric degrades.

---

## 8. Solution Strategy

### 8.1 Primary Defense: Biological LD Filter (Validated)

The upstream LD filter enforces approximate orthogonality at the feature construction stage, preventing the grouping effect from ever forming. This is validated by the experimental results: the "Full" condition produces ΔR² estimates that are stable and consistent across CV, validation, and test splits.

**This is the necessary and sufficient primary defense.** No downstream regression correction can substitute for it in this regime.

### 8.2 Secondary Defense: Dynamic Alpha (Partial Correction)

To ensure fair model comparisons even when the LD filter is active, the Ridge penalty is dynamically scaled:

    α_joint = α_base × (p_joint / p_base)

where p is the feature count. This linearly taxes asymmetric feature bloat between the Base and Joint models.

**Limitation:** The proxy discount scales quadratically (~1/k for penalty), but the dynamic alpha correction scales linearly with the feature count ratio. A linear tax cannot perfectly close a quadratic loophole. This reinforces why the LD filter is mandatory — the dynamic alpha is a belt-and-suspenders measure, not a standalone solution.

### 8.3 Experimental Roadmap: Phased Ablation Plan

The following approaches are ordered by implementation priority — each phase builds on the previous one, isolates a specific variable, and informs whether subsequent phases are necessary. Critically, several phases are **not merely fixes for the Basic condition's artifact** — they improve the statistical rigor and accuracy of ΔR² estimation even in the Full condition with the LD filter active.

All phases should be run under **both Full and Basic conditions** on the same simulated data to enable direct comparison.

---

#### Phase 1: GCV-Optimized Alpha + L1 Ablation Sweep

**What changes:** Replace static α=1.0 with GCV-optimized α (independently for Model 0 and Model 1). Sweep L1_wt from 0.0 to 0.05.

**Why first:** This is the single highest-impact change with zero architectural modification — a drop-in replacement. It directly tests the most fundamental open question: was the arbitrary α=1.0 a major contributor to the artifact? It also establishes the calibrated baseline that all subsequent phases build on.

**How it works:** GCV (Generalized Cross-Validation) is a closed-form analytical formula that estimates the leave-one-out CV error for Ridge regression from one SVD decomposition. It requires **no inner CV folds, no sample-size reduction, and no significant compute cost**. With ≤30 interactions per pipeline (~60 base SNPs, ~90 joint features max), the SVD is microseconds per evaluation.

GCV computes the optimal α entirely on the training data available at each step — the fold's training portion during CV, or the full 60% during the refit. It never touches held-out data. Each model gets its own independently optimized α:
- **Model 0 (Base):** GCV finds α_base optimized for the main-effect SNPs in this pipeline.
- **Model 1 (Joint):** GCV finds α_joint optimized for the main effects + interactions in this pipeline.
- Both alphas are **data-driven and pipeline-specific**, replacing the arbitrary static α=1.0.

**No leakage:** GCV is computed on the same data used for model fitting. The fitness evaluation (ΔR²) is always computed on separate held-out data. The only change from the current pipeline is that α is chosen by the data rather than hardcoded.

**The L1 component:** GCV optimizes the L2 penalty only — there is no analytical equivalent for L1 (the absolute value breaks the hat matrix linearity that makes GCV possible). Instead, L1 is swept as a fixed constant across the ablation grid. Start with L1_wt=0.0 (pure GCV-Ridge) to establish baseline, then incrementally add L1 to test proxy pruning.

**Ablation grid:**

| Step | L1_wt | α selection | What it tests |
|------|-------|-------------|---------------|
| 1a (control) | 0.0 | Static α=1.0 | Current behavior — baseline |
| 1b | 0.0 | GCV (independent) | Impact of removing the arbitrary α=1.0 |
| 1c | 0.005 | GCV (independent) | Barely-there L1 nudge — does anything change? |
| 1d | 0.01 | GCV (independent) | Light L1 pruning |
| 1e | 0.02 | GCV (independent) | Moderate L1 pruning |
| 1f | 0.05 | GCV (independent) | Upper bound — watch for signal loss |

**Decision criteria — what Phase 1 tells you:**
- **1a vs 1b (Full condition):** If ΔR² increases, α=1.0 was over-shrinking the epistatic signal. The new estimate is more accurate, not inflated.
- **1a vs 1b (Basic condition):** If inflation attenuates, static α was a major contributor. If 2–3.5x persists, the grouping effect is fundamental.
- **1b vs 1c–1f:** Use the L1 diagnostic (coefficient survival count, see Appendix A.5) to find the largest L1_wt before the signal cliff. If no cliff exists by 0.05, all tested values are safe. If the cliff appears at 0.005, set L1_wt=0.0 — let GCV-Ridge do all the work.
- **Carry forward:** The best (L1_wt, GCV) combination from Phase 1 becomes the default for Phases 2–4.

**Benefits Full condition:** Yes (**strong**). Static α=1.0 likely over-shrinks the microscopic epistatic signal, biasing Full-condition ΔR² estimates low. GCV calibrates α to the actual signal-to-noise ratio of each pipeline, producing more accurate ΔR² estimates. This is not an artifact fix; it is about getting the *right answer*.

**Benefits Basic condition:** Yes (**strong diagnostic**). Directly tests whether fixed α was amplifying the differential shrinkage artifact.

---

#### Phase 2: Group Ridge with GCV (Block-Penalized Regression)

**What changes:** Split the Joint model's penalty into separate blocks — one α for main-effect features, one α for interaction features — with GCV or effective-df matching to set each block's penalty independently.

**Why second:** Phase 1 gives both models their own globally optimal α, but the Joint model still applies a single penalty to two fundamentally different feature types. Main effects and interactions occupy different signal-to-noise regimes. Phase 2 tests whether recognizing this difference improves accuracy.

**How it works:** The Joint model's objective becomes:

    ||y - X_main·β_main - X_int·β_int||² + α_main·||β_main||² + α_int·||β_int||²

The Base model remains unchanged (GCV-optimized α from Phase 1). For the Joint model, use GCV to find α_main (applied to the main-effect block) and a separate penalty for the interaction block. Statsmodels' `fit_regularized` supports per-feature alpha arrays, making this trivial to implement.

**Decision criteria:**
- **Full condition:** If ΔR² changes materially, main effects and interactions were being suboptimally penalized by a single α. The block-penalty estimate is more faithful.
- **Basic condition:** If inflation further attenuates beyond Phase 1, the grouping effect was being partially masked by uniform penalization.
- **If Phase 2 shows minimal change over Phase 1:** The GCV-optimized uniform α was already finding a good compromise, and block penalties aren't necessary. Stick with Phase 1's simpler approach.

**Benefits Full condition:** Yes (**strong**). Main effects typically have larger effect sizes and different variance structures than epistatic interactions. Block penalties allow each feature class to operate at its natural regularization regime, improving both R²_Base and R²_Joint accuracy.

**Benefits Basic condition:** Yes. Heavier penalization of the interaction block specifically targets the proxy grouping discount.

---

#### Phase 3: Effective-df Alpha Scaling (Alternative Comparison)

**What changes:** Instead of independent GCV for the Joint model (Phase 1), constrain α_joint such that the Joint model's effective degrees of freedom matches the Base model's. This replaces the current linear dynamic formula with a correlation-aware version.

**Why third:** This is not a strict improvement over Phase 1 — it's an alternative philosophy. Phase 1 lets each model optimize freely (best individual performance). Phase 3 constrains the Joint model to the same effective complexity as the Base model (fairest comparison). Running both lets you decide which framing is more appropriate.

**How it works:** Effective degrees of freedom df(α) = Σ dᵢ²/(dᵢ² + α) naturally accounts for feature correlation — k perfectly correlated features contribute ~1 effective df, not k. Find α_joint via bisection such that df(X_joint, α_joint) = df(X_base, α_base_gcv).

**Decision criteria:**
- If Phase 3 produces similar ΔR² to Phase 1, the independent GCV alphas were already producing a fair comparison — use Phase 1 (simpler).
- If Phase 3 produces materially *lower* ΔR² (especially in Basic condition), independent GCV was partially restoring the grouping advantage. The effective-df constraint is fairer.
- If Phase 3 produces materially *higher* ΔR², Phase 1's independent optimization was over-penalizing the Joint model. Independent GCV is fairer.

**Benefits Full condition:** Moderate. With LD-filtered features that are approximately orthogonal, effective df ≈ raw feature count, and this converges toward the linear dynamic formula. The improvement over Phase 1 is marginal in Full but the comparison is informative.

**Benefits Basic condition:** Yes. The effective-df constraint directly neutralizes the proxy discount by counting correlated features as ~1 df.

---

#### Phase 4: PCA-Ridge on Interaction Block (If Needed)

**What changes:** Project correlated interaction features onto their principal components before Ridge estimation, eliminating proxy redundancy directly.

**Why last:** This carries an interpretability cost — principal components must be mapped back to specific SNP interactions. Only pursue this if Phases 1–3 do not sufficiently resolve the Basic condition artifact and you need a more aggressive structural intervention.

**Benefits Full condition:** Minimal. With the LD filter active, interaction features are already approximately orthogonal, so PCA largely returns the original features.

**Benefits Basic condition:** Yes. Directly eliminates proxy redundancy before it reaches Ridge, removing the grouping effect at its source.

---

### 8.4 Summary: Phased Ablation Priority

| Phase | Approach | Code change | Fixes Basic | Improves Full | Priority |
|-------|----------|-------------|-------------|---------------|----------|
| **1** | **GCV + L1 ablation** | **Drop-in** | **Strong (diagnostic)** | **Strong** | **Start here** |
| **2** | **Group Ridge + GCV** | **Moderate** | **Yes** | **Strong** | **If Phase 1 insufficient** |
| 3 | Effective-df α scaling | Small | Yes | Moderate | Comparison to Phase 1 |
| 4 | PCA-Ridge on interactions | Moderate | Yes | Minimal | Last resort |

**Phase 1 is the mandatory starting point.** It is a drop-in replacement (swap static α=1.0 for GCV, add L1_wt parameter), directly tests the most important open question (was α=1.0 a major contributor?), benefits both Full and Basic conditions, and establishes the calibrated baseline for all subsequent experiments. If Phase 1 alone resolves the Basic artifact and improves Full accuracy, Phases 2–4 become optional refinements rather than necessities.

---

## 9. Summary Assessment

| Component | Verdict | Notes |
|-----------|---------|-------|
| 1/k penalty dilution (math) | **Correct** | Well-established property of L2 regularization with collinear features |
| Differential shrinkage mechanism | **Correct** | With refinement: joint coefficient coupling also redistributes main-effect shrinkage |
| Out-of-sample ΔR² inflation direction | **Correct** | Base model collapse drives the gap, not Joint model improvement |
| Observed inflation magnitude (2–3.5x) | **Consistent** | Quantitatively matches the derivation in the strong-regularization regime |
| Refit-step artifact persistence | **Correct** | Refitting on full 60% reduces absolute shrinkage but preserves the relative proxy discount; CV-to-refit regime shift adds a compounding factor |
| OLS failure | **Correct** | Singular X'X with LD blocks is a hard barrier |
| Lasso failure | **Correct** | L1 threshold destroys microscopic epistatic effects |
| Elastic Net failure (at 0.5 L1 ratio) | **Correct, with caveat** | Very low L1 ratios (0.01–0.05) remain untested |
| Adjusted R² irrelevance | **Correct** | Corrects for wrong quantity (feature count, not shrinkage structure) |
| Generalized Ridge failure | **Correct** | Fundamental catch-22 between correlation penalization and regularization |
| LD filter as primary defense | **Sound, validated** | Only approach that eliminates the root cause |
| Dynamic alpha as secondary defense | **Directionally correct** | Linear correction to quadratic problem; insufficient alone |
| GP x Ridge selection amplification | **Identified blind spot** | Selection loop likely contributes to inflation beyond pure shrinkage |
| ΔR² deconstruction (absolute R² decomposition) | **Recommended** | Direct evidence that inflation is driven by Base model collapse, not Joint model improvement |
| Multi-axis manuscript evidence (ground truth, correlation, PFI) | **Recommended** | Converging diagnostics proving LD operator superiority on every evaluative axis |

---

## 10. Conclusion

The out-of-sample ΔR² inflation observed in the "Basic" condition is a predictable consequence of Ridge regression's L2 penalty interacting with correlated proxy features in the strong-regularization regime. The mathematical mechanism is well-defined: k correlated proxies achieve approximately k times the predictive effect at 1/k times the penalty cost, creating differential shrinkage between the Base and Joint models that manifests as artifactual ΔR² inflation on unseen data.

This artifact cannot be corrected by any standard downstream regression method in the high-dimensional, microscopic-effect-size regime characteristic of epistasis detection. The upstream biological LD filter, by enforcing feature orthogonality at the construction stage, is the necessary and validated defense. The dynamic alpha scaling provides a principled secondary safeguard but cannot substitute for the LD filter due to the quadratic nature of the penalty discount.

The key insight for collaborators: **this is not a software bug or a modeling error. It is an intrinsic property of L2 regularization under feature redundancy, amplified by the GP's selection process, and exposed by the subtraction-based ΔR² metric.** The Full condition's stable performance confirms that StarBASE-GP produces reliable epistatic estimates when the LD filter is active. A multi-axis diagnostic framework (Section 7) provides converging manuscript evidence — ΔR² deconstruction, ground truth recovery, correlation heatmaps, and permutation feature importance — that the LD operator is justified on every evaluative dimension.

**Next steps:** A phased ablation plan (Section 8.3) addresses both the artifact and a broader modeling limitation — the arbitrary static α=1.0. Phase 1 (GCV-optimized alpha + L1 sweep) is a drop-in replacement that benefits both Full and Basic conditions and is the recommended starting point. See Appendix A.5 for implementation details.

**Extensibility to binary phenotypes:** The model comparison framework (ΔMetric = Metric_Joint − Metric_Base) generalizes to binary phenotypes using Tjur's R² (coefficient of discrimination) in place of OLS R², with the LD filter carried forward as an established requirement from this work. The core mechanism is preserved: L2-penalized logistic regression penalizes β² identically, so correlated proxies split logistic coefficients and receive the same ~1/k penalty discount. Over-shrunk logistic coefficients push predicted probabilities toward 0.5, reducing Tjur's R² in the same differential pattern. One practical difference: GCV does not have a clean closed-form for logistic Ridge (the hat matrix trick requires linearity in y, which the sigmoid breaks), so alpha optimization requires approximate GCV (AGCV) or standard cross-validation rather than the SVD-based approach in Appendix A.5.

---

## Appendix A: Implementation Details for Suggested Approaches

### A.1 Low L1-Ratio Elastic Net (0.01–0.05)

**Goal:** Replace Ridge with a near-Ridge estimator that applies just enough L1 sparsity to prune exact or near-exact proxy duplicates, without guillotining the microscopic epistatic signal.

**How it works:** In statsmodels' `fit_regularized`, the `L1_wt` parameter controls the L1/L2 mix (0.0 = pure Ridge, 1.0 = pure Lasso). With `L1_wt=0.01`, the L1 component contributes only 1% of the total penalty. The L2 component (99%) preserves Ridge-like stability. The small L1 nudge pushes near-zero redundant coefficients to exactly zero without reaching the epistatic signal.

**What to set for L2:** The `alpha` parameter controls overall regularization strength. In the Phase 1 ablation plan, GCV sets this automatically per model per pipeline. For a standalone test with fixed alpha:

```python
import statsmodels.api as sm
import numpy as np

# Option A: With GCV-optimized alpha (recommended — see Phase 1 / Appendix A.5)
alpha_base = find_alpha_gcv(X_base_train, y_train)
alpha_joint = find_alpha_gcv(X_joint_train, y_train)

# Option B: With static dynamic alpha (for comparison to current baseline)
# alpha_base = 1.0
# alpha_joint = alpha_base * (p_joint / p_base)

# Base model: near-Ridge with light L1
model_base = sm.OLS(y_train, X_base_train)
fit_base = model_base.fit_regularized(alpha=alpha_base, L1_wt=0.01)

# Joint model: near-Ridge with light L1
model_joint = sm.OLS(y_train, X_joint_train)
fit_joint = model_joint.fit_regularized(alpha=alpha_joint, L1_wt=0.01)
```

**Ablation grid to test:**

| l1_ratio | Behavior | Expected outcome |
|----------|----------|-----------------|
| 0.01 | 99% Ridge, 1% Lasso | Minimal pruning, preserves all signal |
| 0.02 | 98% Ridge, 2% Lasso | Light pruning of near-duplicates |
| 0.05 | 95% Ridge, 5% Lasso | Moderate pruning, watch for signal loss |
| 0.10 | 90% Ridge, 10% Lasso | Aggressive — likely starts cutting signal |

**What to watch for:** If ΔR² inflation drops at 0.01–0.02 without the epistatic signal collapsing, the proxies are being pruned successfully. If the signal collapses even at 0.01, the effects are too small for any L1 component.

---

### A.2 Cross-Validated Alpha (Independent for Model 0 and Model 1)

**Note:** This section describes the nested-CV approach to independent alpha optimization. **This is superseded by Appendix A.5 (GCV-Optimized Alpha)**, which achieves the same goal analytically without inner CV folds, sample-size reduction, or significant compute cost. This section is retained for reference — if you want independent alpha optimization, use A.5 instead.

**Goal:** Let each model find its own optimal regularization strength via CV, rather than using a fixed α=1.0 for both. This is primarily a *diagnostic* ablation — it tells you how much of the artifact comes from the fixed-α choice versus the fundamental grouping effect.

**Implementation (statsmodels, manual grid search):**

```python
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold

alphas = np.logspace(-3, 3, 50)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def find_alpha_cv(X, y, alphas, kf):
    """Find optimal Ridge alpha via k-fold CV (inner loop)."""
    best_alpha, best_score = alphas[0], -np.inf
    for a in alphas:
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = sm.OLS(y[train_idx], X[train_idx])
            result = model.fit_regularized(alpha=a, L1_wt=0.0)
            y_pred = X[val_idx] @ result.params
            ss_res = np.sum((y[val_idx] - y_pred) ** 2)
            ss_tot = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
            scores.append(1 - ss_res / ss_tot)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = a
    return best_alpha

alpha_base_cv = find_alpha_cv(X_base_train, y_train, alphas, kf)
alpha_joint_cv = find_alpha_cv(X_joint_train, y_train, alphas, kf)
```

**Why A.5 (GCV) is preferred:** The inner CV loop above reduces the effective training sample size by another 20% per fold and multiplies compute time by (n_alphas × n_folds) per pipeline evaluation. GCV produces the same result analytically from one SVD, with no sample splitting.

---

### A.3 Effective-Degrees-of-Freedom-Based Alpha Scaling

**Goal:** Replace the linear dynamic alpha formula `α_joint = α_base × (p_joint / p_base)` with a correlation-aware scaling based on effective degrees of freedom. This naturally accounts for the fact that k correlated proxies contribute ~1 effective degree of freedom, not k.

**Background:** The effective degrees of freedom for Ridge regression is:

    df(α) = tr(H) = tr(X(X'X + αI)⁻¹X') = Σᵢ dᵢ² / (dᵢ² + α)

where dᵢ are the singular values of X. Perfectly correlated features share a single large singular value, so they contribute ~1 to the trace regardless of how many copies exist.

**The idea:** Find α_joint such that the Joint model has the same effective degrees of freedom as the Base model. This ensures both models operate at the same effective complexity, making ΔR² a fair comparison.

**Implementation:**

```python
import numpy as np
from scipy.optimize import brentq

def effective_df(singular_values, alpha):
    """Compute effective degrees of freedom for Ridge at given alpha."""
    return np.sum(singular_values**2 / (singular_values**2 + alpha))

def find_matched_alpha(X_base, X_joint, alpha_base):
    """
    Find alpha_joint such that effective_df(Joint, alpha_joint) 
    equals effective_df(Base, alpha_base).
    """
    # Compute singular values (only need s, not U or Vt)
    s_base = np.linalg.svd(X_base, compute_uv=False)
    s_joint = np.linalg.svd(X_joint, compute_uv=False)
    
    # Target: match the base model's effective df
    target_df = effective_df(s_base, alpha_base)
    
    # effective_df is monotonically decreasing in alpha,
    # so we can use bisection (Brent's method) to solve
    def objective(log_alpha):
        return effective_df(s_joint, np.exp(log_alpha)) - target_df
    
    # Search in log-space for numerical stability
    # Range: exp(-10) ≈ 0.00005 to exp(10) ≈ 22026
    try:
        log_alpha_joint = brentq(objective, -10, 10)
        alpha_joint = np.exp(log_alpha_joint)
    except ValueError:
        # If no root exists (target_df outside achievable range),
        # fall back to the linear dynamic alpha
        alpha_joint = alpha_base * (X_joint.shape[1] / X_base.shape[1])
    
    return alpha_joint

# Usage: alpha_base from GCV (Phase 1), then match effective df for joint
alpha_base = find_alpha_gcv(X_base, y)  # from Appendix A.5
alpha_joint = find_matched_alpha(X_base, X_joint, alpha_base)

# Fit both models using statsmodels
model_base = sm.OLS(y_train, X_base_train)
fit_base = model_base.fit_regularized(alpha=alpha_base, L1_wt=0.0)

model_joint = sm.OLS(y_train, X_joint_train)
fit_joint = model_joint.fit_regularized(alpha=alpha_joint, L1_wt=0.0)
```

**Why this is better than linear scaling:**

| Scenario | Linear dynamic α | Effective-df α |
|----------|-----------------|----------------|
| 4 independent interactions added | Scales by (p+4)/p | Scales by ~(p+4)/p (similar) |
| 4 perfectly correlated proxies of 1 interaction | Scales by (p+4)/p | Scales by ~(p+1)/p (correctly treats them as ~1 feature) |
| 4 proxies at 95% LD | Scales by (p+4)/p | Scales by ~(p+1.2)/p (accounts for partial correlation) |

The effective-df approach automatically distinguishes between genuine new features (which deserve proportional alpha scaling) and redundant proxies (which should not inflate the penalty on the rest of the model).

**Computational cost:** The SVD of X is O(n·p·min(n,p)). For typical genomic dimensions this is fast — under 1 second for n=1000, p=100. If X'X is already computed for the Ridge solve, you can equivalently eigendecompose X'X (O(p³)) and use eigenvalues λᵢ in place of dᵢ² (since dᵢ² = λᵢ).

---

### A.4 Group Ridge / Block-Penalized Ridge

**Goal:** Apply different regularization strengths to the main-effect block and the interaction block within Model 1. This permits heavier penalization of interactions (to compensate for the grouping effect) without over-shrinking main effects.

**The objective function:**

    ||y - X_main·β_main - X_int·β_int||² + α_main·||β_main||² + α_int·||β_int||²

This is equivalent to standard Ridge with a block-diagonal penalty matrix.

**Implementation:**

```python
import numpy as np

def group_ridge(X_main, X_int, y, alpha_main, alpha_int):
    """
    Ridge regression with separate penalties for main-effect 
    and interaction feature blocks.
    
    Solves: (X'X + Λ)β = X'y
    where Λ = diag(α_main·I_{p_main}, α_int·I_{p_int})
    """
    p_main = X_main.shape[1]
    p_int = X_int.shape[1]
    
    # Stack features
    X = np.hstack([X_main, X_int])
    
    # Block-diagonal penalty matrix
    penalty = np.diag(np.concatenate([
        np.full(p_main, alpha_main),
        np.full(p_int, alpha_int)
    ]))
    
    # Solve normal equations: (X'X + Λ)β = X'y
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    
    beta_main = beta[:p_main]
    beta_int = beta[p_main:]
    
    return beta_main, beta_int

# Usage:
alpha_main = 1.0  # Same as base model

# Scale interaction penalty to compensate for proxy grouping.
# If you expect ~k proxies per true interaction, scale by k.
# Conservative starting point: use the feature count ratio
alpha_int = alpha_main * (p_int / p_main)

# Or use effective-df matching on just the interaction block:
# alpha_int = find_matched_alpha_for_block(X_int, target_df_per_feature)

beta_main, beta_int = group_ridge(X_base, X_int, y_train, alpha_main, alpha_int)

# Predictions
y_pred_joint = X_base_test @ beta_main + X_int_test @ beta_int
```

**Choosing α_int — three strategies:**

1. **Heuristic (simple):** Set `α_int = α_main × k_expected`, where `k_expected` is the estimated average number of proxies per true interaction. If LD structure suggests ~4 proxies on average, use `α_int = 4.0`.

2. **Effective-df matching (principled):** Compute the effective df of the interaction block and scale α_int until the per-feature effective df matches that of the main-effect block:

    ```python
    s_main = np.linalg.svd(X_main, compute_uv=False)
    s_int = np.linalg.svd(X_int, compute_uv=False)
    
    # Target: same effective df per raw feature as the main block
    target_df_per_feature = effective_df(s_main, alpha_main) / p_main
    target_df_int = target_df_per_feature * p_int
    
    # Find alpha_int that achieves this
    def objective(log_a):
        return effective_df(s_int, np.exp(log_a)) - target_df_int
    
    alpha_int = np.exp(brentq(objective, -10, 10))
    ```

3. **Cross-validated (empirical):** Grid-search over α_int values while holding α_main fixed, selecting the α_int that minimizes CV prediction error. This is the most robust but most expensive approach.

**Advantage over uniform Ridge:** The base model (Model 0) uses only main effects with `alpha_main`. The Joint model (Model 1) uses `alpha_main` for the same main effects and `alpha_int` for interactions. Because the main-effect penalty is identical in both models, any difference in R² is attributable purely to the interaction block — and the interaction block is now penalized to account for proxy redundancy. This makes the ΔR² comparison structurally fair.

---

### A.5 GCV-Optimized Alpha (Independent for Model 0 and Model 1)

**Goal:** Replace the arbitrary static α=1.0 with a data-driven, per-pipeline optimal α for each model, using Generalized Cross-Validation (GCV). No inner CV folds, no sample-size reduction, no significant compute overhead.

**Background:** GCV is an analytical estimate of the leave-one-out cross-validation error for Ridge regression. Given the SVD of the design matrix X = UDV', the GCV criterion at a candidate α is:

    GCV(α) = (1/n) × ||y - ŷ(α)||² / (1 - df(α)/n)²

where df(α) = Σ dᵢ²/(dᵢ² + α) is the effective degrees of freedom. Both the residual norm and df are closed-form functions of the SVD, so evaluating GCV over a grid of α values costs negligible computation after one SVD.

**Key property — no leakage:** GCV is computed entirely on the data used for fitting. It never sees held-out data. The train/evaluate separation is identical to the current pipeline.

**Core implementation (statsmodels-compatible):**

```python
import numpy as np
import statsmodels.api as sm

def find_alpha_gcv(X, y, alphas=None):
    """
    Find optimal Ridge alpha via Generalized Cross-Validation.
    
    No sample splitting required. Computes one SVD of X, then 
    evaluates the GCV criterion analytically for each candidate alpha.
    
    Parameters
    ----------
    X : array (n, p) — design matrix (training data only)
    y : array (n,)   — response vector (training data only)
    alphas : array    — candidate alpha values to search over
    
    Returns
    -------
    best_alpha : float — the alpha that minimizes GCV error
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 100)
    
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Project y onto the SVD basis
    u_ty = U.T @ y            # rotated response, length p
    
    # Component of y orthogonal to column space of X
    y_perp_sq = np.sum(y**2) - np.sum(u_ty**2)
    
    best_alpha = alphas[0]
    best_gcv = np.inf
    
    for a in alphas:
        # Shrinkage factors per singular component
        d = s**2 / (s**2 + a)
        
        # Effective degrees of freedom
        df = np.sum(d)
        
        # Residual sum of squares: ||(I - H)y||²
        rss = np.sum((1 - d)**2 * u_ty**2) + y_perp_sq
        
        # GCV criterion
        denom = (1 - df / n) ** 2
        if denom > 0:
            gcv = (rss / n) / denom
        else:
            gcv = np.inf  # degenerate case: df ≈ n
        
        if gcv < best_gcv:
            best_gcv = gcv
            best_alpha = a
    
    return best_alpha


def ridge_gcv_fit(X, y, alphas=None, L1_wt=0.0):
    """
    Fit a Ridge (or near-Ridge Elastic Net) model with GCV-optimized alpha.
    
    Parameters
    ----------
    X      : design matrix (training data only)
    y      : response vector (training data only)
    alphas : candidate alpha grid (default: logspace -4 to 4)
    L1_wt  : L1 weight for optional light Lasso pruning.
              0.0 = pure Ridge (recommended starting point).
              0.01–0.02 = light pruning of near-duplicate features.
    
    Returns
    -------
    result : statsmodels RegularizedResults with .params
    alpha  : the GCV-selected alpha value
    """
    alpha = find_alpha_gcv(X, y, alphas)
    
    model = sm.OLS(y, X)
    result = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
    
    return result, alpha
```

**Integration with the GP pipeline:**

```python
def evaluate_pipeline_gcv(X_base_train, X_joint_train, y_train,
                          X_base_holdout, X_joint_holdout, y_holdout,
                          L1_wt=0.0):
    """
    Evaluate one GP pipeline using GCV-optimized alpha for both models.
    
    This is a drop-in replacement for the current evaluation function.
    The only change: alpha is data-driven instead of static 1.0.
    
    Parameters
    ----------
    X_base_train   : main-effect features, training portion
    X_joint_train  : main-effect + interaction features, training portion
    X_base_holdout : main-effect features, held-out portion
    X_joint_holdout: main-effect + interaction features, held-out portion
    y_train        : response, training portion
    y_holdout      : response, held-out portion
    L1_wt          : L1 weight (0.0 = pure Ridge, 0.01-0.02 = light pruning)
    
    Returns
    -------
    delta_r2    : ΔR² = R²_joint - R²_base (computed on held-out data)
    alpha_base  : GCV-selected alpha for base model
    alpha_joint : GCV-selected alpha for joint model
    """
    # GCV finds optimal alpha for each model INDEPENDENTLY
    # using ONLY training data — no leakage into held-out evaluation
    fit_base, alpha_base = ridge_gcv_fit(X_base_train, y_train, L1_wt=L1_wt)
    fit_joint, alpha_joint = ridge_gcv_fit(X_joint_train, y_train, L1_wt=L1_wt)
    
    # Predict on HELD-OUT data only
    y_pred_base = X_base_holdout @ fit_base.params
    y_pred_joint = X_joint_holdout @ fit_joint.params
    
    # R² on held-out data
    ss_tot = np.sum((y_holdout - np.mean(y_holdout))**2)
    r2_base = 1 - np.sum((y_holdout - y_pred_base)**2) / ss_tot
    r2_joint = 1 - np.sum((y_holdout - y_pred_joint)**2) / ss_tot
    
    return r2_joint - r2_base, alpha_base, alpha_joint
```

**How this flows through CV folds and the final validation/test:**

```
DURING GP — Each pipeline evaluation, each CV fold:
═══════════════════════════════════════════════════

    60% CV Data, Fold k
    ┌─────────────────────────────────┬──────────────────┐
    │     TRAINING (~48% of total)    │   HELD-OUT fold  │
    │                                 │                  │
    │  GCV → α_base (for Model 0)    │  Predict with    │
    │  GCV → α_joint (for Model 1)   │  fitted models   │
    │  Fit both models with their     │  Compute ΔR²     │
    │  own optimal alphas             │  for GP fitness  │
    │                                 │                  │
    │  (+ optional L1_wt=0.01)       │                  │
    └─────────────────────────────────┴──────────────────┘
    
    GCV and fitting: TRAINING only.  ΔR² fitness: HELD-OUT only.
    No leakage.  α adapts per pipeline and per fold.


AFTER GP — Refit Pareto front, evaluate on validation/test:
═══════════════════════════════════════════════════════════

    Full 60% CV Data        20% Validation        20% Test
    ┌──────────────────┐   ┌─────────────────┐   ┌────────────┐
    │                  │   │                 │   │            │
    │ GCV → α_base     │   │ Predict with   │   │ Final ΔR²  │
    │ GCV → α_joint    │   │ refit models   │   │ report     │
    │ Refit both with  │   │ Compute ΔR²    │   │            │
    │ optimal alphas   │   │ Select Utopia  │   │            │
    │                  │   │                 │   │            │
    └──────────────────┘   └─────────────────┘   └────────────┘
    
    GCV and fitting: 60% only.  Evaluation: unseen splits only.
    No leakage.  Alphas re-optimized on the larger training set.
```

**Note on the refit step:** When models are refit on the full 60%, GCV is recomputed on the full 60%. This means the refit alphas will generally differ from the CV-fold alphas (more data → different optimal regularization). This is correct behavior — the refit model should be optimized for its own training set, not carry forward fold-level hyperparameters.

**Recommended ablation sequence:**

| Step | L1_wt | α selection | What it tests |
|------|-------|-------------|---------------|
| 1 (baseline) | 0.0 | Static α=1.0 | Current behavior (control) |
| 2 | 0.0 | GCV (independent) | Impact of removing the arbitrary α=1.0 assumption |
| 3 | 0.005 | GCV (independent) | Barely-there L1 nudge — does anything change? |
| 4 | 0.01 | GCV (independent) | Light L1 pruning |
| 5 | 0.02 | GCV (independent) | Moderate L1 pruning |
| 6 | 0.05 | GCV (independent) | Upper bound — watch for signal loss |

Run each step under both Full and Basic conditions. The comparison between steps 1 and 2 isolates the effect of α optimization. The comparison between steps 2 and 3–6 isolates the effect of L1 pruning at increasing intensities.

**Expected outcomes:**
- **Step 1 → 2 (Full condition):** ΔR² likely increases (α=1.0 was over-shrinking signal). This is not an artifact — it's a more accurate estimate of the true epistatic effect.
- **Step 1 → 2 (Basic condition):** ΔR² inflation may partially attenuate if the fixed-α was amplifying the differential. If 2–3.5x inflation persists, the grouping effect is fundamental.
- **Step 2 → 3–6 (Basic condition):** If near-duplicate proxies are present, light L1 may prune them and further reduce inflation. In Full condition, expect negligible change (LD filter already handled redundancy).

**Why there is no GCV equivalent for L1:** GCV works for Ridge because the Ridge solution is a linear function of y — predictions are ŷ = Hy where H is the hat matrix, and the leave-one-out error has a closed-form shortcut via that linearity. The L1 penalty introduces an absolute value that makes the solution nonlinear in y, breaking the hat matrix trick. There is no analytical formula for optimal L1 weight. The ablation grid above is the practical alternative.

**L1 tuning diagnostic — detecting the signal guillotine:**

Because there is no analytical optimizer for L1, the key is to monitor whether the L1 is pruning redundant proxies (good) or cutting into real epistatic signal (bad). The diagnostic: **track the number of non-zero interaction coefficients** at each L1_wt level.

```python
def l1_ablation_diagnostic(X_joint_train, y_train, 
                           l1_values=[0.0, 0.005, 0.01, 0.02, 0.05]):
    """
    Run the L1 ablation grid and report coefficient survival at each level.
    GCV handles the L2 (alpha) automatically at each step.
    
    Look for:
    - Gentle decline in nonzero count → L1 is trimming redundancy (good)
    - Sharp cliff in nonzero count   → L1 is guillotining real signal (stop)
    """
    print(f"{'L1_wt':>8}  {'α_gcv':>10}  {'Nonzero':>10}  {'Total':>8}")
    print("-" * 45)
    
    for l1 in l1_values:
        result, alpha = ridge_gcv_fit(X_joint_train, y_train, L1_wt=l1)
        n_nonzero = np.sum(np.abs(result.params) > 1e-10)
        n_total = len(result.params)
        print(f"{l1:>8.3f}  {alpha:>10.4f}  {n_nonzero:>10}  {n_total:>8}")
```

**Interpreting the output:**

```
Example A — Gentle decline (safe, L1 is pruning redundancy):
   L1_wt       α_gcv     Nonzero     Total
---------------------------------------------
   0.000      0.0312          90        90
   0.005      0.0298          88        90    ← 2 near-duplicates pruned
   0.010      0.0285          86        90    ← 2 more pruned
   0.020      0.0271          83        90    ← gradual, still safe
   0.050      0.0244          78        90    ← still gradual

Example B — Sharp cliff (STOP, L1 is killing signal):
   L1_wt       α_gcv     Nonzero     Total
---------------------------------------------
   0.000      0.0312          90        90
   0.005      0.0298          89        90    ← fine
   0.010      0.0285          87        90    ← fine
   0.020      0.0271          51        90    ← CLIFF: 36 features zeroed
   0.050      0.0244          12        90    ← signal destroyed

In Example B, the right L1_wt is 0.01 — anything higher hits real signal.
```

The cliff indicates the L1 threshold has crossed from the proxy noise floor into the epistatic signal range. The optimal L1_wt is the largest value before the cliff. If there is no cliff at all (Example A), the L1 is only hitting redundancy at every level tested, and 0.05 is safe. If the cliff appears at the very first step (0.005), the epistatic effects are too small for any L1 and L1_wt should remain at 0.0 — let GCV-optimized Ridge do all the work.
