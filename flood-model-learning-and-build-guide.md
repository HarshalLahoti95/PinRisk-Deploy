# PinRisk — Flood-Risk Modeling: Learning Path & Baseline Build Guide

**Audience:** PinRisk founding team (CS majors, coursework-level ML, no cat-risk/climate domain background).
**Goal:** Get competent enough — fast — to build the *most accurate defensible baseline* flood model for the MVP (Mumbai/Chennai urban flood), and know exactly where to spend limited reading time.

**How to read this doc:** Every reading item is tagged **[MUST-READ]** or **[NICE-TO-HAVE]**. If you only do the MUST-READs you can build the MVP; the NICE-TO-HAVEs make you defensible in an investor/insurer technical conversation. Section 9 is the actual week-by-week plan — if you read nothing else, read Section 9 and Section 10.

**One framing you must internalize before anything else:** what you are building is *not* a "predict tomorrow's flood" model and *not* an "AI segments a flood from a satellite image" model. It is a **flood *susceptibility / risk* model** — a static-to-slowly-evolving map of *how prone each location is* to flooding, calibrated against past events. This is methodologically almost identical to **species distribution modeling (SDM)** in ecology (predict presence/absence of a phenomenon across space from environmental predictors, with very few positive samples). That analogy is the single most useful mental model you have, because SDM has 20 years of hard-won lessons about exactly your problem: rare positives, spatial autocorrelation, pseudo-absence sampling, and misleading validation. Keep it in mind throughout.

---

## 1. ML learning path (targeted — skipping generic ML you already have)

You have supervised learning, training, and evaluation. What you are missing is the *spatial* and *rare-event* machinery. Read in this order.

### 1.1 Geospatial data handling & remote sensing (foundation — everything else sits on this)
- **[MUST-READ]** *Geographic Data Science with Python* — Rey, Arribas-Bel & Wolf (2023, free online at **geographicdata.science/book**). This teaches the Python spatial stack (geopandas, spatial weights, ESDA, spatial autocorrelation) in exactly the idiom you'll code in. Read Parts I–II fully; skim the modeling part.
- **[MUST-READ]** *Cloud-Based Remote Sensing with Google Earth Engine: Fundamentals and Applications* — Cardille et al. (2023, open-access, Springer). GEE is how you'll actually pull and process DEMs, Sentinel, WorldCover, etc. at scale without downloading terabytes. Read the fundamentals + the flood/SAR chapters.
- **[NICE-TO-HAVE]** *Remote Sensing and Image Interpretation* — Lillesand, Kiefer & Chipman. Classic reference for what optical vs. radar bands actually measure. Use as a lookup, not a cover-to-cover read.

**Why it matters:** 80% of flood-model work is data engineering — reprojecting, resampling, aligning rasters, extracting features per grid cell. If you're not fluent here, the ML is irrelevant.

### 1.2 Spatial statistics & (critically) spatial cross-validation
This is the subfield most likely to silently ruin your model. Standard random train/test splits **cheat** when data is spatially autocorrelated — neighboring cells leak information, so your reported accuracy is inflated and the model fails on truly new areas.
- **[MUST-READ]** Roberts et al. (2017), *"Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"*, **Ecography**. This is the paper that will save you from reporting a fake 95% accuracy. Understand block/spatial CV.
- **[MUST-READ]** Ploton et al. (2020), *"Spatial validation reveals poor predictive performance of large-scale ecological mapping models"*, **Nature Communications**. Short, brutal, and directly transferable — it's the cautionary tale for exactly your setup.
- **[NICE-TO-HAVE]** Meyer & Pebesma (2021), *"Predicting into unknown space? Estimating the area of applicability of spatial prediction models"*, **Methods in Ecology and Evolution** (+ the `CAST` R package idea). Teaches you to say *where your model is trustworthy* — a credibility feature for insurers.
- **[NICE-TO-HAVE]** *Spatial Data Science with Applications in R* — Pebesma & Bivand (2023, free online). Deeper reference on spatial autocorrelation (Moran's I), variograms. R-based but concepts transfer.

### 1.3 Class imbalance / rare-event modeling
Floods are rare → you have few positive cells and a huge negative majority. Naïve accuracy is meaningless (a model saying "never floods" scores 98%).
- **[MUST-READ]** King & Zeng (2001), *"Logistic Regression in Rare Events Data"*, **Political Analysis**. Foundational; explains why rare-event probabilities are biased and how to correct them.
- **[MUST-READ]** Barbet-Massin et al. (2012), *"Selecting pseudo-absences for species distribution models: how, where and how many?"*, **Methods in Ecology and Evolution**. This is the SDM playbook for choosing your *non-flooded* samples — the single most consequential and most-often-botched design choice in susceptibility mapping. (Where you draw "not flooded" cells from largely determines your answer.)
- **[NICE-TO-HAVE]** He & Garcia (2009), *"Learning from Imbalanced Data"*, **IEEE TKDE**. Good survey. **Caution:** SMOTE and oversampling are popular but *dangerous in spatial settings* — synthetic points break spatial structure and inflate CV scores. Prefer thoughtful pseudo-absence design + class weights over SMOTE. Treat this survey as "know it exists," not "apply it blindly."

### 1.4 Uncertainty quantification (your non-negotiable credibility feature)
- **[MUST-READ]** Angelopoulos & Bates (2021/2023), *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"* (arXiv). The most practical, model-agnostic way to put honest confidence bands on any predictor. This is how you deliver the "confidence band" the brief demands, defensibly.
- **[NICE-TO-HAVE]** Meinshausen (2006), *"Quantile Regression Forests"*, **JMLR**. If you go with random forests, this gives per-prediction intervals almost for free.
- **[NICE-TO-HAVE]** Beven & Binley (1992), GLUE methodology (hydrology's own uncertainty framework). Domain-flavored; useful vocabulary when talking to hydrologists, less so for the ML baseline.

---

## 2. Climate / domain learning path

You cannot skip this. The credibility gap between "CS students who trained a classifier" and "a defensible cat model" is entirely domain reasoning. Order:

### 2.1 Catastrophe modeling fundamentals (the 4-module spine — read this FIRST)
- **[MUST-READ]** *Natural Catastrophe Risk Management and Modelling: A Practitioner's Guide* — Mitchell-Wallace, Jones, Hillier & Foote (2017, Wiley). This **is** the industry bible for the hazard→exposure→vulnerability→financial structure your MVP mirrors. Read the framework chapters and the vulnerability/financial chapters closely. If you read one book in this whole doc, it's this one — it's the language your insurer customers speak.
- **[NICE-TO-HAVE]** Grossi & Kunreuther (2005), *Catastrophe Modeling: A New Approach to Managing Risk*. Older, shorter, foundational conceptual intro. Good weekend read to orient before the Practitioner's Guide.

### 2.2 Flood physics & hydrology (enough to not embarrass yourselves)
You need to *distinguish the flood types and know what drives each* — not solve Navier-Stokes.
- **[MUST-READ]** Bates (2022), *"Flood Inundation Prediction"*, **Annual Review of Fluid Mechanics**. Written by the founder of Fathom (your competitor). The single best modern overview of how flood inundation is actually modeled, and the accuracy limits. Read it twice.
- **[MUST-READ] — concept, not a book:** nail the three flood types cold:
  - **Pluvial** (rain falls faster than drainage can clear it — *this is the dominant Mumbai/Chennai urban peril*; driven by rainfall intensity + imperviousness + local depressions + drainage capacity).
  - **Fluvial** (rivers overtop banks — driven by upstream catchment, HAND, distance-to-river).
  - **Coastal** (storm surge / sea level — matters for Mumbai's coastline but secondary for the MVP).
  Urban India's headline disasters (Mumbai 2005, Chennai 2015) are *pluvial-dominant with fluvial components* — your feature set and your narrative must reflect that.
- **[NICE-TO-HAVE]** *Fundamentals of Hydrology* — Tim Davie (accessible) **or** *Applied Hydrology* — Chow, Maidment & Mays (the rigorous classic; read selectively — the chapters on rainfall-runoff, return periods, and frequency analysis). Don't read Chow cover to cover.

### 2.3 Vulnerability & depth-damage functions (how physical → financial)
- **[MUST-READ]** Huizinga, de Moel & Szewczyk (2017), *"Global flood depth-damage functions: Methodology and the database with guidelines"*, **JRC Technical Report (EUR 28552)**. This is the actual source of the curves you'll borrow. It even has continental/Asia curves. Read the methodology so you know what you're assuming.
- **[MUST-READ]** Merz, Kreibich, Schwarze & Thieken (2010), *"Review article: Assessment of economic flood damage"*, **NHESS**. Explains why damage estimation is genuinely uncertain and what drives that uncertainty — inoculates you against overclaiming precision on the ₹ numbers.
- **[NICE-TO-HAVE]** FEMA *Hazus Flood Technical Manual* — reference for the US depth-damage curve library; useful to compare against JRC, but US building stock ≠ Indian, so adapt, don't copy.

### 2.4 Flood susceptibility ML literature (your actual method)
- **[MUST-READ]** Mosavi, Ozturk & Chau (2018), *"Flood Prediction Using Machine Learning Models: Literature Review"*, **Water**. Maps the whole ML-for-flood landscape so you can position your choices.
- **[NICE-TO-HAVE]** A few concrete "flood susceptibility mapping with RF/XGBoost" case papers (Tehrany et al.; Chen et al.; Wang et al. — there are dozens, many India/Asia-specific). Read 2–3 for the *feature list and validation ritual* (AUC-ROC against known flood inventory), and note the recurring methodological weakness: almost none do proper spatial CV. That gap is *your* opportunity to be more rigorous.

---

## 3. Baseline model recommendation

### What "baseline" correctly means
A baseline is not "our first attempt." It's a **deliberately simple, fully-validated reference** that any fancier model must beat on a *spatially-honest* metric. You should build a *trivial* baseline and a *real* baseline, and always report both.

### Ranked recommendation for the hazard/susceptibility module

| Rank | Model | Why / trade-off for your constraints |
|---|---|---|
| **0 (trivial baseline)** | **HAND threshold + elevation** — "flooded if HAND < X m." | Pure physics, zero training. If your ML can't beat this on spatial CV, your ML is adding noise, not signal. Non-negotiable to include. |
| **1 (recommended primary)** | **Gradient-boosted trees (XGBoost / LightGBM)** | Best accuracy-per-effort on tabular geospatial features; handles nonlinearity & feature interactions (imperviousness × rainfall); works with modest compute; SHAP gives interpretability. Downside: overfits easily with few positives — *must* pair with spatial CV + class weights + regularization. |
| **2 (interpretable co-baseline)** | **Random Forest** | Slightly more robust out-of-the-box than boosting, less tuning, free uncertainty via quantile forests. Often ties XGBoost on susceptibility tasks. Run it alongside GBM — cheap insurance. |
| **3 (the "we can explain every coefficient" model)** | **Logistic / penalized logistic regression (with rare-event correction)** | Fully interpretable, defensible to an actuary, hard to overfit. Lower ceiling on accuracy but the *honesty floor*. Keep it in the ensemble as the "explainable" fallback. |
| **4 (explicitly NOT for MVP)** | **CNN / U-Net / geospatial foundation models on rasters** | These shine at *flood-extent segmentation from imagery* (mapping a flood that's happening), **not** at susceptibility from few labels. With sparse positive labels and non-expert maintainers, deep learning will overfit, be uninterpretable, and be un-maintainable. Use CNNs only for the *label-making* step (Section 8), never as the risk model in v1. |

**Recommended MVP stack:** trivial HAND baseline → penalized logistic regression → XGBoost + Random Forest, all evaluated identically, ensemble the top two, SHAP for explanation, conformal/quantile intervals for confidence.

### What "good validation" looks like (this is the make-or-break)
1. **Build a flood inventory** = the ground-truth "where it actually flooded" for your validation event (Chennai 2015 extent from Sentinel-1 SAR; Mumbai 2005 reconstructed from records — see Sections 5 & 8, and note the Mumbai data caveat).
2. **Spatial block cross-validation**, never random splits (Roberts 2017). Report metrics per spatial fold.
3. **Right metrics:** **AUC-ROC and AUC-PR** (PR especially, given imbalance), plus a **reliability/calibration curve** (do predicted 30%-risk cells flood ~30% of the time?). Report the **confusion matrix at a chosen threshold** and, ideally, a **spatial overlay map of predicted-vs-actual**. Never report plain accuracy.
4. **Temporal/event holdout as the headline claim:** train on features that predate the event, predict, then compare to the observed extent. "We predicted the flooded footprint of Chennai 2015 with AUC-PR of X on held-out blocks" is the sentence that sells the company. Say it *only* if it's true and spatially honest.
5. **Report where the model is NOT applicable** (area-of-applicability, Meyer & Pebesma). Insurers trust models that admit their limits.

---

## 4. Parameter weighting — how to actually decide it

Stop asking "what weight for elevation vs. rainfall?" Assigning weights by hand is the amateur move (it's the old "AHP / weighted-overlay" GIS method, and it's exactly what you can beat). **You learn the weights from data, constrain them with physics, and then check them against reality.** Four layers:

1. **Learn them, don't guess them.** The model *is* the weighting. Tree models learn nonlinear, interacting importances; logistic regression gives signed coefficients. You never type a number like "elevation = 0.3."

2. **Interpret with SHAP** (SHapley Additive exPlanations — Lundberg & Lee, 2017, NeurIPS) **[MUST-READ paper]**. SHAP tells you, per feature and per location, how much each predictor pushed the risk up or down. This is both your interpretability layer and your sanity check. Raw tree "feature_importance" is biased toward high-cardinality features — prefer SHAP.

3. **Impose physical priors as *constraints*, not values.** You don't set magnitudes, you set *directions and shapes*:
   - Risk should **decrease** with elevation and with HAND (monotonic ↓).
   - Risk should **increase** with imperviousness, rainfall intensity, and drainage density-to-capacity mismatch (monotonic ↑).
   - Distance-to-drainage/coast: closer → higher fluvial/coastal risk.
   XGBoost/LightGBM support **monotonic constraints** — use them. This prevents the model from learning physically absurd relationships just to fit noise, which is critical with few positives.

4. **Sensible physical priors on what dominates *urban* flooding** (so you can smell-test the learned weights):
   - For **pluvial urban** flooding (Mumbai/Chennai): **rainfall intensity + imperviousness + local topographic depressions (low HAND / low elevation) + drainage capacity** dominate. Soil matters less in dense impervious cities than in rural catchments.
   - For **fluvial**: **HAND, distance-to-river, upstream catchment area** dominate.
   - If SHAP tells you rainfall and HAND barely matter but soil type dominates in central Mumbai, **something is wrong** (likely label leakage or bad pseudo-absences) — go debug, don't ship.

5. **Validate the learned weights against reality** three ways: (a) SHAP directions match the physical priors above; (b) high-risk cells coincide with historically flooded wards (Mumbai's low-lying areas — Kurla, Sion, Dharavi belt — are well documented); (c) ablation — drop a feature, see if spatial-CV performance degrades the way physics says it should. Convergence of all three = defensible weights.

**Honest caveat:** feature importances are *correlational and dataset-dependent*, not causal. Two correlated predictors (elevation & HAND) will split importance unstably. Report importances as "what the model used," not "what physically causes floods."

---

## 5. Datasets per parameter

**Yes — use a different, best-in-class dataset per parameter, then fuse.** This is standard and correct. The hard part is *resolution fusion*: pick a **common analysis grid** (recommend **30 m**, matching your DEM), and resample every layer to it — downsample coarse layers (rainfall) with interpolation, aggregate fine layers (10 m land cover) by majority/mean. Do all fusion in a single CRS (use a metric projection like UTM 43N/44N for India, not raw lat/lon, so distances and areas are real). Track each layer's *native* resolution as a provenance/uncertainty field.

| Parameter | Recommended dataset(s) | Native res | Known India limitations |
|---|---|---|---|
| **Elevation (DEM)** | **FABDEM** (primary — buildings & forests removed, best for urban flood), Copernicus GLO-30 (base), MERIT-Hydro (for hydrological conditioning) | ~30 m | Urban DEMs globally struggle with sub-30 m drainage detail; FABDEM is a correction of Copernicus, not lidar-quality. Real limit for narrow-street pluvial flow. |
| **HAND / drainage** | Derive **HAND** yourself from MERIT-Hydro / hydrologically-conditioned DEM (use `pysheds`, WhiteboxTools, or RichDEM); HydroSHEDS/HydroRIVERS for river network | ~30 m | HAND is weak in very flat coastal urban terrain (Mumbai) — small DEM errors → big HAND errors. Flag this. |
| **Rainfall** | **IMD gridded** (0.25°, India-official — use for credibility & the historical events), **GPM IMERG** (~10 km, satellite, near-real-time), ERA5 (reanalysis, for consistency) | 0.25° (~25 km) / ~10 km | **This is your coarsest, most limiting layer.** 25 km rainfall over a city that floods at street scale is a genuine mismatch — you'll need to treat rainfall as a near-uniform forcing per event and let terrain/imperviousness create the local variation. Be honest about this in the deck. |
| **Land use / imperviousness** | **ESA WorldCover** (10 m), Dynamic World (10 m, near-real-time), + optional impervious-surface products | 10 m | Class definitions are global; "built-up" doesn't distinguish drainage quality. Good enough as an imperviousness proxy. |
| **Soil** | SoilGrids (ISRIC, ~250 m), FAO HWSD | ~250 m | Coarse; low value-add in dense impervious urban cores (see Section 4). Include but don't expect it to dominate. |
| **Exposure / buildings** | **Microsoft Global Building Footprints** + **Google Open Buildings** (fuse both; MS better in some Indian cities, Google in others), OSM for attributes | vector | No reliable construction-type or value attributes for India — you'll proxy value (footprint area × assumed value/m² by land-use class; night-lights VIIRS or GHSL/WorldPop as value/exposure proxy). This assumption is a documented uncertainty, not a fact. |
| **Population / value proxy** | WorldPop (100 m), Meta HRSL, GHSL | 100 m | Proxy only; fine for relative exposure, weak for absolute ₹. |
| **Historical flood extent (LABELS)** | **Chennai 2015: Sentinel-1 SAR** (Copernicus, cloud-penetrating — the workhorse). **Mumbai 2005: reconstruct** from Dartmouth Flood Observatory / Global Flood Database / MODIS / municipal & news records | 10–20 m (S1) | **Sentinel-1 launched 2014 → it did NOT observe Mumbai 2005.** This is why Chennai 2015 is the stronger *validation* city even if Mumbai is the *demo* city. Don't paper over this. |
| **Benchmark hazard layers** | WRI Aqueduct Floods (free), JRC Global Flood maps, Bhuvan Flood Hazard Atlas (ISRO) | coarse | Use to sanity-check *your* map against existing ones — not as ground truth. |
| **Pincode polygons** | Community/derived datasets (data.gov.in derivations, OSM-based approximations) | approximate | **No authoritative open pincode shapefile exists in India.** Compute the model on the grid; aggregate to whatever pincode polygons you accept, and label them "approximate boundaries." This gap is part of why the problem is unsolved — state it as a feature of the market, not a bug in your product. |

---

## 6. Competitor & prior-art teardown

**How incumbents actually work:**
- **Fathom (Bristol)** — global 30 m (and finer) flood hazard from a **physics-based hydrodynamic engine (LISFLOOD-FP)** run at scale, with climate-adjusted variants. Founded by Paul Bates (read his 2022 review). This is the technical gold standard for *hazard*. They do physics; you'll do statistics — know the trade-off cold.
- **JBA** — flood maps + cat models, hydrodynamic + statistical, strong in UK/global insurance.
- **Verisk (formerly AIR Worldwide)** and **Moody's RMS** — full cat-model platforms across perils; their flood models blend hydrodynamic hazard + detailed exposure/vulnerability + financial (the full 4-module stack, industrialized). Largely US/EU/Japan-strong, **historically weaker & coarser in India** — that's your wedge.
- **First Street (US)** — consumer-facing "Flood Factor," property-level, blends Fathom-style hazard with their own layers and *forward-looking climate adjustment*; publishes methodology. Closest in spirit to your consumer angle. US-only.

**How the *academic* "flood susceptibility mapping" literature works** (and what you'll actually replicate): assemble a flood inventory (presence points) + non-flood points → stack geo-predictors → train RF/XGBoost/logistic → validate with AUC-ROC against the inventory → produce a susceptibility map. **The general accepted pipeline** = inventory + predictors → ML classifier → AUC validation → map.

**Where you have room to be genuinely better/more local:**
1. **India-tuned & pincode-resolved** where incumbents are regional and coarse.
2. **Methodological rigor the susceptibility literature usually skips:** proper *spatial* cross-validation, honest calibration curves, and *explicit uncertainty + area-of-applicability*. Most published susceptibility maps are spatially overfit — being correct here is a real, defensible edge.
3. **Forward-looking climate adjustment** baked in (Section 7) — most academic susceptibility maps are purely historical.
4. **The physical-to-financial bridge** (vulnerability + ₹ loss) — most academic maps stop at hazard; incumbents charge a fortune for the financial layer. Doing it transparently and locally is the commercial wedge.

**Where you cannot beat them (be honest):** you will *not* out-physics Fathom on hazard with a statistical model on modest compute. Your pitch is *local + forward-looking + transparent + cheaper*, not "more physically accurate than a hydrodynamic engine."

---

## 7. Accounting for climate change (non-stationarity) — the core USP

**The conceptual pivot:** a standard flood model assumes **stationarity** — that the statistical distribution of hazards is fixed, so the past predicts the future. Climate change breaks this.
- **[MUST-READ]** Milly et al. (2008), *"Stationarity Is Dead: Whither Water Management?"*, **Science**. Two pages, foundational, and it's literally the intellectual thesis of your company. Quote it in the deck.

**What concretely must change vs. a historical model:**
1. **Rainfall is the lever, not topography.** Terrain, HAND, imperviousness change slowly; what changes with climate is **rainfall intensity and extreme-event frequency** (a warmer atmosphere holds ~7%/°C more moisture — Clausius-Clapeyron — so extreme downpours intensify). Your forward-looking layer works by **re-driving the *same* trained susceptibility/vulnerability model with *shifted rainfall inputs*.**
2. **Get shifted rainfall from climate projections:** **CMIP6** (raw global models, too coarse alone), **CORDEX South Asia** (regionally downscaled, better for India), **NEX-GDDP-CMIP6** (NASA statistically-downscaled, ~25 km, easiest to use). Pick a scenario (SSP2-4.5 "middle", SSP5-8.5 "high") and a horizon (2050).
3. **Method (realistic for MVP):** rather than re-simulating physics, apply a **"delta / change-factor" approach** — compute how much extreme-rainfall intensity (e.g., the 1-in-100-year daily rainfall) shifts between the historical baseline and the future scenario in the projections, then push that shifted rainfall through your existing model to get a *future risk map*. This is standard, tractable, and explainable.

**The honest limits (state these plainly — it builds trust):**
- Climate models are **coarse (100+ km native)** and **downscaling extreme, convective rainfall — exactly what floods Indian cities — is one of the least reliable things climate science does.** Monsoon dynamics are a known weak spot for CMIP6.
- **Deep uncertainty:** scenario choice (SSP) + model choice + internal variability can span a huge range. You must present the forward-looking numbers as a **scenario range with wide bands**, never a single 2050 figure. Overprecision here is the fastest way to lose credibility with a sophisticated buyer.
- Frame the climate layer as **"directionally rigorous scenario analysis,"** not prediction. That framing is both honest and still far ahead of static incumbents.

---

## 8. Pretrained / fine-tunable models

**Key distinction first — this trips everyone up:** geospatial/EO foundation models are trained for **image → segmentation/classification** (e.g., "which pixels in this Sentinel scene are water/flooded *right now*"). They are **not** susceptibility/risk models. So their real role for you is **making your validation labels** (map the historical flood extent from SAR), *not* being your risk engine.

| Model / resource | What it's good for | Fine-tune realistic for you? | Worth it vs. your own baseline? |
|---|---|---|---|
| **Prithvi (IBM + NASA)** — Prithvi-EO on HuggingFace, incl. flood-mapping fine-tunes | Segmenting flood/water extent from HLS/Sentinel imagery | Partially — inference/light fine-tune is feasible; needs GPU | **Yes for label-making** (get flood extent for Chennai 2015). **No** as your risk model. |
| **Clay** (clay-foundation) | General EO embeddings; could provide features | Embeddings usable; full fine-tune heavier | Optional. Embeddings-as-features is an *advanced* experiment, not MVP. |
| **SatMAE / Satlas / DOFA** | EO representation learning / segmentation | Research-grade; more effort | Skip for MVP. |
| **Sen1Floods11** (Bonafilia et al. 2020) — flood segmentation *benchmark dataset* | Off-the-shelf flood-vs-nonflood training data + pretrained baselines for SAR flood mapping | It's a dataset+baselines — very usable | **Yes** — the fastest path to a working *flood-extent-from-SAR* labeler. |
| Published *susceptibility* models | Rarely released as reusable weights; you'll reimplement the method, not download it | n/a | Reimplement the pipeline; don't expect a plug-in model. |

**Bottom line:** There is **no pretrained flood-*risk* model to fine-tune** — you must train your own susceptibility baseline (Section 3). Foundation models earn their place **upstream**, turning satellite imagery into the ground-truth flood extent you validate against. Use Prithvi/Sen1Floods11 for labels; train your own XGBoost for risk. Don't let the shiny foundation-model rabbit hole eat your timeline.

---

## 9. The prioritized 6–8 week study + build plan

Learning and building interleaved; each week's reading unblocks that week's build. **Decision to lock in Week 0:** use **Chennai 2015 as the validation city** (Sentinel-1 exists) and Mumbai as a secondary demo — this is the single most schedule-relevant call, because Mumbai 2005 labels are the hardest artifact in the whole project.

**Week 1 — Orientation + data plumbing.**
- Read: Cat-modeling framework chapters of Mitchell-Wallace et al. **[MUST]**; skim Geographic Data Science with Python Parts I–II **[MUST]**; the three flood types (Section 2.2).
- Build: GEE + local Python geospatial env; pull DEM (FABDEM), WorldCover, IMD/IMERG rainfall for Chennai; get everything onto one 30 m grid in UTM. **Deliverable: an aligned feature stack.**

**Week 2 — Labels + terrain features (the hardest data week).**
- Read: Bates (2022) flood review **[MUST]**; Sen1Floods11 paper **[MUST]**.
- Build: derive HAND/slope/drainage from DEM; map **Chennai 2015 flood extent from Sentinel-1** (via Sen1Floods11 approach / a Prithvi fine-tune). **Deliverable: a flood inventory (positive cells) + full predictor stack.**

**Week 3 — Rare-event design + trivial baseline.**
- Read: Barbet-Massin (2012) pseudo-absences **[MUST]**; King & Zeng (2001) rare-event logistic **[MUST]**; Roberts (2017) spatial CV **[MUST]**.
- Build: pseudo-absence sampling strategy; set up **spatial block CV harness** (permanent infrastructure!); implement the **HAND-threshold trivial baseline**. **Deliverable: validation harness + a number to beat.**

**Week 4 — Real baseline models.**
- Read: Mosavi et al. (2018) susceptibility review **[MUST]**; SHAP paper (Lundberg & Lee) **[MUST]**.
- Build: penalized logistic + Random Forest + XGBoost with monotonic constraints & class weights; evaluate all under the same spatial CV; AUC-ROC/PR + calibration curves. **Deliverable: a validated hazard susceptibility map that beats the trivial baseline, with SHAP.**

**Week 5 — Exposure + vulnerability + financial (thin).**
- Read: Huizinga JRC depth-damage report **[MUST]**; Merz (2010) damage review **[MUST]**.
- Build: fuse building footprints + value proxy (exposure); apply borrowed JRC depth-damage curves (vulnerability); compute expected loss / AAL per grid → aggregate to (approximate) pincode. **Deliverable: end-to-end physical→₹ per pincode.**

**Week 6 — Uncertainty + validation artifact.**
- Read: Angelopoulos & Bates conformal prediction **[MUST]**; Ploton (2020) **[MUST]**; Meyer & Pebesma area-of-applicability **[NICE]**.
- Build: confidence bands (conformal/quantile); the **model-vs-actual 2015 overlay** validation view; provenance/uncertainty per layer. **Deliverable: the proof artifact — the thing that sells.**

**Week 7 — Front-end + report.**
- Build: web dashboard (Streamlit for speed, or React if investor-facing — lock this per the MVP open-decisions list); map shaded by pincode score; per-pincode drill-down with drivers/confidence/sources; optional Claude-API narrative report constrained to the structured numbers (LLM must not invent figures). **Deliverable: demoable MVP.**

**Week 8 (buffer / stretch) — Climate layer + Mumbai demo.**
- Read: Milly (2008) **[MUST]**; skim NEX-GDDP-CMIP6 docs.
- Build: light delta-change rainfall scenario toggle (2050, SSP2-4.5/SSP5-8.5) with **wide uncertainty bands**; reconstruct a rough Mumbai demo view (with honest label caveats). **Deliverable: forward-looking toggle + Mumbai story.**

**If you fall behind:** cut Week 8 (climate toggle) and the Mumbai demo first; never cut Weeks 3 and 6 (spatial validation + uncertainty) — those are what make the number *defensible* rather than just *pretty*.

---

## 10. Highest-risk knowledge gaps (the 5 things you'll most likely get wrong)

1. **Spatial data leakage → fake accuracy.** *The* classic newcomer failure. Random train/test splits on autocorrelated grids report inflated metrics; the model then fails in production. **Fix:** spatial block CV from Day 1 (Roberts 2017); always report the trivial-baseline comparison; be suspicious of any AUC > ~0.9.

2. **Pseudo-absence / "where it didn't flood" is arbitrary and dominates your answer.** How you sample non-flooded cells silently determines the whole map — sample negatives only from high ground and you'll "prove" a useless model. **Fix:** deliberate pseudo-absence design (Barbet-Massin 2012), test sensitivity to the choice, document it.

3. **Rainfall resolution mismatch.** Driving street-scale urban flooding with 25 km rainfall grids is a real physical mismatch you can't fully solve. **Fix:** treat rainfall as near-uniform per-event forcing, let terrain/imperviousness create local variation, and *state the limitation openly* rather than implying pincode-scale rainfall precision you don't have.

4. **Over-trusting borrowed depth-damage curves and value proxies → false ₹ precision.** JRC/Hazus curves and footprint×assumed-value proxies were not built for Indian building stock; the ₹ loss numbers carry large, compounding uncertainty. **Fix:** propagate uncertainty into the financial number, present ranges not point estimates, and label every assumption as an assumption (Merz 2010).

5. **Confusing stationarity-broken reality with a stationary model — while it's your USP.** It would be embarrassing to sell "forward-looking" and ship a purely historical model, *and* equally damaging to overclaim precise 2050 numbers that rest on the least-reliable part of climate science (extreme monsoon rainfall downscaling). **Fix:** implement the delta-change rainfall layer so the model is genuinely forward-looking, but present it as *scenario ranges with wide bands* (Milly 2008; honest CMIP6/CORDEX limits), never a single confident future number.

**Meta-gap:** the temptation to reach for deep learning / foundation models as the risk engine because they're impressive. For sparse-label susceptibility with non-expert maintainers, interpretable trees + rigorous validation will be *more* accurate and *far* more defensible. Spend your novelty budget on validation rigor and the climate layer, not on model exotica.

---

### Source-reliability note
Author/year/venue are cited from domain knowledge; before quoting any of these in an investor or insurer deck, **verify the exact title, year, and edition** (especially the JRC report number EUR 28552, and the Angelopoulos & Bates and Roberts et al. details) via the actual publications. The *methods* and *datasets* described are current and standard as of this writing; the *reliability trade-offs* (Fathom = physics, you = statistics; climate downscaling of extreme rainfall is weak) are genuine consensus positions in the field, not editorializing.
