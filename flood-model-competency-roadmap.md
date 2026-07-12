# PinRisk — Flood-Model Competency Roadmap

**Audience:** the founding team — CS majors with coursework-level ML, no catastrophe/climate domain background.
**Goal:** the most accurate defensible baseline urban-flood model for the Mumbai/Chennai MVP, plus the self-study path to build and maintain it.
**Tagging:** every resource is marked **[MUST-READ]** or **[NICE-TO-HAVE]**. Must-reads total roughly 2–3 weeks of part-time reading; everything else is reference material you consult when you hit the relevant problem.

One framing note before the sections: what you are building for the MVP is a **flood susceptibility model + single-event backtest + simplified loss layer**. That is not yet a full catastrophe model (no stochastic event set, no exceedance-probability curves from thousands of simulated years). That's fine — but know the difference, because insurers will, and Section 6 explains it.

---

## 1. ML learning path (targeted)

You already know models/training/evaluation. What you *don't* know — and what silently breaks flood models — is **spatial data handling, spatial validation, rare-event labeling, and uncertainty**. In priority order:

### 1a. Geospatial data handling (blocks everything — week 1)
- **[MUST-READ]** *Geographic Data Science with Python* — Rey, Arribas-Bel & Wolf (free online, geographicdata.science). Read Parts 1–2 (data structures, choropleths, spatial weights). This is your working vocabulary: rasters vs. vectors, CRS/projections, resampling, zonal statistics. Every bug you'll have in week 1 is a CRS or resampling bug.
- **[MUST-READ]** *Cloud-Based Remote Sensing with Google Earth Engine* — Cardille, Crowley, Saah, Clinton (free Springer book, 2023). Read the fundamentals + the change-detection/SAR chapters. GEE is how you'll pull Sentinel-1, WorldCover, rainfall, and DEMs without downloading terabytes.
- **[NICE-TO-HAVE]** *Spatial Data Science: With Applications in R* — Pebesma & Bivand (free online). Deeper theory (support, change-of-support, areal aggregation). Skim the concepts even though you'll work in Python — "change of support" is literally your grid→pincode aggregation problem.

### 1b. Spatial validation & spatial statistics (the #1 way newcomers fool themselves)
- **[MUST-READ]** Roberts et al. 2017, *"Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"*, Ecography. The canonical paper on why random train/test splits on spatial data massively inflate accuracy (neighboring cells are near-duplicates). You must use **spatial block cross-validation**. This paper is why.
- **[MUST-READ]** Ploton et al. 2020, *"Spatial validation reveals poor predictive performance of large-scale ecological mapping models"*, Nature Communications. Short, brutal demonstration of the same failure in published work. Read it to internalize the fear.
- **[NICE-TO-HAVE]** Meyer & Pebesma 2021, *"Predicting into unknown space? Estimating the area of applicability of spatial prediction models"*, Methods in Ecology & Evolution. Matters when you later transfer the Mumbai/Chennai model to a new city — tells you where the model is extrapolating.
- **[NICE-TO-HAVE]** Valavi et al. 2019, *"blockCV: an R package for generating spatially or environmentally separated folds"*, MEE — read for the methodology even if you reimplement in Python (scikit-learn `GroupKFold` over spatial blocks gets you 90% there).

### 1c. Rare-event / presence-background modeling (floods are your minority class)
The trick: this problem is nearly isomorphic to **species distribution modeling (SDM)** in ecology — rare presences, abundant background, spatial predictors. That literature is 20 years ahead of the flood-susceptibility literature on methodology. Steal from it.
- **[MUST-READ]** Elith et al. 2008, *"A working guide to boosted regression trees"*, Journal of Animal Ecology. The single best practical guide to gradient boosting on ecological/spatial presence data — learning rate, tree depth, and interpretation. Directly transferable to flood susceptibility.
- **[MUST-READ]** Barbet-Massin et al. 2012, *"Selecting pseudo-absences for species distribution models: how,