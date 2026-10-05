# Objective  
Classify SEP events using contrastive learning (>10 Mev and > 10 PFU)  

# Data  
Comes from OMNI and GOES (Geostationary Operational Environmental Satellites)  

| Abbr | Variable | Description | Source | Units |
| --- | --- | --- | --- | --- |
| $V_x$ | X-axis Solar-Wind Velocity | Velocity component along GSE x-axis, representing radial plasma flow toward Earth | OMNI | $km/s^{-1}$ |
| $V$  | Flow Speed | Total solar-wind flow speed magnitude, derived from vector components (\|V\|) | OMNI | $km/s^{-1}$ |
| $N_p$ | Proton Number Density | ... of the solar-wind plasma | OMNI | $cm^{-3}$ |
| $F$  | Interplanetary Magnetic Field Magnitude | Represents total IMF strength near Earth (\|B\|) | OMNI | $nT$ |
| $T_p$ | Proton Temperature | ... of the solar-wind plasma | OMNI | $K$ |
| $X_s$ | Soft X-ray Flux | X-ray flux measured in the 0.1 - 0.8 nm channel (short wavelength) | GOES | $W m^{-2}$ |
| $X_l$ | Hard X-ray Flux | X-ray flux measured in the 0.05 - 0.4 nm channel (long wavelength) | GOES | $W m^{-2}$  |
| $P4$ | Proton Flux >30 MeV | Recorded by the GOES Energetic Particle Sensor (EPS/HEPAD) | GOES | $pfu$ |
| $P5$ | Proton Flux >50 MeV | Recorded by the GOES Energetic Particle Sensor (EPS/HEPAD) | GOES | $pfu$ |
| $P6$ | Proton Flux >60 MeV | Recorded by the GOES Energetic Particle Sensor (EPS/HEPAD) | GOES | $pfu$ |  

# Misc  
- Highly unbalanced dataset (~ 1% SEP events)  
- No known correlation between SF and SEP events  

# Readings  
- _Contrastive Representation Learning: A Framework and Review_ __[Le-Khac]__
- _Contrastive Representation Learning for Predicting Solar Flares from Extremely Imbalanced Multivariate Time Series Data_ __[Vural]__
- _EXCON: Extreme Instance-based Contrastive Representation Learning of Severely Imbalanced Multivariate Time Series for Solar Flare Prediction_ __[Vural]__
- _Improving Solar Energetic Particle Event Prediction through Multivariate Time Series Data Augmentation_ __[Hosseinzadeh]__  
- _An End-to-end Ensemble Machine Learning Approach for Predicting High-impact Solar Energetic Particle Events Using Multimodal Data_ __[Hosseinzadeh]__  
- _A Multi-Instrument Time-Series Dataset for Pre-Flare Solar Activity and Solar Energetic Particle Event Prediction_ __[Hosseinzadeh]__  
- _Review of Solar Energetic Particle Prediction Models_ __[Whitman]__