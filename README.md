# neural-decoding-VAR
# An Analysis of the Temporal Component of Motor Preparation and Execution in High-Frequency Local Field Potentials: A Theoretical Approach

## Authors
- **Marc Burillo**  
  1. Sorbonne Université, France  
  2. Departament de Matemàtiques i Informàtica, Universitat de Barcelona, Barcelona, Spain  


- **Michael DePass**  
  1. Departament de Matemàtiques i Informàtica, Universitat de Barcelona, Barcelona, Spain  
  3. Universitat Pompeu Fabra, Barcelona, Spain  

- **Ignasi Cos**  
  1. Departament de Matemàtiques i Informàtica, Universitat de Barcelona, Barcelona, Spain  
  4. Institutes of Neuroscience & Mathematics, Universitat de Barcelona, Spain  

---

## Abstract
Electrophysiological recordings have been the fundamental source of brain inner information crucially contributing to the current understanding of brain function and dynamics. Typically recorded in the context of controlled laboratory tasks, state-of-the-art multi-electrode arrays can nowadays provide simultaneously recorded high dimensional time series from across several brain areas, providing an unprecedented insight into brain dynamics. However, as their richness and complexity increase, obtaining reliable methods yielding interpretable characterizations of the underlying neural substrate remains a matter of vivid interest. In so far, most neural time-series analyses are typically reduced to pairwise electrode statistics, such as cross-correlation and Granger causality. Furthermore, it is most often the case that the golden-data neural datasets recorded during specific tasks encompass a few sessions alone, questioning the use of data-voracious techniques. For example, deep learning neural networks come at the expense of reduced interpretability and the requirement of prohibitively large datasets. Our purpose here is to provide exploratory techniques aimed at providing rich statistical characterizations of spatiotemporal brain dynamics within the constraint of modest, multivariate dataset time-series recordings. In brief, we describe the use of a vector autoregressive-machine learning pipeline to characterize spatio-temporal dynamics of high-frequency local field potentials recorded during a movement planning and execution task by means of Utah arrays implanted in the motor areas of a non-human primate. By contrast to basic cross-correlations and granger-causality analyses, our pipeline provides a principled analysis of multivariate time series while preserving dataset and results interpretability. Importantly, the classification accuracies from single-electrode analysis suggest *high degree of intra-region heterogeneity*, while multi-electrode achieved the highest accuracies confirming a network behaviour and the suitability of a more complex description than simply paired time series analysis. In summary, this technique provides a reliable method to characterize the multivariate spatiotemporal neuro-dynamics of motor-related brain states using a single session dataset, while preserving high accuracy and interpretability.

---

## Repository Overview
### Author of the Code
- **Marc Burillo**

### License
Please refer to the `LICENSE` file for details on usage restrictions and permissions.

### Summary
This repository provides software designed to characterize the temporal and connectivity dynamics of Local Field Potential (LFP) recordings from non-human primates during a motor and cognitive experiment.  

The main functionalities include:
1. **Vectorial Autoregressive (VAR) Model Fitting:**  
   Implements a novel multi-trial fitting approach to analyze temporal dynamics.
2. **Multinomial Logistic Regression (MLR) Pipeline:**  
   Assesses the characterization of connectivity and temporal features.
3. **Minimal Feature Characterization:**  
   Performed through Recursive Feature Elimination (RFE) to identify key contributors.

Additionally, we have included the Mathematics Bachelor's Dissertation (`Mathematical, Computer Science and Neuroscience Background`) of Marc Burillo, which only covers the very first exploratory stage of this project. Although the theoretical concepts explained have not changed (Chapters 2 and 3), the actual pipeline and analysis to be published are very different. Therefore, we recommend reading the document (also not to be shared, see `Disclaimer`), as an aid for understanding the code.

### Disclaimer
This project is going to be published very soon. For obvious reasons, all content is extremely confidential and shared only for the PhD Application process of CCMI CDT. No database is provided to test the code, nor all the analysis done with this method is shown. This is only the backbone of a complex method used study fundamental questions in brain coding in terms of temporal and connectivity dynamics.
---

## Getting Started
### Prerequisites
- Python >= 3.x
- Required Python libraries (listed in `requirements.txt`)
- Jupyter notebook for the analysis of the pipeline.
### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/M-Burillo/neural-decoding-VAR
   cd M-Burillo/neural-decoding-VAR
### Structure
In our project, experiment recordings were in MATLAB format. To convert this into Python-tractable data and extract the High-Frequency component:
1. `Preprocessing.py`: extracts the high-frequency component (200-500Hz) with a Butterworth bandpass filter. Additionally, it prepares the Cross-Validation method used throughout the rest of the project. We decided for reasons explained in the `Mathematical, Computer Science and Neuroscience Background` to use a 10-Fold CV, repeating it 10 times, achieving 300 repetitions of the training and testing process

One of the most important components of this project was to decide how to approach a massive dataset of 264 channel recordings with the Vector AutoRegresive method, which allows combining 1 to N electrodes. We were guided by the previous results of DePass et al., using a score based on the Spectral Amplitude of a time series. For the reasons explained, this is not shared. However, any user can set a simple document as channels.txt where each line represents a model to be explored using those channels. In our project, we have studied mainly models with 1 and 3 channels, but we have explored up to 5, where the number of trials constraints adding more channels.
With that document, the entire pipeline can be run sequentially:
2. `order_selection.py`: determines the optimal order (p) to use in a VAR(p) model. It requires determining the channels considered in each model the CV method used and the processed time series

3. `multitrialVARp.py`: implements the multi-trial fitting of a VAR(p) model. It requires determining the channels considered in each model and the order of the model, the CV method used and the processed time series

4. `MLRassessmentVAR.py`: implements the multi-logistic regression of `sklearn` to assess if the VAR(p) characterization matches the 5 labelled states. This code requires to run before multitrialVARp.py. The output is the accuracy in the trainset (called precision) and in the test. Also the Confusion Matrices for each subset. The CV method is 10-Fold

5. `MREoptimalVAR.py`: implements the Recursive Feature Elimination (RFE) algorithm on the VAR(p) characterization matches. This code requires to run before MLRassessmentVARp.py. The output is the optimal features over a training process with the CV method is 10-Fold, when using RFE for a smaller subset of features. It also reports the accuracy and precision distributions. The RFE method is implemented recursively: in each individual training process, only the previous surviving features are re-analysed

6. `AnalysisVARp.ipynb`: Jupyter Notebook showing the performance and the characterization (including the minimal characterization) analysis of the method. No input file of the Jupyter Notebook has been provided, so it can not be run, it is only for illustrative purposes. 


