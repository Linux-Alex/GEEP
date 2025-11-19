# 🏗️ Concrete Compressive Strength Dataset

> *Predicting the fundamental strength of civil engineering's most vital material*

[![UCI Dataset](https://img.shields.io/badge/UCI-Machine%20Learning-blue)](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
[![DOI](https://img.shields.io/badge/DOI-10.24432/C5PK67-green)](https://doi.org/10.24432/C5PK67)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 📊 Dataset Overview

| Property           | Value                 |
|--------------------|-----------------------|
| **Subject Area**   | Physics and Chemistry |
| **Task Type**      | Regression            |
| **Instances**      | 1,030                 |
| **Features**       | 8 input variables     |
| **Target**         | 1 output variable     |
| **Missing Values** | None                  |

## 🎯 Problem Statement

Concrete compressive strength is a **highly nonlinear function** of age and ingredients. This dataset enables the development of predictive models to estimate concrete strength based on its composition and age, which is crucial for:

- Structural design and safety
- Quality control in construction
- Material optimization
- Cost reduction in concrete production

## 🧪 Features Description

### Component Features (kg/m³ mixture)

| Feature                | Type       | Description                                     | Range                |
|------------------------|------------|-------------------------------------------------|----------------------|
| **Cement**             | Continuous | Primary binding material                        | 102.0 - 540.0 kg/m³  |
| **Blast Furnace Slag** | Integer    | Industrial by-product used as cement substitute | 0.0 - 359.4 kg/m³    |
| **Fly Ash**            | Continuous | Pozzolanic material from coal combustion        | 0.0 - 200.1 kg/m³    |
| **Water**              | Continuous | Hydration and workability agent                 | 121.8 - 247.0 kg/m³  |
| **Superplasticizer**   | Continuous | High-range water reducer                        | 0.0 - 32.2 kg/m³     |
| **Coarse Aggregate**   | Continuous | Gravel or crushed stone (>4.75mm)               | 801.0 - 1145.0 kg/m³ |
| **Fine Aggregate**     | Continuous | Sand or crushed stone (<4.75mm)                 | 594.0 - 992.6 kg/m³  |

### Temporal Feature

| Feature | Type    | Description         | Range        |
|---------|---------|---------------------|--------------|
| **Age** | Integer | Curing time in days | 1 - 365 days |

### Target Variable

| Variable                          | Type       | Description                         | Range            |
|-----------------------------------|------------|-------------------------------------|------------------|
| **Concrete Compressive Strength** | Continuous | Ultimate strength under compression | 2.33 - 82.60 MPa |

## 🔬 Key Relationships

### Critical Ratios
- **Water-Cement Ratio**: Fundamental parameter affecting strength
- **Aggregate-Cement Ratio**: Influences workability and density
- **Supplementary Materials**: Slag and fly ash affect long-term strength gain

### Strength Development
- **Early-age strength** (1-7 days): Rapid strength gain
- **Standard strength** (28 days): Industry benchmark
- **Long-term strength** (90-365 days): Continued hydration effects

## 🙏 Special Thanks

This dataset plays a crucial role in the development and validation of the **GEEP (Generalized Engineering Evaluation and Prediction) Framework**. The concrete compressive strength data provides:

- **Benchmark validation** for regression algorithms
- **Real-world engineering context** for framework testing
- **Multivariate analysis** capabilities with 8 input features
- **Proven industrial relevance** in civil engineering applications

The comprehensive nature of this dataset, with its complete feature set and absence of missing values, makes it ideal for demonstrating GEEP's capabilities in:

- **Automated feature engineering**
- **Model performance benchmarking**
- **Engineering domain adaptation**
- **Predictive analytics validation**

## 📚 Complete Citation

```bibtex
@misc{concrete_compressive_strength_165,
  author       = {Yeh, I-Cheng},
  title        = {{Concrete Compressive Strength}},
  year         = {1998},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5PK67}
}