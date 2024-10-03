![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs?ref=badge_shield)

# stochastic-rs

**stochastic-rs** is a Rust library designed for high-performance simulation and analysis of stochastic processes and models. The primary goal is to provide a simple, easy-to-use, and efficient library that caters to a wide range of applications, including quantitative finance, AI training, statistical analysis, and more. This library is under active development, and contributions are welcome. Please note that breaking changes may occur as the library evolves. 🚧

Documentation is available at [stochastic-rs](https://docs.rs/stochastic-rs/).

## Modules

The library is organized into several modules, each targeting specific areas of stochastic processes and their applications.

| Module       | Description                                                                                                                                                                                                                                                                   |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **stochastic** | High-performance data generation for stochastic processes. Optimized for AI training and similar purposes where large amounts of synthetic data are required. It includes efficient algorithms for simulating various stochastic processes with a focus on performance and scalability. |
| **quant**       | Leveraging stochastic models for quantitative finance analysis. Includes tools for modeling financial instruments, pricing derivatives, risk assessment, and other financial computations using stochastic methods. It aims to bridge the gap between theoretical models and practical financial applications.      |
| **stats**       | Focused on statistical analysis related specifically to stochastic processes. While Rust has several excellent statistical libraries, this module provides functions for parameter estimation, calculating fractal dimensions, time-series analysis, and other specialized statistical tools relevant to stochastic processes. |
| **ai**          | Provides out-of-the-box AI and deep neural network (DNN) solutions. Initially developed for PhD research problems, it offers tools and models that can be applied to more general AI topics, facilitating research and development in machine learning and data science.                                                |

---

## Stochastic Processes

The library supports a wide range of stochastic processes, providing both basic and advanced models:

- **Gaussian Noise**: Random noise with a Gaussian (normal) distribution. Useful for simulating white noise in various contexts.
- **Correlated Gaussian Noise**: Gaussian noise with specified correlations between different dimensions or time steps. Essential for modeling multivariate systems where variables are not independent.
- **Brownian Motion**: A continuous-time stochastic process representing random motion. Fundamental to many models in physics and finance, especially in modeling stock prices.
- **Correlated Brownian Motion**: Brownian motion with specified correlations, useful in multi-dimensional modeling where variables influence each other.
- **Geometric Brownian Motion**: Models exponential growth with stochastic volatility, commonly used in financial modeling for asset prices and option pricing.
- **Cox-Ingersoll-Ross (CIR) Process**: Used to model interest rates, ensuring they remain positive. It is mean-reverting and has applications in bond pricing.
- **Ornstein-Uhlenbeck Process**: A mean-reverting process useful in physics (for modeling velocities of particles) and finance (for interest rates and currency exchange rates).
- **Jacobi Process**: A bounded mean-reverting process, taking values within a fixed interval. Useful in modeling proportions and rates that naturally stay within bounds.

---

## Jumps and Lévy Processes (Unstable)

These processes incorporate jumps or discontinuities, adding complexity to the models:

- **Poisson Process**: Counts the number of events occurring in fixed intervals of time or space. Widely used in queueing theory, telecommunications, and reliability engineering.
- **Compound Poisson Process**: Extends the Poisson process by adding random jump sizes. Used in insurance mathematics to model total claims over time.
- **Fractional Ornstein-Uhlenbeck Process with Jumps**: Combines mean-reversion, long memory, and jumps. Useful for modeling phenomena with sudden shifts and memory effects.
- **Lévy Jump Diffusion**: Incorporates both continuous diffusion and discrete jumps. Provides a more accurate model for asset returns by capturing sudden large movements.
- **Inverse Gaussian Distribution**: Used for modeling positive-valued stochastic processes with heavy tails. Applicable in survival analysis and financial modeling.
- **Normal Inverse Gaussian Distribution**: Flexible distribution for modeling asymmetric, heavy-tailed data. Useful in finance for modeling returns that deviate from normality.
- **Variance Gamma Process**: A pure jump process with applications in financial modeling. Captures the leptokurtic features of asset returns.

---

## Stochastic Models

Advanced models built upon stochastic processes for specific applications:

- **Heston Model**: A stochastic volatility model for option pricing, allowing volatility to be random rather than constant. Captures the volatility smile observed in markets.
- **Merton Model**: Incorporates jumps in asset prices, extending the Black-Scholes model. Useful for pricing options in markets with sudden large movements.
- **Bates Model**: Combines the Heston model with jumps from the Merton model. Provides a more comprehensive framework for option pricing.
- **Vasicek Model**: A simple interest rate model featuring mean reversion. Used in the valuation of bonds and interest rate derivatives.
- **SABR Model (Unstable)**: A stochastic volatility model capturing the smile effect in options markets. Useful in modeling the volatility surface.
- **Duffie-Kan Model (Unstable)**: A term structure model for interest rates. Provides a framework for modeling the entire yield curve.
- **Bergomi Model (Unstable)**: Captures volatility clustering and long-range dependence. Used for modeling forward variance and implied volatility surfaces.
- **Rough Bergomi Model (Unstable)**: Incorporates rough volatility for more accurate modeling of financial data. Reflects empirical findings of roughness in volatility.

---

## Fractional Stochastic Processes

Processes that exhibit long-range dependence and self-similarity:

- **Fractional Gaussian Noise**: Generalizes Gaussian noise with a parameter controlling the degree of memory. Useful in modeling processes with persistence or anti-persistence.
- **Correlated Fractional Gaussian Noise**: Fractional noise with correlations, modeling multi-dimensional systems with memory effects.
- **Fractional Brownian Motion**: Extends Brownian motion to include memory effects, characterized by the Hurst exponent. Applicable in finance, hydrology, and telecommunications.
- **Correlated Fractional Brownian Motion**: Multi-dimensional fractional Brownian motion with correlations, capturing both memory and inter-variable dependencies.
- **Fractional Geometric Brownian Motion**: Combines geometric Brownian motion with fractional properties, modeling assets with long memory in volatility.
- **Fractional Ornstein-Uhlenbeck Process**: Mean-reverting process with long memory. Useful in modeling interest rates and volatility with persistence.
- **Fractional Cox-Ingersoll-Ross Process**: Fractional extension of the CIR process, incorporating memory effects in interest rate modeling.
- **Fractional Jacobi Process**: Fractional version of the Jacobi process, useful in bounded processes with memory.

---

## Features

Planned features and models under development:

- **Rough Heston Model**: Combines the Heston model with rough volatility, capturing the fine structure of volatility movements.
- **Hull-White Model**: An interest rate model with time-dependent parameters, providing a good fit to initial term structures.
- **Barndorff-Nielsen & Shephard Model**: A stochastic volatility model with non-Gaussian Ornstein-Uhlenbeck processes.
- **Alpha-Stable Models**: Models based on stable distributions, capturing heavy tails and skewness.
- **CGMY Model**: A pure jump Lévy process model capturing infinite activity and infinite variation.
- **Multi-factor CIR Model**: Extends the CIR model to multiple factors, providing greater flexibility in interest rate modeling.
- **Brace-Gatarek-Musiela (BGM) Model**: A market model for interest rates, modeling the evolution of forward rates.
- **Wu-Zhang Model**: A stochastic volatility model with jumps in returns and volatility.
- **Affine Models**: A class of models where yields are affine functions of factors, facilitating analytical solutions.
- **Heath-Jarrow-Morton Model & Multi-factor HJM Model**: Models the evolution of the entire yield curve over time.

---

## Future Work

We aim to continuously improve and expand the library:

- **More Tests**: Increase test coverage to ensure reliability and correctness. Rigorous testing of numerical methods and edge cases.
- **More Examples**: Provide comprehensive examples and tutorials. Demonstrate practical applications and integrations.
- **Full Documentation**: Enhance documentation for all modules and functions. Include mathematical backgrounds, usage instructions, and API references.
- **Performance Optimization**: Further optimize algorithms for speed and memory efficiency.
- **Cross-platform Support**: Ensure compatibility across different operating systems and environments.
- **User Feedback Integration**: Incorporate suggestions and feedback from the user community to improve usability and features.

---

## Getting Started

To include **stochastic-rs** in your project, add the following to your `Cargo.toml`:

```toml
[dependencies]
stochastic-rs = "0.x.0"
```

Replace `0.x.0` with the latest version available on [Crates.io](https://crates.io/crates/stochastic-rs).

### Installation

Ensure you have Rust and Cargo installed. For installation instructions, visit [rust-lang.org](https://www.rust-lang.org/tools/install).


## Contributing

Welcome contributions from the community! Whether it's reporting bugs, suggesting new features, or improving documentation, your help is appreciated.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs?ref=badge_large)

## Contact

For any questions, suggestions, or discussions, feel free to open an issue or start a discussion on GitHub. You can also reach out via email at [dancixx@gmail.com].

---

## Acknowledgments

I would like to thank all the contributors and the open-source community for their invaluable support.

---

**Note**: This package is currently in a very early development phase. Breaking changes may occur as we work towards a stable release. Your feedback and contributions are crucial to help us improve and reach version 1.0.

---

Feel free to ⭐ star the repository if you find it useful!