![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/dancixx/stochastic-rs/graph/badge.svg?token=SCSp3z7BQJ)](https://codecov.io/gh/dancixx/stochastic-rs)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fdancixx%2Fstochastic-rs?ref=badge_shield)

# stochastic-rs

**stochastic-rs** is a Rust library designed for high-performance simulation and analysis of stochastic processes and models. The primary goal is to provide a simple, easy-to-use, and efficient library that caters to a wide range of applications, including quantitative finance, AI training, statistical analysis, and more. This library is under active development, and contributions are welcome. Please note that breaking changes may occur as the library evolves. 🚧

[RustQuant](https://github.com/avhz/RustQuant): you might want to explore **RustQuant**, another excellent Rust package tailored for quantitative finance.
## Modules

The library is organized into several modules, each targeting specific areas of stochastic processes and their applications.

| Module       | Description                                                                                                                                                                                                                                                                   |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **stochastic** | High-performance data generation for stochastic processes. Optimized for AI training and similar purposes where large amounts of synthetic data are required. It includes efficient algorithms for simulating various stochastic processes with a focus on performance and scalability. |
| **quant**       | Leveraging stochastic models for quantitative finance analysis. Includes tools for modeling financial instruments, pricing derivatives, risk assessment, and other financial computations using stochastic methods. It aims to bridge the gap between theoretical models and practical financial applications.      |
| **stats**       | Focused on statistical analysis related specifically to stochastic processes. While Rust has several excellent statistical libraries, this module provides functions for parameter estimation, calculating fractal dimensions, time-series analysis, and other specialized statistical tools relevant to stochastic processes. |
| **ai**          | Provides out-of-the-box AI and deep neural network (DNN) solutions. Initially developed for PhD research problems, it offers tools and models that can be applied to more general AI topics, facilitating research and development in machine learning and data science.                                                |

---

## Features

Planned features and models under development:
- **Barndorff-Nielsen & Shephard Model**: A stochastic volatility model with non-Gaussian Ornstein-Uhlenbeck processes.
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


## Contact

For any questions, suggestions, or discussions, feel free to open an issue or start a discussion on GitHub. You can also reach out via email at [dancixx@gmail.com].

---

## Acknowledgments

I would like to thank all the contributors and the open-source community for their invaluable support.

---

**Note**: This package is currently in a very early development phase. Breaking changes may occur as we work towards a stable release. Your feedback and contributions are crucial to help us improve and reach version 1.0.

---

Feel free to ⭐ star the repository if you find it useful!
