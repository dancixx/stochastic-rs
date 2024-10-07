//! # Stochastic Processes in Rust
//!
//! `stochastic-rs` is a Rust library designed for simulating and analyzing various types of stochastic processes.
//! It offers an extensive set of tools for researchers, financial engineers, and developers interested in modeling random phenomena in diverse fields.
//!
//! ## Getting Started
//!
//! To start using the library, add `stochastic-rs` to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! stochastic-rs = "*"
//! ```
//!
//! ## Modules
//!
//! | Module         | Description                                                                                                                                                                                                                                                            |
//! |----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | **stochastic**  | High-performance data generation for stochastic processes. Optimized for AI training and similar purposes where large amounts of synthetic data are required. It includes efficient algorithms for simulating various stochastic processes with a focus on performance and scalability. |
//! | **quant**       | Leveraging stochastic models for quantitative finance analysis. Includes tools for modeling financial instruments, pricing derivatives, risk assessment, and other financial computations using stochastic methods. It aims to bridge the gap between theoretical models and practical financial applications.       |
//! | **stats**       | Focused on statistical analysis related specifically to stochastic processes. While Rust has several excellent statistical libraries, this module provides functions for parameter estimation, calculating fractal dimensions, time-series analysis, and other specialized statistical tools relevant to stochastic processes.  |
//! | **ai**          | Provides out-of-the-box AI and deep neural network (DNN) solutions. Initially developed for PhD research problems, it offers tools and models that can be applied to more general AI topics, facilitating research and development in machine learning and data science.        |
//!
//! ## Performance Optimization
//!
//! For performance reasons, the default allocator used in `stochastic-rs` is `jemalloc`, which is known for its efficient memory allocation in high-performance scenarios.  
//! If you prefer to use the system allocator, simply disable the default features in your `Cargo.toml` like this:
//! ```toml
//! [dependencies]
//! stochastic-rs = { version = "*", default-features = false }
//! ```
//!
//! ## License
//!
//! This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.
//!
//! ## Contributions and Acknowledgments
//!
//! Contributions are welcome! Feel free to open issues or submit pull requests on [GitHub](https://github.com/dancixx/stochastic-rs).
//! - Developed and maintained by [dancixx](https://github.com/dancixx).
//!
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![warn(missing_docs)]

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod ai;
#[doc(hidden)]
mod macros;
pub mod quant;
pub mod stats;
pub mod stochastic;
