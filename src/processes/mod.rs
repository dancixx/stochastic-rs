//! This module contains the implementations of various stochastic processes.
//!
//! The following stochastic processes are implemented:
//!
//! - **Brownian Motion (BM)**
//!   - Standard Brownian motion, also known as Wiener process.
//!   - SDE: `dX(t) = dW(t)`
//!
//! - **Correlated Brownian Motions (Correlated BMs)**
//!   - Generates two correlated Brownian motion paths.
//!
//! - **Fractional Brownian Motion (fBM)**
//!   - Generates fractional Brownian motion, which includes long-range dependence.
//!   - SDE: `dX(t) = dW^H(t)` where `H` is the Hurst parameter.
//!
//! - **Poisson Process**
//!   - Models the occurrence of events over time.
//!   - SDE: `dN(t) ~ Poisson(\lambda t)`
//!
//! - **Compound Poisson Process**
//!   - Models the occurrence of events over time, where each event has a random magnitude (jump).
//!   - SDE: `dX(t) = \sum_{i=1}^{N(t)} Z_i` where `N(t)` is a Poisson process with rate `\lambda` and `Z_i` are i.i.d. jump sizes.

pub mod bm;
pub mod cbms;
pub mod cfbms;
pub mod cpoisson;
pub mod fbm;
pub mod poisson;
