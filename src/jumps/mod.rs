//! This module contains the implementations of various diffusion processes.
//!
//! The following diffusion processes are implemented:
//!
//! - **Bates (1996) Model**
//!   - Combines stochastic volatility with jump diffusion.
//!   - SDE: `dX(t) = mu * S(t) * dt + S(t) * sqrt(V(t)) * dW(t) + Jumps`
//!
//! - **Inverse Gaussian (IG) Process**
//!   - Models heavy-tailed distributions.
//!   - SDE: `dX(t) = gamma * dt + dW(t)`
//!
//! - **Normal Inverse Gaussian (NIG) Process**
//!   - Models stock returns with normal and inverse Gaussian components.
//!   - SDE: `dX(t) = theta * IG(t) + sigma * sqrt(IG(t)) * dW(t)`
//!
//! - **Lévy Diffusion Process**
//!   - Incorporates Gaussian and jump components.
//!   - SDE: `dX(t) = gamma * dt + sigma * dW(t) * Jump`
//!
//! - **Merton Jump Diffusion Process**
//!   - Combines continuous diffusion with jumps.
//!   - SDE: `dX(t) = (alpha * sigma^2 / 2 - lambda * theta) * dt + sigma * dW(t) + Jumps`
//!
//! - **Variance Gamma (VG) Process**
//!   - Captures excess kurtosis and skewness in asset returns.
//!   - SDE: `dX(t) = mu * Gamma(t) + sigma * sqrt(Gamma(t)) * dW(t)`
//!
//! Each process has its own module and functions to generate sample paths.

pub mod bates;
pub mod ig;
pub mod levy_diffusion;
pub mod merton;
pub mod vg;
