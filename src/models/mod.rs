//! This module contains the implementations of various diffusion processes.
//!
//! The following diffusion processes are implemented:
//!
//! - **Heston Model**
//!   - A stochastic volatility model used to describe the evolution of the volatility of an underlying asset.
//!   - SDE: `dS(t) = mu * S(t) * dt + S(t) * sqrt(V(t)) * dW_1(t)`
//!   - SDE: `dV(t) = kappa * (theta - V(t)) * dt + eta * sqrt(V(t)) * dW_2(t)`
//!
//! - **Vasicek Model**
//!   - An Ornstein-Uhlenbeck process used to model interest rates.
//!   - SDE: `dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t)`
//!
//! - **Fractional Vasicek (fVasicek) Model**
//!   - Incorporates fractional Brownian motion into the Vasicek model.
//!   - SDE: `dX(t) = theta * (mu - X(t)) * dt + sigma * dW^H(t)`
//!
//! Each process has its own module and functions to generate sample paths.

pub mod heston;
pub mod vasicek;
