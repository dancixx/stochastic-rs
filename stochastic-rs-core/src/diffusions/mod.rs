//! This module contains the implementations of various diffusion processes.
//!
//! The following diffusion processes are implemented:
//!
//! - **Cox-Ingersoll-Ross (CIR)**
//!   - SDE: `dX(t) = theta(mu - X(t))dt + sigma sqrt(X(t))dW(t)`
//!
//! - **Fractional Cox-Ingersoll-Ross (fCIR)**
//!   - SDE: `dX(t) = theta(mu - X(t))dt + sigma sqrt(X(t))dW^H(t)`
//!
//! - **Geometric Brownian Motion (GBM)**
//!   - SDE: `dX(t) = mu X(t)dt + sigma X(t)dW(t)`
//!
//! - **Fractional Geometric Brownian Motion (fGBM)**
//!   - SDE: `dX(t) = mu X(t)dt + sigma X(t)dW^H(t)`
//!
//! - **Jacobi Process**
//!   - SDE: `dX(t) = (alpha - beta X(t))dt + sigma (X(t)(1 - X(t)))^(1/2)dW(t)`
//!
//! - **Fractional Jacobi Process**
//!   - SDE: `dX(t) = (alpha - beta X(t))dt + sigma (X(t)(1 - X(t)))^(1/2)dW^H(t)`
//!
//! - **Ornstein-Uhlenbeck (OU)**
//!   - SDE: `dX(t) = theta(mu - X(t))dt + sigma dW(t)`
//!
//! - **Fractional Ornstein-Uhlenbeck (fOU)**
//!   - SDE: `dX(t) = theta(mu - X(t))dt + sigma dW^H(t)`
//!
//! Each process has its own module and functions to generate sample paths.

pub mod cir;
pub mod fcir;
pub mod fgbm;
pub mod fjacobi;
pub mod fou;
pub mod gbm;
pub mod jacobi;
pub mod ou;
