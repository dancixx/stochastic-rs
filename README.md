[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![Crates.io](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)

# Stochastic-rs

A Rust library for stochastic processes.


# Implementations

- [Rust](https://github.com/dancixx/stochastic-rs)
- [Typescript](https://github.com/dancixx/stochastic-js)


# Features
- [x] Gaussian noise
- [x] Fractional Gaussian noise
- [x] Brownian motion
- [x] Geometric Brownian motion
- [x] Ornstein-Uhlenbeck process
- [x] Cox-Ingersoll-Ross process
- [x] Jacobi process
- [x] Fractional Brownian motion
- [x] Fractional Geometric Brownian motion
- [x] Fractional Ornstein-Uhlenbeck process
- [x] Fractional Cox-Ingersoll-Ross process
- [x] Fractional Jacobi process
- [ ] Heston model
- [ ] Merton model
- [ ] Jump-diffusion model
- [ ] Variance Gamma model
- [ ] Normal Inverse Gaussian model

# Usage

**Gaussian noise**

```rust
use stochastic_rs::noises::gn;

let n = 1000;
let noise = gaussian(n);
println!("{:?}", noise);
```


**Fractional Gaussian noise**
```rust
use stochastic_rs::noises::fgn;

let n = 1000;
let hurst = 0.7;
let noise = fgn(n, hurst);
println!("{:?}", noise);
```

**Brownian motion**
```rust
use stochastic_rs::processes::bm;

let n = 1000;
let t = 1;
let bm = bm(n, t);
println!("{:?}", bm);
```

**Fractional Brownian motion**
```rust
// using cholesky decomposition
use stochastic_rs::processes::fbm_cholesky;

let n = 1000;
let hurst = 0.7;
let t = 1;
let fbm = fbm_cholesky(n, hurst, t);
println!("{:?}", fbm);
```
