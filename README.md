![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)
![Crates.io](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)

# Stochastic-rs

A Rust library for stochastic processes


**Features**
- [x] Gaussian noise
- [x] Fractional Gaussian noise
- [x] Brownian motion
- [x] Fractional Brownian motion
- [ ] Geometric Brownian motion
- [ ] Ornstein-Uhlenbeck process
- [ ] Fractional Ornstein-Uhlenbeck process
- [ ] Cox-Ingersoll-Ross process
- [ ] Vasicek model
- [ ] CIR model
- [ ] Heston model
- [ ] Merton model
- [ ] Jump-diffusion model
- [ ] Variance Gamma model
- [ ] Normal Inverse Gaussian model

**Usage**

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