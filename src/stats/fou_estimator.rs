use impl_new_derive::ImplNew;
use ndarray::{array, Array1};
use statrs::function::gamma::gamma;
use std::f64::consts::SQRT_2;

#[derive(ImplNew)]
pub struct FOUParameterEstimation {
  pub path: Array1<f64>,
  pub T: f64,
  pub filter_type: FilterType,
  // Estimated parameters
  hurst: Option<f64>,
  sigma: Option<f64>,
  mu: Option<f64>,
  theta: Option<f64>,
  // Filter coefficients
  a: Array1<f64>,
  L: usize,
  V1: f64,
  V2: f64,
}

#[derive(PartialEq)]
pub enum FilterType {
  Daubechies,
  Classical,
}

impl FOUParameterEstimation {
  pub fn estimate_parameters(&mut self) -> (f64, f64, f64, f64) {
    self.linear_filter();
    self.hurst_estimator();
    self.sigma_estimator();
    self.mu_estimator();
    self.theta_estimator();

    (
      self.hurst.unwrap(),
      self.sigma.unwrap(),
      self.mu.unwrap(),
      self.theta.unwrap(),
    )
  }

  fn hurst_estimator(&mut self) {
    let hurst = 0.5 * ((self.V2 / self.V1).log2());
    self.hurst = Some(hurst);
  }

  fn sigma_estimator(&mut self) {
    let hurst = self.hurst.unwrap();
    let V1 = self.V1;
    let a = &self.a;
    let L = self.L;

    let series_length = self.path.len();
    let delta = 1.0 / series_length as f64;

    let mut const_filter = 0.0;

    for i in 0..L {
      for j in 0..L {
        const_filter += a[i] * a[j] * ((i as f64 - j as f64).abs()).powf(2.0 * hurst);
      }
    }

    let numerator = -2.0 * V1 / ((series_length - L) as f64);
    let denominator = const_filter * delta.powf(2.0 * hurst);

    let sigma_squared = numerator / denominator;
    let sigma = sigma_squared.sqrt();
    self.sigma = Some(sigma);
  }

  fn mu_estimator(&mut self) {
    let mean = self.path.mean().unwrap();
    self.mu = Some(mean);
  }

  fn theta_estimator(&mut self) {
    let mean_square = self.path.mapv(|x| x.powi(2)).mean().unwrap();
    let sigma = self.sigma.unwrap();
    let hurst = self.hurst.unwrap();

    let numerator = 2.0 * mean_square;
    let denominator = sigma.powi(2) * gamma(2.0 * hurst + 1.0);
    let theta = (numerator / denominator).powf(-1.0 / (2.0 * hurst));

    self.theta = Some(theta / self.T);
  }

  fn linear_filter(&mut self) {
    let (a, L) = self.get_filter_coefficients();
    self.a = a.clone();
    self.L = L;

    let a_2 = self.get_a2_coefficients(&a);

    let V1_path = self.lfilter(&self.a, &array![1.0], &self.path);
    self.V1 = V1_path.mapv(|x| x.powi(2)).sum();

    let V2_path = self.lfilter(&a_2, &array![1.0], &self.path);
    self.V2 = V2_path.mapv(|x| x.powi(2)).sum();
  }

  fn get_filter_coefficients(&self) -> (Array1<f64>, usize) {
    let a: Array1<f64>;
    let L: usize;
    if self.filter_type == FilterType::Daubechies {
      a = array![
        0.482962913144534 / SQRT_2,
        -0.836516303737808 / SQRT_2,
        0.224143868042013 / SQRT_2,
        0.12940952255126 / SQRT_2
      ];
      L = a.len();
    } else if self.filter_type == FilterType::Classical {
      unimplemented!("Classical filter not implemented yet.");
    } else {
      a = array![
        0.482962913144534 / SQRT_2,
        -0.836516303737808 / SQRT_2,
        0.224143868042013 / SQRT_2,
        0.12940952255126 / SQRT_2
      ];
      L = a.len();
    }
    (a, L)
  }

  fn get_a2_coefficients(&self, a: &Array1<f64>) -> Array1<f64> {
    // Inserting zeros between the coefficients
    let mut a_2 = Array1::<f64>::zeros(a.len() * 2);
    for (i, &val) in a.iter().enumerate() {
      a_2[i * 2 + 1] = val;
    }
    a_2
  }

  fn lfilter(&self, b: &Array1<f64>, a: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    // Implementing the difference equation: y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - ...
    let n = x.len();
    let mut y = Array1::<f64>::zeros(n);

    for i in 0..n {
      let mut acc = 0.0;
      for j in 0..b.len() {
        if i >= j {
          acc += b[j] * x[i - j];
        }
      }
      for j in 1..a.len() {
        if i >= j {
          acc -= a[j] * y[i - j];
        }
      }
      y[i] = acc;
    }

    y
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::stochastic::{diffusion::fou::FOU, noise::fgn::FGN, Sampling};

  #[test]
  fn test_fou_parameter_estimation() {
    const N: usize = 10000;
    const X0: f64 = 0.0;

    let fgn = FGN::new(0.75, 1599, Some(1.0), None);
    let fou = FOU::new(3.0, 5.0, 3.0, 1600, Some(X0), Some(16.0), None, fgn);
    let path = fou.sample();
    let mut estimator = FOUParameterEstimation::new(path, 1.0, FilterType::Daubechies);

    // Estimate the parameters
    let (estimated_hurst, estimated_sigma, estimated_mu, estimated_theta) =
      estimator.estimate_parameters();

    // Print the estimated parameters
    println!("Estimated Hurst exponent: {}", estimated_hurst);
    println!("Estimated sigma: {}", estimated_sigma);
    println!("Estimated mu: {}", estimated_mu);
    println!("Estimated theta: {}", estimated_theta);

    // Assert that the estimated parameters are close to the original ones
    let tolerance = 0.1; // Adjust tolerance as needed

    assert!((estimated_hurst - 0.75).abs() < tolerance);
    assert!((estimated_sigma - 2.0).abs() < tolerance);
    assert!((estimated_mu - 3.0).abs() < tolerance);
    assert!((estimated_theta - 2.0).abs() < tolerance);
  }
}
