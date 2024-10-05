use std::sync::Mutex;

use ndarray::{s, Array1};
use statrs::function::gamma;

use crate::stochastic::{noise::fgn::FGN, Sampling};

#[derive(Default)]
pub struct Fbm {
  pub hurst: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
  pub calculate_malliavin: Option<bool>,
  malliavin: Mutex<Option<Array1<f64>>>,
}

impl Fbm {
  pub fn new(params: &Self) -> Self {
    if !(0.0..=1.0).contains(&params.hurst) {
      panic!("Hurst parameter must be in (0, 1)")
    }

    let fgn = FGN::new(params.hurst, params.n, params.t, None);

    Self {
      hurst: params.hurst,
      t: params.t,
      n: params.n,
      m: params.m,
      fgn,
      calculate_malliavin: Some(false),
      malliavin: Mutex::new(None),
    }
  }
}

impl Sampling<f64> for Fbm {
  fn sample(&self) -> Array1<f64> {
    let fgn = self.fgn.sample();
    let mut fbm = Array1::<f64>::zeros(self.n + 1);
    fbm.slice_mut(s![1..]).assign(&fgn);

    for i in 1..=self.n {
      fbm[i] += fbm[i - 1];
    }

    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut malliavin = Array1::zeros(self.n + 1);
      let dt = self.t.unwrap_or(1.0) / self.n as f64;
      for i in 1..=self.n {
        malliavin[i] = 1.0 / (gamma::gamma(self.hurst + 0.5)) * dt.powf(self.hurst - 0.5);
      }

      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }
    fbm.slice(s![..self.n()]).to_owned()
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Calculate the Malliavin derivative
  ///
  /// The Malliavin derivative of the fractional Brownian motion is given by:
  /// D_s B^H_t = 1 / Γ(H + 1/2) (t - s)^{H - 1/2}
  ///
  /// where B^H_t is the fractional Brownian motion with Hurst parameter H in Mandelbrot-Van Ness representation as
  /// B^H_t = 1 / Γ(H + 1/2) ∫_0^t (t - s)^{H - 1/2} dW_s
  /// which is a truncated Wiener integral.
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().clone().unwrap()
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Axis;
  use plotly::{common::Line, Plot, Scatter};

  use super::*;

  #[test]
  fn plot() {
    let fbm = Fbm::new(&Fbm {
      hurst: 0.9,
      n: 1000,
      t: Some(1.0),
      m: Some(1),
      ..Default::default()
    });

    let mut plot = Plot::new();
    let d = fbm.sample_par();
    for data in d.axis_iter(Axis(0)) {
      let trace = Scatter::new((0..data.len()).collect::<Vec<_>>(), data.to_vec())
        .mode(plotly::common::Mode::Lines)
        .line(
          Line::new()
            .color("orange")
            .shape(plotly::common::LineShape::Linear),
        )
        .name("Fbm");
      plot.add_trace(trace);
    }
    plot.show();
  }
}
