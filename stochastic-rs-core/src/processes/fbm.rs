use crate::{noises::fgn::Fgn, Sampling};
use ndarray::{s, Array1};

pub struct Fbm {
  #[allow(dead_code)]
  hurst: f64,
  #[allow(dead_code)]
  n: usize,
  m: Option<usize>,
  fgn: Option<Fgn>,
}

impl Fbm {
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be in (0, 1)")
    }

    Self {
      hurst,
      n,
      m,
      fgn: Some(Fgn::new(hurst, n - 1, t, None)),
    }
  }
}

impl Sampling<f64> for Fbm {
  fn sample(&self) -> Array1<f64> {
    let fgn = self.fgn.as_ref().unwrap().sample();

    let mut fbm = Array1::<f64>::zeros(self.n);
    fbm.slice_mut(s![1..]).assign(&fgn);

    for i in 1..self.n {
      fbm[i] += fbm[i - 1];
    }

    fbm
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Axis;
  use plotly::{common::Line, Plot, Scatter};

  use super::*;

  #[test]
  fn plot() {
    let fbm = Fbm::new(0.45, 1000, Some(1.0), Some(10));
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
