use crate::quant::noises::fgn::FGN;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

pub trait FractionalProcess<T>: Send + Sync {
  fn drift(&self, x: T, t: T) -> T;
  fn diffusion(&self, x: T, t: T) -> T;
  fn hurst(&self) -> T;
  fn fgn(&self) -> FGN<T>;
  fn params(&self) -> (usize, T, T, T);
}

pub trait SamplingF<T> {
  fn sample(&self) -> Array1<T>;
  fn sample_as_vec(&self) -> Vec<T>;
  fn sample_parallel(&self, m: usize) -> Array2<T>;
  fn sample_parallel_as_vec(&self, m: usize) -> Vec<Vec<T>>;
}

impl<T: FractionalProcess<f64>> SamplingF<f64> for T {
  fn sample(&self) -> Array1<f64> {
    let (n, x_0, t_0, t) = self.params();
    let dt = (t - t_0) / n as f64;
    let mut x = Array1::zeros(n);
    x[0] = x_0;
    let noise = self.fgn().sample();
    let times = Array1::linspace(t_0, t, n);

    // TODO: test idx
    noise.into_iter().enumerate().for_each(|(idx, dw)| {
      x[idx + 1] =
        x[idx] + self.drift(x[idx], times[idx]) * dt + self.diffusion(x[idx], times[idx]) * dw;
    });

    x
  }

  fn sample_as_vec(&self) -> Vec<f64> {
    self.sample().to_vec()
  }

  fn sample_parallel(&self, m: usize) -> Array2<f64> {
    let (n, ..) = self.params();
    let mut xs = Array2::<f64>::zeros((m, n));

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&self.sample());
    });

    xs
  }

  fn sample_parallel_as_vec(&self, m: usize) -> Vec<Vec<f64>> {
    self
      .sample_parallel(m)
      .axis_iter(Axis(0))
      .into_par_iter()
      .map(|x| x.to_vec())
      .collect()
  }
}

#[cfg(feature = "f32")]
impl<T: FractionalProcess<f32>> SamplingF<f32> for T {
  fn sample(&self) -> Array1<f32> {
    let (n, x_0, t_0, t) = self.params();
    let dt = (t - t_0) / n as f32;
    let mut x = Array1::zeros(n);
    x[0] = x_0;
    let noise = self.fgn().sample();
    let times = Array1::linspace(t_0, t, n);

    noise.into_iter().enumerate().for_each(|(idx, dw)| {
      x[idx + 1] =
        x[idx] + self.drift(x[idx], times[idx]) * dt + self.diffusion(x[idx], times[idx]) * dw;
    });

    x
  }

  fn sample_as_vec(&self) -> Vec<f32> {
    self.sample().to_vec()
  }

  fn sample_parallel(&self, m: usize) -> Array2<f32> {
    let (n, ..) = self.params();
    let mut xs = Array2::<f32>::zeros((m, n));

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&self.sample());
    });

    xs
  }

  fn sample_parallel_as_vec(&self, m: usize) -> Vec<Vec<f32>> {
    self
      .sample_parallel(m)
      .axis_iter(Axis(0))
      .into_par_iter()
      .map(|x| x.to_vec())
      .collect()
  }
}
