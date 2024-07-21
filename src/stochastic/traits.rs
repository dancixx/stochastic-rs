use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rayon::prelude::*;

pub trait Process: Send + Sync {
  fn drift(&self, x: f64, t: f64) -> f64;

  fn diffusion(&self, x: f64, t: f64) -> f64;

  fn sample(&self, n: usize, x_0: f64, t_0: f64, t: f64) -> Array1<f64> {
    let dt = (t - t_0) / n as f64;
    let mut x = Array1::zeros(n);
    x[0] = x_0;
    let noise = Array1::random(n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let times = Array1::linspace(t_0, t, n);

    noise.into_iter().enumerate().for_each(|(idx, dw)| {
      x[idx + 1] =
        x[idx] + self.drift(x[idx], times[idx]) * dt + self.diffusion(x[idx], times[idx]) * dw;
    });

    x
  }

  fn sample_as_vec(&self, n: usize, x_0: f64, t_0: f64, t: f64) -> Vec<f64> {
    self.sample(n, x_0, t_0, t).to_vec()
  }

  fn sample_parallel(&self, m: usize, n: usize, x_0: f64, t_0: f64, t: f64) -> Array2<Array1<f64>> {
    let mut xs = Array2::<Array1<f64>>::default((m, n));

    xs.par_iter_mut().for_each(|x| {
      *x = self.sample(n, x_0, t_0, t);
    });

    xs
  }

  fn sample_parallel_as_vec(
    &self,
    m: usize,
    n: usize,
    x_0: f64,
    t_0: f64,
    t: f64,
  ) -> Vec<Vec<f64>> {
    self
      .sample_parallel(m, n, x_0, t_0, t)
      .into_par_iter()
      .map(|x| x.to_vec())
      .collect()
  }
}

#[cfg(feature = "f32")]
pub trait ProcessF32: Send + Sync {
  #[cfg(feature = "f32")]
  fn drift(&self, x: f32, t: f32) -> f32;

  #[cfg(feature = "f32")]
  fn diffusion(&self, x: f32, t: f32) -> f32;

  #[cfg(feature = "f32")]
  fn sample(&self, n: usize, x_0: f32, t_0: f32, t: f32) -> Array1<f32> {
    let dt = (t - t_0) / n as f32;
    let mut x = Array1::zeros(n);
    x[0] = x_0;
    let noise = Array1::random(n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let times = Array1::linspace(t_0, t, n);

    noise.into_iter().enumerate().for_each(|(idx, dw)| {
      x[idx + 1] =
        x[idx] + self.drift(x[idx], times[idx]) * dt + self.diffusion(x[idx], times[idx]) * dw;
    });

    x
  }

  #[cfg(feature = "f32")]
  fn sample_as_vec(&self, n: usize, x_0: f32, t_0: f32, t: f32) -> Vec<f32> {
    self.sample(n, x_0, t_0, t).to_vec()
  }

  #[cfg(feature = "f32")]
  fn sample_parallel(&self, m: usize, n: usize, x_0: f32, t_0: f32, t: f32) -> Array2<Array1<f32>> {
    let mut xs = Array2::<Array1<f32>>::default((m, n));

    xs.par_iter_mut().for_each(|x| {
      *x = self.sample(n, x_0, t_0, t);
    });

    xs
  }

  #[cfg(feature = "f32")]
  fn sample_parallel_as_vec(
    &self,
    m: usize,
    n: usize,
    x_0: f32,
    t_0: f32,
    t: f32,
  ) -> Vec<Vec<f32>> {
    self
      .sample_parallel(m, n, x_0, t_0, t)
      .into_par_iter()
      .map(|x| x.to_vec())
      .collect()
  }
}
