use std::error::Error;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum NoiseGenerationMethod {
  Cholesky,
  Fft,
}

pub trait Generator: Sync + Send {
  fn sample(&self) -> Vec<f64>;
  fn sample_par(&self) -> Vec<Vec<f64>>;
}

pub trait Exporter: Sync + Send {
  fn to_csv(&self, path: &str) -> Result<(), Box<dyn Error>>;
  fn to_csv_par(&self, path: &str) -> Result<(), Box<dyn Error>>;
}
