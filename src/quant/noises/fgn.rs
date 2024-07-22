use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndrustfft::{ndfft_par, FftHandler};
use num_complex::{Complex, ComplexDistribution};

#[derive(Clone)]
pub struct FGN<T> {
  hurst: T,
  n: usize,
  offset: usize,
  t: T,
  sqrt_eigenvalues: Array1<Complex<T>>,
  fft_handler: FftHandler<T>,
  fft_fgn: Array1<Complex<T>>,
}

impl FGN<f64> {
  #[must_use]
  #[inline(always)]
  pub fn new(hurst: f64, n: usize, t: f64) -> Self {
    if !(0_f64..=1_f64).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }
    let n_ = n.next_power_of_two();
    let offset = n_ - n;
    let n = n_;
    let mut r = Array1::linspace(0_f64, n as f64, n + 1);
    r.par_mapv_inplace(|x| {
      if x == 0_f64 {
        1_f64
      } else {
        0.5
          * ((x + 1_f64).powf(2_f64 * hurst) - 2_f64 * x.powf(2_f64 * hurst)
            + (x - 1_f64).powf(2_f64 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, 0_f64));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f64>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.par_mapv_inplace(|x| Complex::new((x.re / (2_f64 * n as f64)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t,
      sqrt_eigenvalues,
      fft_handler: FftHandler::new(2 * n),
      fft_fgn: Array1::<Complex<f64>>::zeros(2 * n),
    }
  }

  pub fn sample(&self) -> Array1<f64> {
    let rnd = Array1::<Complex<f64>>::random(
      2 * self.n,
      ComplexDistribution::new(StandardNormal, StandardNormal),
    );
    let fgn = &self.sqrt_eigenvalues * &rnd;
    let fft_handler = self.fft_handler.clone();
    let mut fgn_fft = self.fft_fgn.clone();
    ndfft_par(&fgn, &mut fgn_fft, &fft_handler, 0);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x: Complex<f64>| (x.re * (self.n as f64).powf(-self.hurst)) * self.t.powf(self.hurst));
    fgn
  }
}

#[cfg(feature = "f32")]
impl FGN<f32> {
  #[must_use]
  #[inline(always)]
  pub fn new_f32(hurst: f32, n: usize, t: f32) -> Self {
    if !(0_f32..=1_f32).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }
    let n_ = n.next_power_of_two();
    let offset = n_ - n;
    let n = n_;
    let mut r = Array1::linspace(0_f32, n as f32, n + 1);
    r.par_mapv_inplace(|x| {
      if x == 0_f32 {
        1_f32
      } else {
        0.5
          * ((x + 1_f32).powf(2_f32 * hurst) - 2_f32 * x.powf(2_f32 * hurst)
            + (x - 1_f32).powf(2_f32 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, 0_f32));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f32>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.par_mapv_inplace(|x| Complex::new((x.re / (2_f32 * n as f32)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t,
      sqrt_eigenvalues,
      fft_handler: FftHandler::new(2 * n),
      fft_fgn: Array1::<Complex<f32>>::zeros(2 * n),
    }
  }

  pub fn sample(&self) -> Array1<f32> {
    let rnd = Array1::<Complex<f32>>::random(
      2 * self.n,
      ComplexDistribution::new(StandardNormal, StandardNormal),
    );
    let fgn = &self.sqrt_eigenvalues * &rnd;
    let fft_handler = self.fft_handler.clone();
    let mut fgn_fft = self.fft_fgn.clone();
    ndfft_par(&fgn, &mut fgn_fft, &fft_handler, 0);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x: Complex<f32>| (x.re * (self.n as f32).powf(-self.hurst)) * self.t.powf(self.hurst));
    fgn
  }
}
