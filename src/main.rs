use std::time::Instant;

use indicatif::ProgressBar;
use plotly::{Plot, Scatter};
use stochastic_rs::{prelude::*, processes::fbm::Fbm, statistics::fractal_dim::higuchi_fd};

fn main() {
  let start = Instant::now();
  let fbm = Fbm::new(0.7, 10000, None, Some(10000), None);
  let m = 10000;
  let pb = ProgressBar::new(m);
  // let mut plot = Plot::new();
  for _ in 0..m {
    let path = fbm.sample();
    // let h = higuchi_fd(&path, 10);
    // println!("Higuchi FD: {}", 2.0 - h);
    // plot.add_trace(Scatter::new((0..5000).collect::<Vec<usize>>(), path));
    // plot.show();
    pb.inc(1);
  }
  pb.finish();
  println!("Time elapsed: {:?}", start.elapsed().as_secs_f64());

  // let start = Instant::now();
  // let _ = par_fbm(10000, 0.7, 2500, None, None);
  // println!("Time elapsed: {:?}", start.elapsed());

  // let start = Instant::now();
  // let _ = Fbm::new(
  //   0.7,
  //   5000,
  //   None,
  //   Some(10000),
  //   Some(NoiseGenerationMethod::Fft(
  //     FractionalNoiseGenerationMethod::DaviesHarte,
  //   )),
  // )
  // .sample_par();
  // println!("Time elapsed: {:?}", start.elapsed());

  // CIR
  // let mut plot = Plot::new();
  // let path = || cir::cir(1.0, 0.6, 0.5, 1000, None, None, None);

  // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // FCIR
  // let mut plot = Plot::new();
  // let path = || cir::cir(1.0, 1.2, 0.2, 1000, None, None, None);

  // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // OU
  // let mut plot = Plot::new();
  // let path = || ou::ou(1.0, 0.2, 3.0, 1000, None, None);

  // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // FOU
  // let mut plot = Plot::new();
  // let path = || ou::fou(0.7, 2.0, 0.2, 3.0, 10000, None, None);

  // plot.add_trace(Scatter::new((0..10000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // GBM
  // let mut plot = Plot::new();
  // let path = || gbm::gbm(0.1, 0.3, 200, None, None);

  // plot.add_trace(Scatter::new((0..200).collect::<Vec<usize>>(), path()));
  // plot.show();

  // FGBM
  // let mut plot = Plot::new();
  // let path = || gbm::fgbm(0.2, 0.1, 0.3, 1000, None, None, None);

  // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // Jacobi
  // let mut plot = Plot::new();
  // let path = || jacobi::jacobi(1.0, 1.2, 0.2, 1000, None, None);

  // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // FJacobi
  // let mut plot = Plot::new();
  // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

  // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
  // plot.show();

  // Variance Gamma
}
