use std::time::Instant;

use stochastic_rs::{noises::fgn, utils::Generator};

fn main() {
  let m = 10000;
  let hurst = 0.5;
  let n = 2048;
  let fgn = fgn::FgnFft::new(hurst, n, None, None);

  let start = Instant::now();
  for _ in 0..m {
    fgn.sample();
  }
  println!("elasped {}", start.elapsed().as_secs_f64());
}
