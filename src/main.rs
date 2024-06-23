use std::time::Instant;

use stochastic_rs::{
  noises::fgn, processes::fbm, statistics::fractal_dim::higuchi_fd, utils::Generator,
};

fn main() {
  let m = 20000;
  let hurst = 0.1;
  let n = 50000;
  let fgn = fbm::Fbm::new(hurst, n, None, None);

  let start = Instant::now();
  for _ in 0..m {
    let data = fgn.sample();
    let h = 2.0 - higuchi_fd(data.as_slice(), 10);
    println!("{}", h);
  }
  println!("elasped {}", start.elapsed().as_secs_f64());
}
