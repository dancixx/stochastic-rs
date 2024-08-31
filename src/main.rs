use std::time::Instant;

use stochastic_rs::{processes::fbm::Fbm, utils::Generator};

fn main() {
  let start = Instant::now();
  let fbm = Fbm::new(0.95, 10000, Some(1.0), None);
  (0..1000).for_each(|_| {
    fbm.sample();
  });
  println!("{}", start.elapsed().as_secs_f64());

  let start = Instant::now();
  (0..1000).for_each(|_| {
    fbm.sample();
  });
  println!("{}", start.elapsed().as_secs_f64());

  let start = Instant::now();
  (0..1000).for_each(|_| {
    fbm.sample();
  });
  println!("{}", start.elapsed().as_secs_f64());

  let start = Instant::now();
  (0..1000).for_each(|_| {
    fbm.sample();
  });
  println!("{}", start.elapsed().as_secs_f64());

  let start = Instant::now();
  (0..1000).for_each(|_| {
    fbm.sample();
  });
  println!("{}", start.elapsed().as_secs_f64());
}
