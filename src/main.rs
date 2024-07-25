use std::time::Instant;

use plotly::{common::Line, Plot, Scatter};
use rand_distr::{Gamma, Normal};
use stochastic_rs::{
  jumps::{bates::bates_1996, jump_fou::jump_fou, levy_diffusion::levy_diffusion, merton::merton},
  processes::{cpoisson::compound_poisson, fbm::Fbm, poisson::poisson},
  utils::Generator,
};

fn main() {
  let start = Instant::now();

  let mut plot = Plot::new();
  let fbm = Fbm::new(0.9, 1000, Some(1.0), Some(10));

  for i in 0..1 {
    // let d = poisson(10.0, Some(50), None);
    // let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), d.to_vec())
    //   .mode(plotly::common::Mode::Lines)
    //   .line(
    //     Line::new()
    //       .color("orange")
    //       .shape(plotly::common::LineShape::Hv),
    //   )
    //   .name("Poisson");
    //plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
