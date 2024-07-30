use std::time::Instant;

use plotly::{common::Line, Plot, Scatter};
use rand_distr::Gamma;
use stochastic_rs::jumps::jump_fou::{jump_fou, JumpFou};

fn main() {
  let start = Instant::now();

  let mut plot = Plot::new();

  let jump_distr = Gamma::new(2.0, 0.07).unwrap();
  let params = JumpFou {
    hurst: 0.7,
    mu: 1.0,
    sigma: 4.0,
    theta: 2.0,
    lambda: 5.25,
    n: 1000,
    x0: Some(0.0),
    t: Some(10.0),
  };

  for _ in 0..1 {
    let s = jump_fou(&params, jump_distr);

    let trace = Scatter::new((0..s.len()).collect::<Vec<_>>(), s.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Spline),
      )
      .name("Bates");
    plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
