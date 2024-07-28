use std::time::Instant;

use plotly::{common::Line, Plot, Scatter};
use rand_distr::Normal;
use stochastic_rs::jumps::bates::{bates_1996, Bates1996};

fn main() {
  let start = Instant::now();

  let mut plot = Plot::new();

  let jump_distr = Normal::new(0.0, 0.07).unwrap();
  let params = Bates1996 {
    mu: Some(0.05),
    b: Some(0.05),
    r: None,
    r_f: None,
    lambda: 0.01,
    k: 0.,
    alpha: 0.0225,
    beta: 4.0,
    sigma: 0.1,
    rho: 0.1,
    n: 100,
    s0: Some(40.),
    v0: Some(0.025),
    t: Some(25.0),
    use_sym: Some(false),
  };
  println!("{:?}", params);
  for _ in 0..10 {
    let [s, _v] = bates_1996(&params, jump_distr);

    let trace = Scatter::new((0..s.len()).collect::<Vec<_>>(), s.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Bates");
    plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
