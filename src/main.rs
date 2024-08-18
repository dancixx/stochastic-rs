use std::time::Instant;

use plotly::{common::Line, Plot, Scatter};
use rand_distr::{Gamma, Normal, StandardNormal};
use stochastic_rs::{
  diffusions::fou::{fou, Fou},
  jumps::{
    bates::{bates_1996, Bates1996},
    jump_fou::{jump_fou, JumpFou},
  },
  processes::fbm::Fbm,
  utils::Generator,
};

fn main() {
  let start = Instant::now();

  let mut plot = Plot::new();

  let jdistr = Gamma::new(1.45, 2.43).unwrap();
  let jtdistr = Gamma::new(1.45, 2.43).unwrap();
  let params = JumpFou {
    hurst: 0.7,
    mu: 4.0,
    sigma: 1.0,
    theta: 2.0,
    n: 1000,
    lambda: Some(0.5),
    x0: Some(0.0),
    t: Some(1.0),
  };

  // let bates_params = Bates1996 {
  //   mu: Some(0.0),
  //   b: Some(0.0),
  //   r: Some(0.0),
  //   r_f: Some(0.0),
  //   lambda: 0.5,
  //   k: 0.5,
  //   alpha: 0.5,
  //   beta: 0.5,
  //   sigma: 0.5,
  //   rho: 0.5,
  //   n: 1000,
  //   s0: Some(0.0),
  //   v0: Some(0.0),
  //   t: Some(1.0),
  //   use_sym: Some(false),
  // };
  for _ in 0..1 {
    let s = jump_fou(&params, StandardNormal);
    let fbm = Fbm::new(0.95, 200, Some(200.0), None);
    //let bates = bates_1996(&bates_params, StandardNormal);
    //let f = fbm.sample();

    let trace = Scatter::new((0..s.len()).collect::<Vec<_>>(), s.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Bates");
    plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
