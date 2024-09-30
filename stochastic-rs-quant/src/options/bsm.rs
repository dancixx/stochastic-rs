use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::OptionType;

#[derive(Default, Debug, Clone, Copy)]
pub enum BsmCoc {
  /// Black-Scholes-Merton 1973 (stock option)
  /// Cost of carry = risk-free rate
  #[default]
  BSM1973,
  /// Black-Scholes-Merton 1976 (stock option)
  /// Cost of carry = risk-free rate - dividend yield
  MERTON1973,
  /// Black 1976 (futures option)
  /// Cost of carry = 0
  BLACK1976,
  /// Asay 1982 (futures option)
  /// Cost of carry = 0
  ASAY1982,
  /// Garman-Kohlhagen 1983 (currency option)
  /// Cost of carry = (domestic - foregin) risk-free rate
  GARMAN1983,
}

/// Black-Scholes-Merton model
#[derive(Default, Debug)]
pub struct Bsm {
  /// Underlying price
  pub s: f64,
  /// Volatility
  pub v: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Domestic risk-free rate
  pub r_d: Option<f64>,
  /// Foreign risk-free rate
  pub r_f: Option<f64>,
  /// Dividend yield
  pub q: Option<f64>,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
  /// Option type
  pub option_type: OptionType,
  /// Cost of carry
  pub b: BsmCoc,
}

impl Bsm {
  /// Create a new BSM model
  #[must_use]
  pub fn new(params: &Self) -> Self {
    if params.tau.is_none() && params.eval.is_none() && params.expiration.is_none() {
      panic!("At least one of the following parameters is missing: tau, eval, expiration");
    }

    let tau = if let Some(tau) = params.tau {
      tau
    } else {
      let eval = params
        .eval
        .unwrap_or(chrono::Local::now().naive_local().into());
      let expiration = params.expiration.unwrap();
      let tau = (expiration - eval).num_days() as f64 / 365.0;
      tau
    };

    Self {
      s: params.s,
      v: params.v,
      k: params.k,
      r: params.r,
      r_d: params.r_d,
      r_f: params.r_f,
      q: params.q,
      tau: Some(tau),
      eval: params.eval,
      expiration: params.expiration,
      option_type: params.option_type,
      b: params.b,
    }
  }

  /// Calculate the option price
  #[must_use]
  pub fn price(&mut self) -> f64 {
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();
    let tau = self.tau();

    if self.option_type == OptionType::Call {
      self.s * ((self.b() - self.r) * tau).exp() * n.cdf(d1)
        - self.k * (-self.r * tau).exp() * n.cdf(d2)
    } else {
      -self.s * ((self.b() - self.r) * tau).exp() * n.cdf(-d1)
        + self.k * (-self.r * tau).exp() * n.cdf(-d2)
    }
  }

  /// Calculate d1
  fn d1_d2(&self) -> (f64, f64) {
    let d1 = (1.0 / (self.v * self.tau().sqrt()))
      * ((self.s / self.k).ln() + (self.b() + 0.5 * self.v.powi(2)) * self.tau());
    let d2 = d1 - self.v * self.tau().sqrt();

    (d1, d2)
  }

  /// Calculate b (cost of carry)
  fn b(&self) -> f64 {
    match self.b {
      BsmCoc::BSM1973 => self.r,
      BsmCoc::MERTON1973 => self.r - self.q.unwrap(),
      BsmCoc::BLACK1976 => 0.0,
      BsmCoc::ASAY1982 => 0.0,
      BsmCoc::GARMAN1983 => self.r_d.unwrap() - self.r_f.unwrap(),
    }
  }

  /// Calculate the delta
  pub fn delta(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let n = Normal::default();
    let tau = self.tau();
    let exp_bt = ((self.b() - self.r) * tau).exp();

    if self.option_type == OptionType::Call {
      exp_bt * n.cdf(d1)
    } else {
      exp_bt * (n.cdf(d1) - 1.0)
    }
  }

  /// Calculate the gamma
  pub fn gamma(&self) -> f64 {
    let T = self.tau();
    let (d1, _) = self.d1_d2();
    let n = Normal::default();

    ((self.b() - self.r) * T).exp() * n.pdf(d1) / (self.s * self.v * self.tau().sqrt())
  }

  /// Calculate the gamma percent
  pub fn gamma_percent(&self) -> f64 {
    self.gamma() / self.s * 100.0
  }

  /// Calculate the theta
  pub fn theta(&self) -> f64 {
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_bt = ((self.b() - self.r) * self.tau()).exp();
    let exp_rt = (-self.r * self.tau()).exp();
    let pdf_d1 = n.pdf(d1);

    let first_term = -self.s * exp_bt * pdf_d1 * self.v / (2.0 * self.tau().sqrt());

    if self.option_type == OptionType::Call {
      let second_term = -(self.b() - self.r) * self.s * exp_bt * n.cdf(d1);
      let third_term = -self.r * self.k * exp_rt * n.cdf(d2);
      first_term + second_term + third_term
    } else {
      let second_term = (self.b() - self.r) * self.s * exp_bt * n.cdf(-d1);
      let third_term = -self.r * self.k * exp_rt * n.cdf(-d2);
      first_term + second_term + third_term
    }
  }

  /// Calculate the vega
  pub fn vega(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let n = Normal::default();

    self.s * ((self.b() - self.r) * self.tau()).exp() * n.pdf(d1) * self.tau().sqrt()
  }

  /// Calculate the rho
  pub fn rho(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_rt = (-self.r * self.tau()).exp();

    if self.option_type == OptionType::Call {
      self.k * self.tau() * exp_rt * n.cdf(d2)
    } else {
      -self.k * self.tau() * exp_rt * n.cdf(-d2)
    }
  }

  /// Calculate the vomma
  pub fn vomma(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.vega() * d1 * d2 / self.v
  }

  /// Calculate the charm
  pub fn charm(&self) -> f64 {
    let v = self.v;
    let r = self.r;
    let b = self.b();
    let tau = self.tau();
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_bt = ((b - r) * tau).exp();
    let pdf_d1 = n.pdf(d1);
    let sqrt_T = tau.sqrt();

    match self.option_type {
      OptionType::Call => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_T)) - (d2 / (2.0 * tau))) + (b - r) * n.cdf(d1))
      }
      OptionType::Put => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_T)) - (d2 / (2.0 * tau))) - (b - r) * n.cdf(-d1))
      }
    }
  }

  /// Calculate the vanna
  pub fn vanna(&self) -> f64 {
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();

    -((self.b() - self.r) * self.tau()).exp() * n.pdf(d1) * d2 / self.v
  }

  /// Calculate the zomma
  pub fn zomma(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.gamma() * (d1 * d2 - 1.0) / self.v
  }

  /// Calculate the zomma percent
  pub fn zomma_percent(&self) -> f64 {
    self.zomma() * self.s / 100.0
  }

  /// Calculate the speed
  pub fn speed(&self) -> f64 {
    let (d1, _) = self.d1_d2();

    -self.gamma() * (1.0 + d1 / (self.v * self.tau().sqrt())) / self.s
  }

  /// Calculate the color
  pub fn color(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.gamma()
      * (self.r - self.b()
        + self.b() * d1 / (self.v * self.tau().sqrt())
        + (1.0 - d1 * d2) / (2.0 * self.tau()))
  }

  /// Calculate the ultima
  pub fn ultima(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    -self.vomma() / self.v * (d1 * d2 - (d1 / d2) + (d2 / d1) - 1.0)
  }

  /// Calculate the DvegaDtime
  pub fn dvega_dtime(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.vega()
      * (self.r - self.b() + self.b() * d1 / (self.v * self.tau().sqrt())
        - (d1 * d2 + 1.0) / (2.0 * self.tau()))
  }

  /// Calculating Lambda (elasticity)
  pub fn lambda(&mut self) -> f64 {
    self.delta() * self.s / self.price()
  }

  /// Calculate the phi
  pub fn phi(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let n = Normal::default();

    let exp_bt = ((self.b() - self.r) * self.tau()).exp();

    if self.option_type == OptionType::Call {
      -self.tau() * self.s * exp_bt * n.cdf(d1)
    } else {
      self.tau() * self.s * exp_bt * n.cdf(-d1)
    }
  }

  /// Calculate the zeta
  pub fn zeta(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    if self.option_type == OptionType::Call {
      n.cdf(d2)
    } else {
      -n.cdf(-d2)
    }
  }

  /// Calculate the strike delta
  pub fn strike_delta(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_rt = (-self.r * self.tau()).exp();

    if self.option_type == OptionType::Call {
      -exp_rt * n.cdf(d2)
    } else {
      exp_rt * n.cdf(-d2)
    }
  }

  /// Calculate the strike gamma
  pub fn strike_gamma(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    n.pdf(d2) * (-self.r * self.tau()).exp() / (self.k * self.v * self.tau().sqrt())
  }

  fn tau(&self) -> f64 {
    self.tau.unwrap()
  }
}
